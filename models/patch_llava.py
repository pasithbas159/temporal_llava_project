import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

from models.tc_attention import TCAttention
from models.mivc_pooling import MIVCPooling

def patch_llava_with_mivc_tcattention(frame_size=576, gamma=0.5, mivc_dim=1024):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        # device_map="auto",
        device_map = device
    )
        
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    # ------------------------------------------
    # 🔹 Patch 1: Add MIVC pooling after vision encoder
    # ------------------------------------------
    vit = model.model.vision_tower
    frame_size = (vit.config.image_size // vit.config.patch_size) ** 2 + 1  # +1 for CLS
    vit_dim = model.model.vision_tower.config.hidden_size
    model.model.vision_mivc_pooling = MIVCPooling(in_dim=vit_dim, hidden_dim=mivc_dim)

    def encode_images_with_mivc(pixel_values):
        B, N, C, H, W = pixel_values.shape
        pixel_values = pixel_values.view(B * N, C, H, W)
        vit_outputs = model.model.vision_tower(pixel_values)  # full ViT forward
        vision_embeds = vit_outputs[:, 0]  # CLS token only, per image
        vision_embeds = vision_embeds.view(B, N, -1)
        pooled = model.model.vision_mivc_pooling(vision_embeds)  # <- MIVC applied here
        return pooled

    model.model.encode_images = encode_images_with_mivc
    
    # ------------------------------------------
    # 🔹 Patch 2: Replace last 5 layers of LLM with TCAttention
    # ------------------------------------------
    
    num_layers = len(model.model.language_model.layers)

    for i in range(num_layers - 5, num_layers):
        blk = model.model.language_model.layers[i]
        orig_attn = blk.self_attn
        blk.self_attn = TCAttention(orig_attn, frame_size=frame_size, gamma=0.5)
    
    # ------------------------------------------
    # 🔹 Set trainable parameters
    # ------------------------------------------
    
    # Freeze all
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze MIVC
    for param in model.model.vision_mivc_pooling.parameters():
        param.requires_grad = True

    # Unfreeze TCAttention
    for blk in model.model.language_model.layers:
        if isinstance(blk.self_attn, TCAttention):
            for param in blk.self_attn.parameters():
                param.requires_grad = True
                
    # Check trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✅ Trainable parameters: {trainable:,} / {total:,}")
    
    return model, processor
