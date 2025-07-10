import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

from models.tc_attention import TCAttention

def patch_llava_with_tcattention(frame_size=576, gamma=0.5, lora=True):
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
    num_layers = len(model.model.language_model.layers)

    for i in range(num_layers - 5, num_layers):
        blk = model.model.language_model.layers[i]
        orig_attn = blk.self_attn
        blk.self_attn = TCAttention(orig_attn, frame_size=576, gamma=0.5)
    
    return model, processor
