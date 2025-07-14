import requests
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch

from models.patch_llava import patch_llava_with_mivc_tcattention
from utils import compute_visual_token_mask

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model, processor = patch_llava_with_mivc_tcattention(
        frame_size=576, gamma=0.5, mivc_dim=1024
    )
    
    model.from_pretrained("pasithbas159/TC_LLaVA_hydro_v0", device_map=device, torch_dtype=torch.float16, low_cpu_mem_usage=True,)
    model.eval()
    
    processor = AutoProcessor.from_pretrained("pasithbas159/TC_LLaVA_hydro_v0")

    conversation = [
        {

        "role": "user",
        "content": [
            {"type": "text", "text": "What are these?"},
            {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
    raw_image = Image.open(requests.get(image_file, stream=True).raw)
    # inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt')
    
    # Compute visual_token_mask and ensure shape matches input_ids
    visual_token_mask = compute_visual_token_mask(inputs["input_ids"])
    # Force shape to [batch_size, seq_len]
    # visual_token_mask = visual_token_mask.view(inputs["input_ids"].shape)
    visual_token_mask = visual_token_mask.view_as(inputs["input_ids"])
    inputs["visual_token_mask"] = visual_token_mask

    # Only pass required 2D tensors to model.generate
    gen_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "visual_token_mask": inputs["visual_token_mask"]
    }
    print(gen_inputs)
    output = model.generate(**gen_inputs, max_new_tokens=200, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))
