import requests
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch

from models.patch_llava import patch_llava_with_tcattention

if __name__ == "__main__":
    
    model = LlavaForConditionalGeneration.from_pretrained("pasithbas159/TC_LLaVA_hydro_v0")
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
    inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print(processor.decode(output[0][2:], skip_special_tokens=True))
