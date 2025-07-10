# Custom dataset and collate function
import requests
from PIL import Image
import torch

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data  # List of (image_path, prompt, target_text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, prompt, target = self.data[idx]
        image = Image.open(requests.get(image_path, stream=True).raw)
        return {"image": image, "prompt": prompt, "target": target}

def collate_fn(batch):
    images = [item["image"] for item in batch]
    prompts = [item["prompt"] for item in batch]
    targets = [item["target"] for item in batch]

    # Combine prompt + target as one input string (generation target)
    texts = [f"{p}\n{t}" for p, t in zip(prompts, targets)]

    # Tokenize with image + prompt+target
    inputs = processor(texts, images, padding=True, return_tensors="pt")
    inputs = {k: v.to("cuda", torch.float16) if v.dtype == torch.float else v.to("cuda") for k, v in inputs.items()}

    # Use input_ids as labels (same sequence) and mask out image/prompt tokens later
    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100  # ignore padding

    return inputs, labels
