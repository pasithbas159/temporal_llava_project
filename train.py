# Main training script
import os
from torch.utils.data import DataLoader
import torch
from transformers import get_cosine_schedule_with_warmup
import tqdm

from data.dataset import SimpleDataset, make_collate_fn
from models.patch_llava import patch_llava_with_tcattention

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

train_data = [
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    ("./data/train/000000039769.jpg", "<image>\nWhat is happening?", "A person is cooking."),
    # Add more (image_path, prompt, target)
]

def main(train_data):
    
    model, processor = patch_llava_with_tcattention(frame_size=576, gamma=0.5, lora=True)
    
    collate_fn = make_collate_fn(processor)
    
    dataset = SimpleDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=100)
    
    # ==== Training Loop ====
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"🔁 Epoch {epoch+1}/{num_epochs}")
        pbar = tqdm.tqdm(dataloader)
        for inputs, labels in pbar:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # Cast trainable parameters to float32 before computing gradients
                for param in model.parameters():
                    if param.requires_grad:
                        param.data = param.data.to(torch.float32)
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            pbar.set_description(f"Loss: {loss.item():.4f}")
    
    model.save_pretrained("pasithbas159/TC_LLaVA_hydro_v0", push_to_hub=True, token=HUGGINGFACE_TOKEN)
    processor.save_pretrained("pasithbas159/TC_LLaVA_hydro_v0", push_to_hub=True, token=HUGGINGFACE_TOKEN)
    
if __name__ == "__main__":
    main(train_data)
