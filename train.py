# Main training script
import os
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

from utils import convert_to_conversation
from models.patch_llava import patch_llava_with_mivc_tcattention

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

instruction = """
  What is happening?
"""

samples = [
    {
        "image": ["./data/train/000000039769.jpg"],
        "text": "A person is cooking."
    } for _ in range(20)
]

def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = True if device == "cuda" else False
    
    train_conversation_dataset = Dataset.from_list([convert_to_conversation(sample, instruction) for sample in samples])
    validation_conversation_dataset = Dataset.from_list([convert_to_conversation(sample, instruction) for sample in samples])

    model, processor = patch_llava_with_mivc_tcattention(frame_size=576, gamma=0.5, mivc_dim=1024)

    # Prepare SFTConfig
    training_args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        num_train_epochs = 30,
        max_seq_length=512,
        output_dir="/workspace/temporal_llava_project/output",
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        learning_rate = 2e-4,
        # report_to = "tensorboard",
        eval_strategy="steps",
        eval_steps=1,
        do_eval=True,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        seed = 3407,
    )

    # Prepare SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_conversation_dataset,
        eval_dataset = validation_conversation_dataset,
        args=training_args
    )
    
    trainer.add_callback(EarlyStoppingCallback(
        early_stopping_patience=6
    ))

    # Train
    trainer.train()
    
    # Save the model
    model.save_pretrained("pasithbas159/TC_LLaVA_hydro_v0", push_to_hub=True, token=HUGGINGFACE_TOKEN)
    processor.save_pretrained("pasithbas159/TC_LLaVA_hydro_v0", push_to_hub=True, token=HUGGINGFACE_TOKEN)
    
if __name__ == '__main__':
    main()