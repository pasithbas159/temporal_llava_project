# Temporal LLaVA Project

This project implements a temporal-aware LLaVA model with TCAttention and LoRA.

## Structure
- `configs/`: Training configuration
- `models/`: Model modules
- `data/`: Dataset and collate function
- `train.py`: Training script
- `inference.py`: Inference/demo script
- `utils.py`: Utilities

## Usage
- Edit `configs/train_config.yaml` for training settings.
- Run `train.py` to train, `inference.py` for inference.
