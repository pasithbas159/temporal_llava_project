# Logging, scheduler, freezing utils

def setup_logging():
    pass
    # ...implementation...

def get_scheduler():
    pass
    # ...implementation...

def convert_to_conversation(sample, instruction):
    conversation = [
        { "role": "user",
          "content":
          [
              item
              for i in range(len(sample["image"]))
              for item in (
                  {"type": "image", "image": sample["image"][i]},
              )
          ] + # images placeholder
          [{"type": "text", "text": instruction}]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["text"]} ]
        },
    ]
    return { "messages" : conversation }

def compute_visual_token_mask(input_ids, image_token_id=32001):  # or your model’s image ID
        # Detect which tokens are image tokens
        return (input_ids == image_token_id)

class DataCollatorWithVisualMask:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # features is a list of dicts with 'input_ids' (already tokenized)
        # Pad input_ids to the same length
        import torch
        input_ids = [torch.tensor(f["input_ids"]) for f in features]
        batch_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        visual_token_mask = compute_visual_token_mask(batch_input_ids)
        return {"input_ids": batch_input_ids, "visual_token_mask": visual_token_mask}