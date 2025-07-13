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