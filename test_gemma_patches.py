import torch
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained('google/gemma-3-4b-it')
print("Processor loaded")
from PIL import Image
import numpy as np
img = Image.fromarray(np.zeros((500, 500, 3), dtype=np.uint8))
messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "hello"}]}]
text = processor.apply_chat_template(messages, tokenize=False)
result = processor(text=[text], images=[img], return_tensors="pt")
input_ids = result['input_ids'][0]
img_tokens = (input_ids == processor.image_token_id).sum().item()
print("Number of image tokens:", img_tokens)
import math
print("sqrt:", math.sqrt(img_tokens))
