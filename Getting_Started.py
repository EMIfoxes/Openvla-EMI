# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import torch

# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Load Processor & VLA
processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path="/media/lxx/Elements/project/openvla/checkpoint/openvla/openvla-7b", trust_remote_code=True) #openvla/openvla-7b
vla = AutoModelForVision2Seq.from_pretrained(
    "/media/lxx/Elements/project/openvla/checkpoint/openvla/openvla-7b", 
    # attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16, 
    low_cpu_mem_usage=True, 
    trust_remote_code=True
).to("cuda:0")

print("OK")

# # Grab image input & format prompt
# image: Image.Image = get_from_camera(...)
# prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"


image = Image.open('test.png')
# arr = (img)      # shape=(H, W, C)，RGB 顺序
prompt = "In: What action should the robot take to {pick up banana}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(action)
# Execute...
# robot.act(action, ...)