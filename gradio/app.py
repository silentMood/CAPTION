import os
import sys
import random
import numpy as np
import torch
from PIL import Image
import gradio as gr
from diffusers import DiffusionPipeline
from blip3o.conversation import conv_templates
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import get_model_name_from_path
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

pretrained_dir = os.path.join(os.getcwd(), 'pretrained')
processor = AutoProcessor.from_pretrained(os.path.join(pretrained_dir, 'qwen-7b'))

# Constants
MAX_SEED = 10000

model_path = os.path.join(pretrained_dir, 'blip3o-8b')
diffusion_path = model_path + "/diffusion-decoder"


def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def add_template(prompt_list: list[str]) -> str:
    conv = conv_templates['qwen'].copy()
    conv.append_message(conv.roles[0], prompt_list[0])
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()

def make_prompt(text: str) -> list[str]:
    raw = f"Please generate image based on the following caption: {text}"
    return [add_template([raw])]

def randomize_seed_fn(seed: int, randomize: bool) -> int:
    return random.randint(0, MAX_SEED) if randomize else seed

def generate_image(prompt: str, seed: int, guidance_scale: float, randomize: bool) -> list[Image.Image]:
    seed = randomize_seed_fn(seed, randomize)
    set_global_seed(seed)
    formatted = make_prompt(prompt)
    images = []
    for _ in range(4):
        out = pipe(formatted, guidance_scale=guidance_scale)
        images.append(out.image)
    return images

def process_image(prompt: str, img: Image.Image) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt},
        ],
    }]
    text_prompt_for_qwen = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt_for_qwen],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to('cuda:0')
    generated_ids = multi_model.generate(**inputs, max_new_tokens=1024)
    input_token_len = inputs.input_ids.shape[1]
    generated_ids_trimmed = generated_ids[:, input_token_len:]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    return output_text

# Initialize model + pipeline
disable_torch_init()
# model_path = os.path.expanduser(sys.argv[1])
tokenizer, multi_model, _ = load_pretrained_model(model_path)

pipe = DiffusionPipeline.from_pretrained(
    diffusion_path,
    custom_pipeline="pipeline_llava_gen",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    variant="bf16",
    multimodal_encoder=multi_model,
    tokenizer=tokenizer,
    safety_checker=None
)
pipe.vae.to('cuda:0')
pipe.unet.to('cuda:0')

if __name__ == "__main__":
    image_path = '/home/catjam/project/BLIP3o/gradio/animal-compare.png'
    img = Image.open(image_path)
    prompt = 'Please generate a textual description for this indoor equirectangular panoramic image, styled as a diffusion generation prompt, make it elegant and short.'
    txt = process_image(prompt, img)
