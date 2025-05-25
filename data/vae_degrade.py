#%%
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
# from types import MethodType
import numpy as np
import os
# from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#%%
model_id = "peter-sushko/RealEdit"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, cache_dir = 'cache')
pipe.to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

#%%
def encode(pipe, image):
    # Load and preprocess the image
    peter_pixels = np.asarray(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    peter_tensor = torch.tensor(peter_pixels).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
    # Move to the same dtype and device as the VAE expects
    peter_tensor = peter_tensor.to(dtype=pipe.vae.dtype, device=pipe.device)
    # Encode
    encoded = pipe.vae.encode(peter_tensor)
    latent = encoded.latent_dist.sample()
    return latent

def decode(pipe, latent):
    decoded = pipe.vae.decode(latent)
    # 1. Get the image tensor (usually in decoded.sample)
    image_tensor = decoded.sample  # This is usually a tensor like (1, 3, H, W)
    # 2. Squeeze batch dimension and move to CPU
    image_tensor = image_tensor.squeeze(0).detach().cpu()
    # 3. Clamp values to [0, 1] to make it valid for display
    image_tensor = image_tensor.clamp(0, 1)
    # 4. Convert to numpy and then PIL image
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    return image_pil

#%%
"Degrade images with the VAE"

output_dir_gt = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/gt"
output_dir_lq = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/lq"
os.makedirs(output_dir_gt, exist_ok=True)
os.makedirs(output_dir_lq, exist_ok=True)

prefixes = [f"{i:02d}" for i in range(60, 70)]
for prefix in prefixes:
    print(f"Processing prefix: {prefix}")
    image_path = f"/mmfs1/gscratch/amath/vilin/BFRffusion/data/images1024x1024/{prefix}000"
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for i in trange(1000):
        try:
            img = Image.open(os.path.join(image_path, image_files[i])).convert("RGB")

            size = 512
            img_resized = ImageOps.pad(img, (size, size), method=Image.LANCZOS, color=(0, 0, 0))

            # Encode and decode
            latent = encode(pipe, img_resized)
            decoded_img = decode(pipe, latent)

            # Save images
            img_resized.save(os.path.join(output_dir_gt, f"{os.path.splitext(image_files[i])[0]}.png"))
            decoded_img.save(os.path.join(output_dir_lq, f"{os.path.splitext(image_files[i])[0]}.png"))
        except Exception as e:
            print(f"Error processing image {image_files[i]}: {e}")
            continue




#%%
# "Make edits"
# image_path = "my_images"
# image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# img = Image.open(os.path.join(image_path, image_files[0])).convert("RGB")
# img = img.resize((550, 550), Image.LANCZOS)

# seed=1
# generator = torch.Generator("cuda").manual_seed(seed)

# output = pipe(
#     prompt='give him a hat',
#     image=img,
#     num_inference_steps=50,
#     generator=generator,
#     image_guidance_scale=1.5,
#     guidance_scale=7
# )

# generated_image = output.images[0]
# display(generated_image)

