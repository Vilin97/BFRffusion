#%%
from PIL import Image, ImageOps
import torch
# from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
# from types import MethodType
import numpy as np
import os
# from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
import shutil
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

# output_dir_gt = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/gt"
# output_dir_lq = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/lq"
# os.makedirs(output_dir_gt, exist_ok=True)
# os.makedirs(output_dir_lq, exist_ok=True)

# prefixes = [f"{i:02d}" for i in range(60, 70)]
# for prefix in prefixes:
#     print(f"Processing prefix: {prefix}")
#     image_path = f"/mmfs1/gscratch/amath/vilin/BFRffusion/data/images1024x1024/{prefix}000"
#     image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     for i in trange(1000):
#         try:
#             img = Image.open(os.path.join(image_path, image_files[i])).convert("RGB")

#             size = 512
#             img_resized = ImageOps.pad(img, (size, size), method=Image.LANCZOS, color=(0, 0, 0))

#             # Encode and decode
#             latent = encode(pipe, img_resized)
#             decoded_img = decode(pipe, latent)

#             # Save images
#             img_resized.save(os.path.join(output_dir_gt, f"{os.path.splitext(image_files[i])[0]}.png"))
#             decoded_img.save(os.path.join(output_dir_lq, f"{os.path.splitext(image_files[i])[0]}.png"))
#         except Exception as e:
#             print(f"Error processing image {image_files[i]}: {e}")
#             continue


#%%
# "make the validation set"

# val_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/val/gt"
# val_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/val/lq"
# train_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/train/gt"
# train_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/train/lq"

# os.makedirs(val_gt_dir, exist_ok=True)
# os.makedirs(val_lq_dir, exist_ok=True)
# os.makedirs(train_gt_dir, exist_ok=True)
# os.makedirs(train_lq_dir, exist_ok=True)

# for subdir in ["gt", "lq"]:    
#     src_dir = f"/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/{subdir}"
#     files = [fname for fname in os.listdir(src_dir) if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     for fname in tqdm(files, desc=f"Copying {subdir} images"):
#         prefix = fname[:1]
#         if prefix == "6":
#             dst_dir = val_gt_dir if subdir == "gt" else val_lq_dir
#         else:
#             dst_dir = train_gt_dir if subdir == "gt" else train_lq_dir
#         src_path = os.path.join(src_dir, fname)
#         dst_path = os.path.join(dst_dir, fname)
#         shutil.copy(src_path, dst_path)

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

#%%
# # Make a nano validation set with 10 images

# val_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/val/gt"
# val_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/val/lq"
# val_nano_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/val_nano/gt"
# val_nano_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/val_nano/lq"
# os.makedirs(val_nano_gt_dir, exist_ok=True)
# os.makedirs(val_nano_lq_dir, exist_ok=True)

# val_gt_files = [f for f in os.listdir(val_gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# val_lq_files = [f for f in os.listdir(val_lq_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # Sort to ensure matching order
# val_gt_files.sort()
# val_lq_files.sort()

# # Take only the first 10 images (assuming filenames match between gt and lq)
# for fname in val_gt_files[:10]:
#     src_path = os.path.join(val_gt_dir, fname)
#     dst_path = os.path.join(val_nano_gt_dir, fname)
#     shutil.copy(src_path, dst_path)

# for fname in val_lq_files[:10]:
#     src_path = os.path.join(val_lq_dir, fname)
#     dst_path = os.path.join(val_nano_lq_dir, fname)
#     shutil.copy(src_path, dst_path)

#%%
"Make a nano training set with 10 images"

# train_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/train/gt"
# train_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/train/lq"
# train_nano_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/train_nano/gt"
# train_nano_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/train_nano/lq"
# os.makedirs(train_nano_gt_dir, exist_ok=True)
# os.makedirs(train_nano_lq_dir, exist_ok=True)

# train_gt_files = [f for f in os.listdir(train_gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# train_lq_files = [f for f in os.listdir(train_lq_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # Sort to ensure matching order
# train_gt_files.sort()
# train_lq_files.sort()

# # Take only the first 10 images (assuming filenames match between gt and lq)
# for fname in train_gt_files[:10]:
#     src_path = os.path.join(train_gt_dir, fname)
#     dst_path = os.path.join(train_nano_gt_dir, fname)
#     shutil.copy(src_path, dst_path)

# for fname in train_lq_files[:10]:
#     src_path = os.path.join(train_lq_dir, fname)
#     dst_path = os.path.join(train_nano_lq_dir, fname)
#     shutil.copy(src_path, dst_path)

#%%
input_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images512x512/train_nano/gt"
output_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images256x256/train_nano/gt"
output_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images256x256/train_nano/lq"
os.makedirs(output_gt_dir, exist_ok=True)
os.makedirs(output_lq_dir, exist_ok=True)

image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
for fname in tqdm(image_files, desc="Processing train_nano images"):
    try:
        img = Image.open(os.path.join(input_dir, fname)).convert("RGB")
        img_resized = ImageOps.pad(img, (256, 256), method=Image.LANCZOS, color=(0, 0, 0))
        img_resized.save(os.path.join(output_gt_dir, fname))

        latent = encode(pipe, img_resized)
        decoded_img = decode(pipe, latent)
        decoded_img.save(os.path.join(output_lq_dir, fname))
    except Exception as e:
        print(f"Error processing image {fname}: {e}")
        continue

#%%
"Duplicate each file in train_nano/gt and train_nano/lq 20 times with _i suffix"

for subdir in ["gt", "lq"]:
    dir_path = f"/mmfs1/gscratch/amath/vilin/BFRffusion/data/images256x256/train_nano/{subdir}"
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for fname in tqdm(files, desc=f"Duplicating {subdir} images"):
        src_path = os.path.join(dir_path, fname)
        name, ext = os.path.splitext(fname)
        for i in range(20):
            dst_fname = f"{name}_{i}{ext}"
            dst_path = os.path.join(dir_path, dst_fname)
            shutil.copy(src_path, dst_path)

#%%
val_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images256x256/val_nano/gt"
val_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images256x256/val_nano/lq"
os.makedirs(val_gt_dir, exist_ok=True)
os.makedirs(val_lq_dir, exist_ok=True)

src_gt_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images256x256/train_nano/gt"
src_lq_dir = "/mmfs1/gscratch/amath/vilin/BFRffusion/data/images256x256/train_nano/lq"

gt_files = [f for f in os.listdir(src_gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_" not in f]
lq_files = [f for f in os.listdir(src_lq_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_" not in f]

gt_files.sort()
lq_files.sort()

for fname in gt_files:
    shutil.copy(os.path.join(src_gt_dir, fname), os.path.join(val_gt_dir, fname))
for fname in lq_files:
    shutil.copy(os.path.join(src_lq_dir, fname), os.path.join(val_lq_dir, fname))