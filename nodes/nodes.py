import numpy as np
from pathlib import Path
import torch
from PIL import Image, ImageOps

class ImageLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_idx": ("INT", {"default": "0"}),
                "random_idx": ("BOOLEAN", {"default": False}),
                "shuffle": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "image_path")
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_image_path"

    def get_image_path(self, folder_path, image_idx, random_idx, shuffle, seed):
        images = []
        image_path = ''
        if not folder_path or not Path(folder_path).is_dir():
            print("Invalid folder path provided.")
        for ext in ["jpg", "jpeg", "png"]:
            images.extend(Path(folder_path).glob(f"*.{ext}"))
        if not images:
            print("No images found in the specified folder.")
        np.random.seed(seed)
        if shuffle:
            np.random.shuffle(images)
        if not random_idx:
            if image_idx >= len(images):
                print(f"Index {image_idx} out of range, out of {len(images)} images found.")
            else:
                image_path = str(images[image_idx])
        else:
            image_path = images[np.random.choice(len(images))]

        image, mask = load_image(image_path)
        return image, mask, image_path


class GetNumFramesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("INT", {"default": 1, "min": 1, "max": 6}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("num_frames",)
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_num_frames"

    def get_num_frames(self, seconds):
        fps = 16
        num_frames = (seconds * fps) + 1
        return (num_frames,)


class GetResolutionNode:
    # TODO add some presets depending on models
    # 1280x720 960x544 832x480
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_dimension": ("INT", {"default": 512, "step": 32}),
                "aspect_ratio": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.2}),
                "orientation": (["landscape", "portrait"], {"default": "portrait"}),
            }
        }

    RETURN_TYPES = ("INT","INT")
    RETURN_NAMES = ("width","height")
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_resolution"

    def get_resolution(self, base_dimension, aspect_ratio, orientation):
        if orientation == "landscape":
            height = base_dimension
            width = int(height * aspect_ratio)
        else:
            width = base_dimension
            height = int(width * aspect_ratio)
        return width, height


def load_image(image_path):
    i = Image.open(image_path)
    i = ImageOps.exif_transpose(i)
    width = i.size[0]
    height = i.size[1]

    image = i.convert("RGB")
    image = np.array(image).astype(np.float32) / 255.0
    image = torch.from_numpy(image)[None,]

    if 'A' in i.getbands():
        mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
        if mask.shape != (height, width):
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0),
                                                   size=(height, width),
                                                   mode='bilinear',
                                                   align_corners=False).squeeze()
    else:
        mask = torch.zeros((height, width), dtype=torch.float32, device="cpu")
    return image, mask