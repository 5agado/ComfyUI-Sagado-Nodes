import numpy as np
from pathlib import Path
import torch
from PIL import Image, ImageOps
import comfy

from server import PromptServer

MESSAGE_TYPE = "sagado_nodes.textmessage"

class ImageLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_idx": ("INT", {"default": "0", "control_after_generate": True}),
                "random_idx": ("BOOLEAN", {"default": False}),
                "shuffle": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "image_path")
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_image"
    DESCRIPTION = "Util to load images from a folder"

    def get_image(self, folder_path, image_idx, random_idx, shuffle, seed):
        image_path = get_media(folder_path, image_idx, random_idx, shuffle, seed, exts=["png", "jpg", "jpeg", "webp"])
        if image_path:
            image, mask = load_image(image_path)
            return image, mask, image_path
        else:
            return None, None, image_path


class VideoLoaderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"default": ""}),
            },
            "optional": {
                "video_idx": ("INT", {"default": "0", "control_after_generate": True}),
                "random_idx": ("BOOLEAN", {"default": False}),
                "shuffle": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 42}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "video_path")
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_video_path"
    DESCRIPTION = "Util to load videos from a folder"

    def get_video_path(self, folder_path, video_idx, random_idx, shuffle, seed):
        video_path = get_media(folder_path, video_idx, random_idx, shuffle, seed, exts=["mp4", "mov", "avi", "mkv"])

        if video_path:
            # TODO load videos
            #image, mask = load_image(image_path)
            return None, None, video_path
        else:
            return None, None, video_path


class GetNumFramesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("INT", {"default": 1, "min": 0, "max": 20}),
                "fps": ("INT", {"default": 16}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("num_frames",)
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_num_frames"
    DESCRIPTION = "Get number of frames from seconds"

    def get_num_frames(self, seconds, fps):
        if seconds == 0:
            num_frames = 1
        else:
            num_frames = (seconds * fps) + 1
        return (num_frames,)


class GetResolutionNode:
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
    DESCRIPTION = "Get resolution from base dimension and aspect ratio"

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


def get_media(folder_path, media_idx, random_idx, shuffle, seed, exts):
    media = []
    media_path = ''
    if not folder_path or not Path(folder_path).is_dir():
        print("Invalid folder path provided.")
        PromptServer.instance.send_sync(MESSAGE_TYPE, {"message": "Invalid folder path provided."})
    for ext in exts:
        media.extend(Path(folder_path).glob(f"*.{ext}"))
    if not media:
        print(f"No media found in the specified folder ({str(exts)}).")
        PromptServer.instance.send_sync(MESSAGE_TYPE, {"message": f"No media found in the specified folder ({str(exts)})."})
    else:
        print(f"Found {len(media)} media in the specified folder ({str(exts)}).")
        np.random.seed(seed)
        if shuffle:
            np.random.shuffle(media)
        if not random_idx:
            if media_idx >= len(media):
                print(f"Index {media_idx} out of range, out of {len(media)} media found.")
                PromptServer.instance.send_sync(MESSAGE_TYPE,
                                                {"message":f"Index {media_idx} out of range, out of {len(media)} media found."})
            else:
                media_path = str(media[media_idx])
        else:
            media_path = media[np.random.choice(len(media))]
    return media_path


class FilmGrainNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.07, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "add_film_grain"
    DESCRIPTION = "Add film grain to image/video"


    def add_film_grain(self, images, strength):
        device = comfy.model_management.get_torch_device()
        images = images.to(device)

        noise = torch.randn_like(images) * strength
        grainy_images = torch.clamp(images + noise, 0.0, 1.0)

        grainy_images = grainy_images.to(comfy.model_management.intermediate_device())
        return (grainy_images,)


class StepEveryNNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0}),
                "increment_every_n": ("INT", {"default": 1}),
                "current_index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "step_seed"
    DESCRIPTION = "Increment seed every N steps"

    def step_seed(self, seed, increment_every_n, current_index):
        # Calculate the offset based on integer division
        # Example: if N=3, seed increases at index 3, 6, 9...
        modified_seed = current_index + (seed // increment_every_n)

        return (modified_seed,)


class AnyTypeSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "on_true": ("*", {"lazy": True}),
                "on_false": ("*", {"lazy": True}),
                "switch": ("BOOLEAN", {"default": True}),
            },
        }

    # The return type is also set to "*" to pass through whatever was sent in
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("OUTPUT",)
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "select_input"
    DESCRIPTION = "Select between two inputs of any type based on a boolean switch"

    def check_lazy_status(self, on_true, on_false, switch):
        return ["on_true"] if switch else ["on_false"]

    def select_input(self, on_true, on_false, switch):
        return (on_true if switch else on_false,)


class AnyListSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_list": ("*",),
                "index": ("INT", {"default": 0, "step": 1}),
            },
        }

    # "*" allows the node to output any data type
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("element",)
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "select_element"
    DESCRIPTION = "Select an element from a list by index, supporting negative indexing for reverse access"

    def select_element(self, any_list, index):
        # check if any_list is actually a list, if not raise an error
        if not isinstance(any_list, list):
            raise TypeError(f"Expected a list for 'any_list', but got {type(any_list).__name__}.")
        if not any_list:
            raise ValueError("The input list is empty.")
        try:
            selected = any_list[index]
            return (selected,)
        except IndexError:
            raise IndexError(f"Index {index} is out of range for list of length {len(any_list)}.")


class StringSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "delimiter": ("STRING", {"default": ","}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("string_list",)
    FUNCTION = "split_string"
    CATEGORY = "Sagado-Nodes"
    DESCRIPTION = "Split a string into a list based on a specified delimiter, supporting escaped characters like \\n and \\t"

    def split_string(self, text, delimiter):
        # Replace escaped newlines if the user literally types '\n'
        actual_delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t")
        result = text.split(actual_delimiter)
        result = [item.strip() for item in result]

        return (result,)
