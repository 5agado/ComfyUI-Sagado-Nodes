from .nodes.nodes import *
from .nodes.llm_nodes import OllamaNode

NODE_CLASS_MAPPINGS = {
    "SGD_Image_Loader": ImageLoaderNode,
    "SGD_Get_Num_Frames": GetNumFramesNode,
    "SGD_Get_Resolution": GetResolutionNode,
    "SGD_Video_Loader": VideoLoaderNode,
    "SGD_Film_Grain": FilmGrainNode,
    "SGD_Call_Ollama": OllamaNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SGD_Image_Loader": "Image Loader",
    "SGD_Get_Num_Frames": "Get Num Frames",
    "SGD_Get_Resolution": "Get Resolution",
    "SGD_Video_Loader": "Video Loader",
    "SGD_Film_Grain": "Film Grain",
    "SGD_Call_Ollama": "Call Ollama",
}