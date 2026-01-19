from .nodes.nodes import *
from .nodes.llm_nodes import OllamaNode

NODE_CLASS_MAPPINGS = {
    "Image Loader" : ImageLoaderNode,
    "Get Num Frames" : GetNumFramesNode,
    "Get Resolution": GetResolutionNode,
    "Video Loader": VideoLoaderNode,
    "Film Grain": FilmGrainNode,
    "Call Ollama": OllamaNode,
}