from .nodes.nodes import *
from .nodes.llm_nodes import OllamaNode, GetLlamaCppModelNode, GetLlmResponseNode

NODE_CLASS_MAPPINGS = {
    "SGD_Image_Loader": ImageLoaderNode,
    "SGD_Get_Num_Frames": GetNumFramesNode,
    "SGD_Get_Resolution": GetResolutionNode,
    "SGD_Video_Loader": VideoLoaderNode,
    "SGD_Film_Grain": FilmGrainNode,
    "SGD_Step_Every_N": StepEveryNNode,
    "SGD_Any_Type_Switch": AnyTypeSwitch,
    "SGD_Call_Ollama": OllamaNode,
    "SGD_Get_Llama_Cpp_Model": GetLlamaCppModelNode,
    "SGD_Get_Llm_Response": GetLlmResponseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SGD_Image_Loader": "Image Loader",
    "SGD_Get_Num_Frames": "Get Num Frames",
    "SGD_Get_Resolution": "Get Resolution",
    "SGD_Video_Loader": "Video Loader",
    "SGD_Film_Grain": "Film Grain",
    "SGD_Step_Every_N": "Step Every N",
    "SGD_Any_Type_Switch": "Any Type Switch",
    "SGD_Call_Ollama": "Call Ollama",
    "SGD_Get_Llama_Cpp_Model": "Get LlamaCPP Model",
    "SGD_Get_Llm_Response": "Get LLM Response",
}