import json
from pathlib import Path
import numpy as np
from PIL import Image
import io
import base64

class OllamaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "llama2"}),
                "prompt": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": -1, "max": 32000, "step": 128}),
            },
            "optional": {
                "image_path": ("STRING", {"default": ""}),
                "image_base64": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_response"
    DESCRIPTION = "Util to get response from local Ollama models"

    def get_response(self, model_name, prompt, temperature, max_tokens, image_path, image_base64):
        response = get_ollama_response(
            model_name, prompt, image_path or None, image_base64 or None, temperature, max_tokens
        )
        return (str(response.message.content),)

def get_ollama_response(model_name: str, prompt: str, image_path: str = None, image_base64 = None,
                        temperature = 0.7, max_tokens = 1024):
    import ollama
    try :
        response = ollama.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                    'images': [image_path or image_base64] if image_path or image_base64 else []
                }
            ],
            options={
                'num_predict': max_tokens,
                'temperature': temperature,
                'repeat_penalty': 1.1,
            }
        )
    except Exception as e:
        print(f'Error getting LLM response: {e}')
        raise e
    return response

class GetLlamaCppModelNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": ""}),
                "models_dir_path": ("STRING", {"default": ""}),
                "chat_format": ("STRING", {"default": ""}),
                "n_ctx": ("INT", {"default": 4094, "min": -1, "max": 128000, "step": 256}),
            },
            "optional": {
                "chat_handler": ("MODEL",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_llama_cpp_model"
    DESCRIPTION = "Load a local model using llama-cpp-python and return the model object for subsequent calls"


    def get_llama_cpp_model(self, model_name, models_dir_path, chat_format, n_ctx, chat_handler=None):
        from llama_cpp import Llama
        model_path = Path(models_dir_path) / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        model = Llama(
            model_path=str(model_path),
            chat_format=chat_format,
            chat_handler=chat_handler,  # Injected vision handler
            verbose=False,
            n_gpu_layers=-1,
            n_ctx = n_ctx,
        )
        return (model,)

class GetLlmResponseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant."}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": -1, "max": 32000, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "image_path": ("STRING", {"default": ""}),
                "image_base64": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_message", "full_response")

    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_llm_response"
    DESCRIPTION = "Get response from the provided llama.cpp model"

    def get_llm_response(self, model, prompt: str, system_prompt: str,
                         image_path: str = None, image_base64 = None,
                         temperature = 0.7, max_tokens = 1024, top_p = 0.95, seed = 42):
        try:
            if image_path or image_base64:
                user_content = [
                    {"type" : "text", "text": prompt},
                    {"type": "image_url", "image_url": [image_path or image_base64][0]},
                ]
            else:
                user_content = prompt
            response = model.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        'role': 'user',
                        'content': user_content,
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response_json = json.dumps(response, indent=4)
            response_message = response['choices'][0]['message']['content']
            # hardcoded extract of message when "thinking" model
            response_message = response_message.split('</think>')[-1].strip()
        except Exception as e:
            print(f'Error getting LLM response: {e}')
            raise e
        return response_message, response_json


class GetLlamaVLChatHandlerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_model_name": ("STRING", {"default": ""}),
                "models_dir_path": ("STRING", {"default": ""}),
                "handler_type": (["Qwen3VL", "Qwen25VL"], {"default": "Qwen3VL"}),
                "image_min_tokens": ("INT", {"default": 2048, "min": -1, "max": 12000, "step": 128}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("chat_handler",)
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_chat_handler"
    DESCRIPTION = "Get the appropriate llama chat-handler for VL models"

    def get_chat_handler(self, clip_model_name, models_dir_path, handler_type, image_min_tokens):
        from llama_cpp.llama_chat_format import (
            Qwen3VLChatHandler,
            Qwen25VLChatHandler
        )

        model_path = Path(models_dir_path) / clip_model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Chat-handler not found at: {model_path}")

        if handler_type == "Qwen3VL":
            handler_cls = Qwen3VLChatHandler
        elif handler_type == "Qwen25VL":
            handler_cls = Qwen25VLChatHandler
        else:
            raise ValueError(f"Unsupported handler type: {handler_type}")
        handler = handler_cls(clip_model_path=str(model_path), image_min_tokens=image_min_tokens)

        return (handler,)


class ImageToPNGDataURINode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("data_uri",)
    CATEGORY = "Sagado-Nodes"
    FUNCTION = "encode_to_png_uri"
    DESCRIPTION = "Converts a ComfyUI image to a PNG Data URI (data:image/png;base64,...)."

    def encode_to_png_uri(self, image):
        img_tensor = image[0]
        i = 255. * img_tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # Save to Bytes as PNG
        buffered = io.BytesIO()
        img.save(buffered, format="PNG", compress_level=0)

        # Encode to Base64
        base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_string}"

        return (data_uri,)