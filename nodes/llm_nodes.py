import json
from pathlib import Path

class OllamaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "llama2"}),
                "prompt": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048}),
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
                "model_name": ("STRING", {"default": "llama2"}),
                "models_dir_path": ("STRING", {"default": ""}),
                "chat_format": ("STRING", {"default": "llama-2"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_llama_cpp_model"
    DESCRIPTION = "Load a local model using llama-cpp-python and return the model object for subsequent calls"


    def get_llama_cpp_model(self, model_name, models_dir_path, chat_format):
        from llama_cpp import Llama
        models_dir_path = Path(models_dir_path)
        model = Llama(
            model_path=str(models_dir_path / model_name),
            chat_format=chat_format,
            verbose=False,
        )
        return (model,)

class GetLlamaCppResponseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "prompt": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "image_path": ("STRING", {"default": ""}),
                "image_base64": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response_message", "full_response")

    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_llama_cpp_response"
    DESCRIPTION = "Get response from the provided llama.cpp model"

    def get_llama_cpp_response(self, model, prompt: str, image_path: str = None, image_base64 = None,
                               temperature = 0.7, max_tokens = 1024, top_p = 0.95):
        try:
            response = model.create_chat_completion(
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            response_json = json.dumps(response, indent=4)
            response_message = response['choices'][0]['message']['content']
        except Exception as e:
            print(f'Error getting LLM response: {e}')
            raise e
        return response_message, response_json