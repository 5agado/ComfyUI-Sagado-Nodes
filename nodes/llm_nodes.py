import ollama

class OllamaNode():
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (str, {"default": "llama2"}),
                "prompt": (str, {"default": ""}),
                "image_path": (str, {"default": "", "optional": True}),
                "image_base64": (str, {"default": "", "optional": True}),
                "temperature": (float, {"default": 0.6}),
                "max_tokens": (int, {"default": 2048}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    CATEGORY = "Sagado-Nodes"
    FUNCTION = "get_response"
    DESCRIPTION = "Util to get response from local Ollama models"

    def get_response(self, model_name, prompt, image_path, image_base64, temperature, max_tokens):
        response = get_ollama_response(
            model_name, user_message, image_path or None, image_base64 or None, temperature, max_tokens
        )
        return (str(response.message.content),)

def get_ollama_response(model_name: str, prompt: str, image_path: str = None, image_base64 = None,
                        temperature = 0.7, max_tokens = 1024):
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