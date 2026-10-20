import json
from pathlib import Path
import numpy as np
from PIL import Image
import io
import os
import base64
import requests

thinking_model_options = {
    'top_p': 0.95,
    'top_k': 20,
    'min_p': 0.05,
    'num_ctx': 32768,
    'repeat_penalty': 1.0,
    'presence_penalty': 0.0
}

standard_model_options = {
    'top_p': 0.8,
    'top_k': 20,
    'min_p': 0.0,
    'num_ctx': 16384,
    'repeat_penalty': 1.1,
    'presence_penalty': 1.5
}

class OllamaNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "llama2"}),
                "prompt": ("STRING", {"default": ""}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": -1, "max": 32000, "step": 128}),
                "think": ("BOOLEAN", {"default": False}),
                "num_ctx": ("INT", {"default": 8192, "min": 1024, "max": 131072, "step": 512}),
            },
            "optional": {
                "image_path": ("STRING", {"default": ""}),
                "image_base64": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    CATEGORY = "Sagado-Nodes/llm"
    FUNCTION = "get_response"
    DESCRIPTION = "Util to get response from local Ollama models"

    def get_response(self, model_name, prompt, temperature, max_tokens, think, num_ctx, image_path, image_base64):
        llm_options = {**(thinking_model_options if think else standard_model_options), "num_ctx": num_ctx}
        response = get_ollama_response(
            model_name, prompt, image_path or None, image_base64 or None, temperature, max_tokens,
            enable_thinking=think, **llm_options
        )
        return (str(response.message.content),)

def get_ollama_response(model_name: str, prompt: str, image_path: str = None, image_base64 = None,
                        temperature = 0.7, max_tokens=1024, enable_thinking=True, **llm_options):
    import ollama
    try :
        response = ollama.chat(
            model=model_name,
            think=enable_thinking,
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
                **llm_options  # merge in arbitrary kargs
            }
        )
    except Exception as e:
        print(f'Error getting LLM response: {e}')
        raise e
    return response

class GetLlamaCppResponseNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "server_url": ("STRING", {"default": "http://127.0.0.1:8080"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "think": ("BOOLEAN", {"default": False}),
                "max_tokens": ("INT", {"default": 2048, "min": -1, "max": 32000, "step": 256}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "image_path": ("STRING", {"default": ""}),
                "image_base64": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    CATEGORY = "Sagado-Nodes/llm"
    FUNCTION = "get_llama_cpp_response"
    DESCRIPTION = "Get response from the provided llama.cpp server"

    def get_llama_cpp_response(self, prompt: str, server_url: str,
                               temperature, think, max_tokens, top_p, top_k, min_p,
                               repeat_penalty, presence_penalty, seed,
                               image_path: str = None, image_base64=None):
        base_url = server_url.rstrip('/')
        url = f"{base_url}/v1/chat/completions"

        llm_options = {
            'top_p': top_p,
            'top_k': top_k,
            'min_p': min_p,
            'repeat_penalty': repeat_penalty,
            'presence_penalty': presence_penalty,
        }

        # Build messages
        messages = []
        user_content = [{"type": "text", "text": prompt}]
        if image_path or image_base64:
            user_content.append({"type": "image_url", "image_url": {"url": image_path or image_base64}})
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": "llama-cpp-model", # llama-server completely ignores this
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "stream": False,
            **llm_options,
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens

        try:
            # Send request to llama-server
            response = requests.post(url, json=payload, timeout=360)
            response.raise_for_status()
            res_data = response.json()
            content = res_data["choices"][0]["message"]["content"]
            return (content,)
        except requests.exceptions.RequestException as e:
            error_msg = f"Failed to connect to llama.cpp server at {url}. Error: {str(e)}"
            print(f"[LocalLlamaCppVisionNode] {error_msg}")
            raise Exception(error_msg)


GROK_MODELS = [
    "grok-4.3",
]

class CallLLMAPINode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (GROK_MODELS, {"default": "grok-4.3"}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "reasoning_effort": (["none", "low", "medium", "high"], {"default": "none"}),
                "max_tokens": ("INT", {"default": 2048, "min": -1, "max": 131072, "step": 128}),
            },
            "optional": {
                "image_url": ("STRING", {"default": ""}),
                "image_base64": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)

    CATEGORY = "Sagado-Nodes/llm"
    FUNCTION = "get_response"
    DESCRIPTION = "Call an LLM API and return the text response"

    def get_response(self, model_name, prompt, reasoning_effort, max_tokens, image_url="", image_base64=""):
        from openai import OpenAI

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is not set")

        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

        if image_url or image_base64:
            img_src = image_base64 if image_base64.startswith("data:") else f"data:image/png;base64,{image_base64}" if image_base64 else image_url
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": img_src},
            ]
        else:
            content = prompt

        try:
            response = client.responses.create(
                model=model_name,
                reasoning={"effort": reasoning_effort},
                input=content,
                store=False,
                max_output_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Error calling LLM API: {e}")
            raise e

        return (response.output_text,)


class ImageToPNGDataURINode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("data_uri", "base64_string")
    CATEGORY = "Sagado-Nodes/llm"
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

        return data_uri, base64_string