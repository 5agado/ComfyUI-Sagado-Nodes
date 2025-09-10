from pathlib import Path
from typing import Dict
import json
import os
import subprocess

civitai_base_download_url = 'https://civitai.com/api/download/models'


def download_model(url, rename = None, civitai_api_key = None):
    """Download a model from a given URL using aria2 (!apt-get -qq -y install aria2)"""
    base_aria_command = 'aria2c --console-log-level=error -c -x 16 -s 16 -k 1M'
    # huggingface
    if 'huggingface.co' in url:
        filename = rename or url.split('/')[-1].removesuffix('?download=true')
        cmd = f'{base_aria_command} {url} -o {filename}'
    # civitai
    else:
        token_url = f"{url}{'&' if '?' in url else '?'}token={civitai_api_key}"
        cmd = f'{base_aria_command} "{token_url}"'
        cmd += f' -o {rename}' if rename else ''
    # run the command
    subprocess.run(cmd, shell=True, check=True)


def download_lora(comfyui_root, lora_config: Dict, base_model, civitai_api_key):
    main_category = lora_config['main_category'].replace('/', '-')
    second_category = lora_config['second_category'].replace('/', '-')
    dest_path = Path(f'{comfyui_root}/models/loras/{base_model}/{main_category}/{second_category}'.strip())
    dest_path.mkdir(exist_ok=True, parents=True)
    # change working directory to the destination path
    os.chdir(str(dest_path))
    print('=====')
    print(f'Downloading {lora_config['description']}')
    download_model(f"{civitai_base_download_url}/{lora_config['model_id']}?type=Model&format=SafeTensor", civitai_api_key=civitai_api_key)


def download_loras(json_path: str, include_main_cats: list, exclude_main_cats: list, base_models_enabled: Dict[str, bool],
                   comfyui_root: str, civitai_api_key: str):
    """Download custom loras from a json file"""
    with open(json_path, 'r') as f:
        loras = json.load(f)
        for idx, lora in enumerate(loras):
            print(f'===== {idx}/{len(loras)} =====')
            main_category = lora['main_category'].replace('/', '-')
            skip_lora = lora.get('skip', False)
            # download only selected loras based on main category and skip flag
            if (include_main_cats and main_category not in include_main_cats) or (main_category in exclude_main_cats) or skip_lora:
                continue
            for base_model in [bm for bm, enabled in base_models_enabled.items() if enabled]:
                lora_model_id = lora.get(f'model_id_{base_model}', '')
                if lora_model_id:
                    lora['model_id'] = lora_model_id
                    download_lora(comfyui_root, lora, base_model, civitai_api_key)