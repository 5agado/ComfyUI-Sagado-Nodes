import logging
import os
import random
import re
import threading

import numpy as np
import yaml
from aiohttp import web
from server import PromptServer

import folder_paths

wildcards_path = os.path.join(folder_paths.base_path, "wildcards")

RE_WildCardQuantifier = re.compile(r"(?P<quantifier>\d+)#__(?P<keyword>[\w.\-+/*\\]+?)__", re.IGNORECASE)
wildcard_lock = threading.Lock()
wildcard_dict = {}


def wildcard_normalize(x):
    return x.replace("\\", "/").replace(' ', '-').lower()


def is_numeric_string(input_str):
    return re.match(r'^-?(\d*\.?\d+|\d+\.?\d*)$', input_str) is not None


# ---------- loading ----------

def _read_wildcard(k, v):
    if isinstance(v, list):
        wildcard_dict[wildcard_normalize(k)] = v
    elif isinstance(v, dict):
        for k2, v2 in v.items():
            _read_wildcard(f"{k}/{k2}", v2)
    elif isinstance(v, str):
        wildcard_dict[wildcard_normalize(k)] = [v]
    elif isinstance(v, (int, float)):
        wildcard_dict[wildcard_normalize(k)] = [str(v)]


def read_wildcard_dict(wildcard_path):
    for root, _, files in os.walk(wildcard_path, followlinks=True):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, wildcard_path)

            if file.endswith('.txt'):
                key = wildcard_normalize(os.path.splitext(rel_path)[0])
                try:
                    with open(file_path, 'r', encoding="ISO-8859-1") as f:
                        lines = f.read().splitlines()
                except (UnicodeDecodeError, Exception):
                    with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                        lines = f.read().splitlines()
                wildcard_dict[key] = [x for x in lines if x.strip() and not x.strip().startswith('#')]

            elif file.endswith('.yaml') or file.endswith('.yml'):
                try:
                    with open(file_path, 'r', encoding="ISO-8859-1") as f:
                        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
                except (yaml.reader.ReaderError, UnicodeDecodeError):
                    with open(file_path, 'r', encoding="UTF-8", errors="ignore") as f:
                        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
                if yaml_data:
                    # Use the yaml file's directory-relative prefix so nested yaml keys
                    # reflect their path (e.g. subdir/colors.yaml key "warm" → "subdir/colors/warm")
                    prefix = wildcard_normalize(os.path.splitext(rel_path)[0])
                    for k, v in yaml_data.items():
                        _read_wildcard(f"{prefix}/{k}", v)


def wildcard_load():
    global wildcard_dict
    with wildcard_lock:
        wildcard_dict = {}
        if os.path.exists(wildcards_path):
            read_wildcard_dict(wildcards_path)
            logging.info(f"[Sagado] Wildcards loaded: {len(wildcard_dict)} entries from {wildcards_path}")
        else:
            logging.info(f"[Sagado] Wildcards directory not found: {wildcards_path}")


def get_wildcard_list():
    with wildcard_lock:
        return sorted(wildcard_dict.keys())


# ---------- processing ----------

def process_comment_out(text):
    lines = text.split('\n')
    lines0 = []
    flag = False
    for line in lines:
        if line.lstrip().startswith('#'):
            flag = True
            continue
        if len(lines0) == 0:
            lines0.append(line)
        elif flag:
            lines0[-1] += ' ' + line
            flag = False
        else:
            lines0.append(line)
    return '\n'.join(lines0)


def process(text, seed=None):
    text = process_comment_out(text)

    if seed is not None:
        random.seed(seed)
    random_gen = np.random.default_rng(seed)

    with wildcard_lock:
        local_dict = dict(wildcard_dict)

    def get_wildcard_value(keyword):
        keyword = wildcard_normalize(keyword)
        return local_dict.get(keyword)

    def get_wildcard_options(string):
        pattern = r"__([\w.\-+/*\\]+?)__"
        options = []
        for match in re.findall(pattern, string):
            keyword = wildcard_normalize(match.lower())
            if '*' in keyword:
                subpattern = keyword.replace('*', '.*').replace('+', '\\+')
                for k, v in local_dict.items():
                    if re.match(subpattern, k) or re.match(subpattern, k + '/'):
                        options.extend(v)
            else:
                v = get_wildcard_value(keyword)
                if v is not None:
                    options.extend(v)
        return options

    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')

            multi_select_pattern = options[0].split('$$')
            select_range = None
            select_sep = ' '
            range_pattern = r'(\d+)(-(\d+))?'
            range_pattern2 = r'-(\d+)'
            wildcard_pattern = r"__([\w.\-+/*\\]+?)__"

            if len(multi_select_pattern) > 1:
                r = re.match(range_pattern, options[0])
                if r is None:
                    r = re.match(range_pattern2, options[0])
                    a = '1'
                    b = r.group(1).strip()
                else:
                    a = r.group(1).strip()
                    b = r.group(3)
                    b = b.strip() if b is not None else a

                if r is not None:
                    if b is not None and is_numeric_string(a) and is_numeric_string(b):
                        select_range = int(a), int(b)
                    elif is_numeric_string(a):
                        x = int(a)
                        select_range = (x, x)

                    def expand_wildcard_or_return_string(opts, pattern, wc_pattern):
                        if len(opts) == 1 and re.findall(wc_pattern, pattern):
                            return get_wildcard_options(pattern)
                        else:
                            opts[0] = pattern
                            return opts

                    if select_range is not None and len(multi_select_pattern) == 2:
                        options = expand_wildcard_or_return_string(options, multi_select_pattern[1], wildcard_pattern)
                    elif select_range is not None and len(multi_select_pattern) == 3:
                        select_sep = multi_select_pattern[1]
                        options = expand_wildcard_or_return_string(options, multi_select_pattern[2], wildcard_pattern)

            adjusted_probabilities = []
            total_prob = 0
            for option in options:
                parts = option.split('::', 1) if isinstance(option, str) else f"{option}".split('::', 1)
                config_value = float(parts[0].strip()) if len(parts) == 2 and is_numeric_string(parts[0].strip()) else 1
                adjusted_probabilities.append(config_value)
                total_prob += config_value

            normalized_probabilities = [p / total_prob for p in adjusted_probabilities]

            if select_range is None:
                select_count = 1
            else:
                def calc_max(opt_len, max_range):
                    return min(max_range + 1, opt_len + 1) if max_range > 0 else opt_len + 1

                def calc_count(max_val, min_range, rng):
                    if max(max_val, min_range) <= 0:
                        return 0
                    elif max_val == min_range:
                        return max_val
                    lo, hi = min(min_range, max_val), max(min_range, max_val)
                    return rng.integers(low=lo, high=hi, size=1)

                select_count = calc_count(calc_max(len(options), select_range[1]), select_range[0], random_gen)

            if select_count > len(options) or total_prob <= 1:
                random_gen.shuffle(options)
                selected_items = options
            else:
                selected_items = random_gen.choice(options, p=normalized_probabilities, size=select_count, replace=False)

            selected_items2 = [re.sub(r'^\s*[0-9.]+::', '', str(x), count=1) for x in selected_items]
            replacements_found = True
            return select_sep.join(selected_items2)

        pattern = r'(?<!\\)\{((?:[^{}]|(?<=\\)[{}])*?)(?<!\\)\}'
        replaced_string = re.sub(pattern, replace_option, string)
        return replaced_string, replacements_found

    def replace_wildcard(string):
        pattern = r"__([\w.\-+/*\\]+?)__"
        replacements_found = False

        for match in re.findall(pattern, string):
            keyword = wildcard_normalize(match.lower())
            options = get_wildcard_value(keyword)

            if options is not None:
                adjusted_probabilities = []
                total_prob = 0
                for option in options:
                    parts = option.split('::', 1)
                    config_value = float(parts[0].strip()) if len(parts) == 2 and is_numeric_string(parts[0].strip()) else 1
                    adjusted_probabilities.append(config_value)
                    total_prob += config_value
                normalized_probabilities = [p / total_prob for p in adjusted_probabilities]
                selected_item = random_gen.choice(options, p=normalized_probabilities, replace=False)
                replacement = re.sub(r'^\s*[0-9.]+::', '', selected_item, count=1)
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)

            elif '*' in keyword:
                subpattern = keyword.replace('*', '.*').replace('+', '\\+')
                total_patterns = []
                for k, v in local_dict.items():
                    if re.match(subpattern, k) or re.match(subpattern, k + '/'):
                        total_patterns.extend(v)
                if total_patterns:
                    replacement = random_gen.choice(total_patterns)
                    replacements_found = True
                    string = string.replace(f"__{match}__", replacement, 1)

            elif '/' not in keyword:
                # depth-agnostic fallback: try __*/keyword__
                string_fallback = string.replace(f"__{match}__", f"__*/{match}__", 1)
                string, replacements_found = replace_wildcard(string_fallback)

        return string, replacements_found

    replace_depth = 100
    stop_unwrap = False
    while not stop_unwrap and replace_depth > 1:
        replace_depth -= 1

        option_quantifier = [e.groupdict() for e in RE_WildCardQuantifier.finditer(text)]
        for match in option_quantifier:
            keyword = match['keyword'].lower()
            quantifier = int(match['quantifier']) if match['quantifier'] else 1
            replacement = '__|__'.join([keyword] * quantifier)
            wilder_keyword = keyword.replace('*', '\\*')
            RE_TEMP = re.compile(fr"(?P<quantifier>\d+)#__(?P<keyword>{wilder_keyword})__", re.IGNORECASE)
            text = RE_TEMP.sub(f"__{replacement}__", text)

        pass1, is_replaced1 = replace_options(text)
        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        text, is_replaced2 = replace_wildcard(pass1)
        stop_unwrap = not is_replaced1 and not is_replaced2

    return text


# ---------- ComfyUI node ----------

WILDCARD_CHOOSER_PLACEHOLDER = "Select wildcard to insert"


class WildcardProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "wildcard_chooser": (
                    [WILDCARD_CHOOSER_PLACEHOLDER] + [f"__{k}__" for k in get_wildcard_list()],
                    {"default": WILDCARD_CHOOSER_PLACEHOLDER},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "doit"
    CATEGORY = "Sagado-Nodes/text"

    def doit(self, text, seed, wildcard_chooser):
        return (process(text, seed),)


# ---------- API routes ----------

@PromptServer.instance.routes.get("/sagado/wildcards/list")
async def api_wildcards_list(request):
    return web.json_response({"data": [f"__{k}__" for k in get_wildcard_list()]})


@PromptServer.instance.routes.get("/sagado/wildcards/reload")
async def api_wildcards_reload(request):
    wildcard_load()
    return web.json_response({"status": "ok", "count": len(wildcard_dict)})


# ---------- init ----------

wildcard_load()
