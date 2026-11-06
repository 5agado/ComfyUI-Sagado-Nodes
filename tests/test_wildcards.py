import copy
import importlib.util
import io
import json
import random
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


class WildcardMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=Mock())
        server.PromptServer.instance.routes.get.side_effect = lambda path: lambda handler: handler
        folder_paths = types.ModuleType("folder_paths")
        path = Path(__file__).resolve().parents[1] / "nodes" / "wildcards.py"
        spec = importlib.util.spec_from_file_location("wildcards_under_test", path)
        cls.wildcards = importlib.util.module_from_spec(spec)
        with tempfile.TemporaryDirectory() as directory:
            folder_paths.base_path = directory
            with patch.dict(sys.modules, {"server": server, "folder_paths": folder_paths}):
                spec.loader.exec_module(cls.wildcards)

    def setUp(self):
        self.random_state = random.getstate()
        self.wildcards.wildcard_dict = {"colors": ["red", "green", "blue"]}
        self.processor = self.wildcards.WildcardProcessor()

    def tearDown(self):
        random.setstate(self.random_state)

    def make_prompt(self, text="__colors__", seed=17, populated_text="previous run", mode="populate"):
        inputs = {
            "text": text,
            "seed": seed,
            "wildcard_chooser": self.wildcards.WILDCARD_CHOOSER_PLACEHOLDER,
            "mode": mode,
            "populated_text": populated_text,
        }
        names = ["text", "seed", "control_after_generate", "wildcard_chooser", "mode", "populated_text"]
        values = [
            text if isinstance(text, str) else "",
            seed if isinstance(seed, int) else 17,
            "fixed",
            inputs["wildcard_chooser"],
            mode,
            populated_text,
        ]
        return {
            "prompt": {"7": {"class_type": "SGD_Wildcard_Processor", "inputs": inputs}},
            "extra_data": {
                "extra_pnginfo": {
                    "workflow": {
                        "nodes": [{
                            "id": 7,
                            "type": "SGD_Wildcard_Processor",
                            "widgets_values": values,
                            "widgets_values_named": dict(zip(names, values)),
                        }],
                    },
                },
            },
        }

    def workflow_node(self, data):
        return data["extra_data"]["extra_pnginfo"]["workflow"]["nodes"][0]

    def execute(self, data, **resolved_inputs):
        inputs = {**data["prompt"]["7"]["inputs"], **resolved_inputs}
        return self.processor.doit(
            **inputs,
            unique_id="7",
            extra_pnginfo=data["extra_data"]["extra_pnginfo"],
        )

    def assert_saved(self, data, result):
        text = result["result"][0]
        node = self.workflow_node(data)
        self.assertEqual(node["widgets_values"][4:6], ["reproduce", text])
        self.assertEqual(node["widgets_values_named"]["mode"], "reproduce")
        self.assertEqual(node["widgets_values_named"]["populated_text"], text)
        self.assertEqual(result["ui"]["populated_text"], [text])
        self.assertEqual(result["ui"]["mode"], ["populate"])

    def test_linked_text_saves_current_result_on_consecutive_runs(self):
        previous = "previous run"
        for text in ("first __colors__", "second __colors__"):
            with self.subTest(text=text):
                data = self.make_prompt(text=["3", 0], populated_text=previous)
                self.wildcards.onprompt(data)
                result = self.execute(data, text=text)
                self.assertEqual(result["result"][0], self.wildcards.process(text, 17))
                self.assertNotEqual(result["result"][0], previous)
                self.assert_saved(data, result)
                self.assertEqual(data["prompt"]["7"]["inputs"]["text"], ["3", 0])
                previous = result["result"][0]

    def test_linked_seed_saves_current_result(self):
        data = self.make_prompt(seed=["3", 0])
        self.wildcards.onprompt(data)
        result = self.execute(data, seed=29)
        expected = np.random.default_rng(29).choice(["red", "green", "blue"], p=[1 / 3] * 3)
        self.assertEqual(result["result"][0], expected)
        self.assert_saved(data, result)
        self.assertEqual(data["prompt"]["7"]["inputs"]["seed"], ["3", 0])

    def test_empty_linked_text_clears_previous_result(self):
        data = self.make_prompt(text=["3", 0])
        self.wildcards.onprompt(data)
        result = self.execute(data, text="")
        self.assertEqual(result["result"], ("",))
        self.assert_saved(data, result)

    def test_linked_workflow_png_round_trip_reproduces_saved_result(self):
        data = self.make_prompt(text=["3", 0])
        self.wildcards.onprompt(data)
        result = self.execute(data, text="current __colors__")
        metadata = PngInfo()
        metadata.add_text("workflow", json.dumps(data["extra_data"]["extra_pnginfo"]["workflow"]))
        buffer = io.BytesIO()
        Image.new("RGB", (1, 1)).save(buffer, format="PNG", pnginfo=metadata)
        buffer.seek(0)
        with Image.open(buffer) as image:
            workflow = json.loads(image.info["workflow"])
        saved = workflow["nodes"][0]["widgets_values_named"]
        with patch.object(self.wildcards, "process", side_effect=AssertionError("Must not repopulate")):
            replay = self.processor.doit(
                text="different upstream text",
                seed=999,
                wildcard_chooser=saved["wildcard_chooser"],
                mode=saved["mode"],
                populated_text=saved["populated_text"],
            )
        self.assertEqual(replay["result"], result["result"])

    def test_inline_hook_updates_both_metadata_formats_before_execution(self):
        data = self.make_prompt()
        node = self.workflow_node(data)
        original_values = node["widgets_values"][:4]
        self.wildcards.onprompt(data)
        inputs = data["prompt"]["7"]["inputs"]
        self.assertEqual(inputs["mode"], "reproduce")
        self.assertEqual(node["widgets_values_named"]["populated_text"], inputs["populated_text"])
        self.assertEqual(node["widgets_values_named"]["mode"], "reproduce")
        self.assertEqual(node["widgets_values"][:4], original_values)
        with patch.object(self.wildcards, "process", side_effect=AssertionError("Already populated")):
            self.assert_saved(data, self.execute(data))

    def test_reproduce_hook_updates_metadata_even_with_linked_text(self):
        data = self.make_prompt(text=["3", 0], mode="reproduce", populated_text="saved result")
        with patch.object(self.wildcards, "process", side_effect=AssertionError("Must not repopulate")):
            self.wildcards.onprompt(data)
            self.assertEqual(self.workflow_node(data)["widgets_values"][4:6], ["reproduce", "saved result"])
            result = self.execute(data, text="different upstream text")
        self.assertEqual(result["result"], ("saved result",))
        self.assert_saved(data, result)

    def run_queued_with_cache(self, template):
        cache = {}
        executions = 0
        for _ in range(2):
            data = self.wildcards.onprompt(copy.deepcopy(template))
            inputs = data["prompt"]["7"]["inputs"]
            fingerprint = self.processor.IS_CHANGED(**inputs)
            key = (json.dumps(inputs, sort_keys=True), fingerprint)
            if key not in cache:
                resolved = {}
                if isinstance(inputs["text"], list):
                    resolved["text"] = "__colors__"
                cache[key] = self.execute(data, **resolved)
                executions += 1
            self.assert_saved(data, cache[key])
        return executions

    def test_repeated_queued_linked_prompts_do_not_skip_metadata_updates(self):
        self.assertEqual(self.run_queued_with_cache(self.make_prompt(text=["3", 0])), 2)

    def test_inline_and_reproduce_prompts_remain_cacheable(self):
        for template in (
            self.make_prompt(),
            self.make_prompt(text=["3", 0], mode="reproduce", populated_text="saved result"),
        ):
            with self.subTest(inputs=template["prompt"]["7"]["inputs"]):
                self.assertEqual(self.run_queued_with_cache(template), 1)

    def test_legacy_and_named_only_workflows(self):
        for removed_key in ("widgets_values", "widgets_values_named"):
            with self.subTest(removed_key=removed_key):
                data = self.make_prompt(text=["3", 0])
                node = self.workflow_node(data)
                del node[removed_key]
                self.wildcards.onprompt(data)
                result = self.execute(data, text="current result")
                if removed_key == "widgets_values":
                    self.assertEqual(node["widgets_values_named"]["populated_text"], result["result"][0])
                else:
                    self.assertEqual(node["widgets_values"][4:6], ["reproduce", result["result"][0]])

    def test_api_execution_without_workflow_metadata(self):
        data = self.make_prompt()
        del data["extra_data"]
        self.wildcards.onprompt(data)
        result = self.processor.doit(**data["prompt"]["7"]["inputs"])
        self.assertEqual(result["result"][0], self.wildcards.process("__colors__", 17))

    def test_unmatched_workflow_node_logs_metadata_limitation(self):
        data = self.make_prompt(text=["3", 0])
        self.workflow_node(data)["id"] = 99
        with self.assertLogs(level="WARNING") as logs:
            result = self.execute(data, text="current result")
        self.assertEqual(result["result"], ("current result",))
        self.assertIn("workflow node 7 was not found", logs.output[0])


if __name__ == "__main__":
    unittest.main()
