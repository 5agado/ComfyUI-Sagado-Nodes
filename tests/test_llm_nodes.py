import base64
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, UnidentifiedImageError


class LLMNodeTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "nodes" / "llm_nodes.py"
        spec = importlib.util.spec_from_file_location("llm_nodes_under_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)


class APIImageRequestTests(LLMNodeTestCase):
    def setUp(self):
        self.client = Mock()
        self.client.responses.create.return_value.output_text = "image description"
        sdk = types.ModuleType("openai")
        sdk.OpenAI = Mock(return_value=self.client)
        self.enterContext(patch.dict(sys.modules, {"openai": sdk}))
        self.enterContext(patch.dict(self.module.os.environ, {"XAI_API_KEY": "test-only"}))
        self.node = self.module.CallLLMAPINode()

    def request(self, **images):
        result = self.node.get_response("grok-4.3", "Describe the image", "low", 512, **images)
        self.assertEqual(result, ("image description",))
        return self.client.responses.create.call_args.kwargs

    def test_image_requests_use_user_messages_and_responses_content_types(self):
        cases = [
            ({"image_url": "https://example.invalid/image.jpg"}, "https://example.invalid/image.jpg"),
            ({"image_base64": "cG5n"}, "data:image/png;base64,cG5n"),
            ({"image_base64": "data:image/jpeg;base64,anBlZw=="}, "data:image/jpeg;base64,anBlZw=="),
        ]
        for images, expected_url in cases:
            with self.subTest(images=images):
                payload = self.request(**images)
                self.assertEqual(payload["input"], [{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe the image"},
                        {"type": "input_image", "image_url": expected_url},
                    ],
                }])

    def test_base64_still_takes_precedence_over_image_url(self):
        for image_base64 in ("cG5n", "data:image/jpeg;base64,anBlZw=="):
            with self.subTest(image_base64=image_base64):
                payload = self.request(
                    image_url="https://example.invalid/unused.png",
                    image_base64=image_base64,
                )
                expected_url = image_base64 if image_base64.startswith("data:") else "data:image/png;base64," + image_base64
                self.assertEqual(payload["input"][0]["content"][1]["image_url"], expected_url)

    def test_text_only_requests_remain_unchanged(self):
        self.assertEqual(self.request(), {
            "model": "grok-4.3",
            "reasoning": {"effort": "low"},
            "input": "Describe the image",
            "store": False,
            "max_output_tokens": 512,
        })


class LlamaCppRequestTests(LLMNodeTestCase):
    def setUp(self):
        self.response = Mock()
        self.response.json.return_value = {"choices": [{"message": {"content": "image description"}}]}
        self.post = self.enterContext(patch.object(self.module.requests, "post", return_value=self.response))
        self.node = self.module.GetLlamaCppResponseNode()

    def inputs(self, **overrides):
        inputs = {
            "prompt": "Describe the image",
            "server_url": "http://example.invalid:8080/",
            "temperature": 0.7,
            "think": False,
            "max_tokens": 512,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "seed": 42,
        }
        return {**inputs, **overrides}

    def request(self, expected_response="image description", **overrides):
        result = self.node.get_response(**self.inputs(**overrides))
        self.assertEqual(result, (expected_response,))
        return self.post.call_args.kwargs["json"]

    def test_thinking_is_forwarded_without_changing_text_request_options(self):
        for think in (False, True):
            with self.subTest(think=think):
                payload = self.request(think=think)
                self.assertEqual(payload, {
                    "model": "llama-cpp-model",
                    "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe the image"}]}],
                    "temperature": 0.7,
                    "seed": 42,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": think},
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.05,
                    "repeat_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "max_tokens": 512,
                })
                self.assertEqual(self.post.call_args.args, ("http://example.invalid:8080/v1/chat/completions",))
                self.assertEqual(self.post.call_args.kwargs["timeout"], 360)

    def test_local_images_are_encoded_with_detected_mime_type_and_take_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            for image_format, expected_mime in (("PNG", "image/png"), ("JPEG", "image/jpeg")):
                with self.subTest(image_format=image_format):
                    path = Path(directory, "image.bin")
                    Image.new("RGB", (2, 2), (255, 0, 0)).save(path, format=image_format)
                    original_bytes = path.read_bytes()
                    payload = self.request(image_path=str(path), image_base64="unused")
                    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
                    header, encoded = image_url.split(",", 1)
                    self.assertEqual(header, f"data:{expected_mime};base64")
                    self.assertEqual(base64.b64decode(encoded, validate=True), original_bytes)

    def test_existing_url_data_uri_and_raw_base64_inputs_are_preserved(self):
        cases = [
            {"image_path": "http://example.invalid/image.png"},
            {"image_path": "https://example.invalid/image.jpg"},
            {"image_path": "file://server-image.png"},
            {"image_path": "data:image/png;base64,cG5n"},
            {"image_base64": "cG5n"},
            {"image_base64": "data:image/jpeg;base64,anBlZw=="},
        ]
        for images in cases:
            with self.subTest(images=images):
                payload = self.request(**images)
                expected_url = next(iter(images.values()))
                self.assertEqual(payload["messages"][0]["content"][1], {
                    "type": "image_url",
                    "image_url": {"url": expected_url},
                })

    def test_invalid_or_missing_local_images_fail_before_http_request(self):
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory, "invalid.png")
            invalid.write_bytes(b"not an image")
            for path, error in (
                (Path(directory, "missing.png"), FileNotFoundError),
                (invalid, UnidentifiedImageError),
            ):
                with self.subTest(path=path):
                    with self.assertRaises(error):
                        self.request(image_path=str(path))
                    self.post.assert_not_called()

    def test_repeated_node_execution_makes_fresh_http_requests(self):
        self.request()
        self.response.json.return_value = {"choices": [{"message": {"content": "fresh response"}}]}
        self.request(expected_response="fresh response")
        self.assertEqual(self.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
