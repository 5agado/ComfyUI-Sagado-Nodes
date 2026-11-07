import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


class MultiSwitchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "nodes" / "nodes.py"
        spec = importlib.util.spec_from_file_location("utility_nodes_under_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            "torch": types.ModuleType("torch"),
            "comfy": types.ModuleType("comfy"),
        }):
            spec.loader.exec_module(module)
        cls.node_class = module.AnyTypeSwitchMulti

    def execute_lazy_inputs(self, switch, inputcount, sources):
        node = self.node_class()
        optional = node.INPUT_TYPES()["optional"]
        values = {}
        for name, source in sources.items():
            specification = optional.get(name, ("*", {}))
            values[name] = None if specification[1].get("lazy") else source()
        for name in node.check_lazy_status(switch, inputcount, **values):
            values[name] = sources[name]()
        self.assertEqual(node.check_lazy_status(switch, inputcount, **values), [])
        return node.select_input(switch, inputcount, **values)

    def test_only_selected_branch_executes_and_preserves_output_slots(self):
        payloads = [np.zeros((1, 2, 2, 3)), "text", {"key": "value"}, 42, Image.new("RGB", (2, 2))]
        for switch in (False, True):
            for inputcount in (1, 3, 5):
                with self.subTest(switch=switch, inputcount=inputcount):
                    selected = "true" if switch else "false"
                    inactive = "false" if switch else "true"
                    sources = {}
                    for i in range(1, inputcount + 1):
                        sources[f"on_{selected}_{i}"] = Mock(return_value=payloads[i - 1])
                        sources[f"on_{inactive}_{i}"] = Mock(side_effect=AssertionError("Inactive branch executed"))
                    result = self.execute_lazy_inputs(switch, inputcount, sources)
                    self.assertEqual(len(result), 5)
                    for i in range(1, inputcount + 1):
                        self.assertIs(result[i - 1], payloads[i - 1])
                        sources[f"on_{selected}_{i}"].assert_called_once_with()
                        sources[f"on_{inactive}_{i}"].assert_not_called()
                    self.assertEqual(result[inputcount:], (None,) * (5 - inputcount))

    def test_only_unresolved_connected_selected_inputs_are_requested(self):
        for switch in (False, True):
            with self.subTest(switch=switch):
                selected = "true" if switch else "false"
                inactive = "false" if switch else "true"
                values = {
                    f"on_{selected}_1": "already resolved",
                    f"on_{selected}_2": None,
                    f"on_{inactive}_1": None,
                    f"on_{inactive}_2": None,
                    f"on_{inactive}_3": None,
                }
                self.assertEqual(
                    self.node_class().check_lazy_status(switch, 3, **values),
                    [f"on_{selected}_2"],
                )

    def test_unconnected_selected_slots_do_not_evaluate_inactive_inputs(self):
        for switch in (False, True):
            with self.subTest(switch=switch):
                inactive = "false" if switch else "true"
                sources = {f"on_{inactive}_1": Mock(side_effect=AssertionError("Inactive branch executed"))}
                self.assertEqual(self.execute_lazy_inputs(switch, 1, sources), (None,) * 5)
                sources[f"on_{inactive}_1"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
