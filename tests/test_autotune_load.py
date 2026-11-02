import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark-aws"))

from autotune import analyze, workloads  # noqa: E402

try:
    from autotune import loadgen
except ImportError:  # aiohttp is installed in the benchmark venv
    loadgen = None


class WorkloadTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "images": [{"label_index": i, "tensor_path": f"tensors/{i:04d}.bin"} for i in range(30)]
        }

    def test_unique_and_repeated_lengths(self):
        self.assertEqual(len(workloads.select_images(self.manifest, "unique", 10)), 10)
        repeated = workloads.select_images(self.manifest, "repeated", 45)
        self.assertEqual(len(repeated), 45)
        self.assertLessEqual(len({item["label_index"] for item in repeated}), 20)

    def test_mixed_is_deterministic(self):
        first = workloads.select_images(self.manifest, "mixed", 50, seed=7)
        second = workloads.select_images(self.manifest, "mixed", 50, seed=7)
        self.assertEqual(first, second)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            workloads.select_images(self.manifest, "other", 1)


@unittest.skipUnless(loadgen is not None, "aiohttp is required for loadgen tests")
class LoadgenTests(unittest.TestCase):
    def test_percentile_edges(self):
        self.assertIsNone(loadgen.percentile([], 0.5))
        self.assertEqual(loadgen.percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(loadgen.percentile([1.0, 2.0, 3.0], 1.0), 3.0)

    def test_summary_records_errors_and_slo(self):
        rows = [
            {"success": True, "latency_ms": 100.0, "cached": False, "batch_size": 1},
            {"success": True, "latency_ms": 200.0, "cached": True, "batch_size": 2},
            {"success": False, "latency_ms": 300.0, "status": 500},
        ]
        summary = loadgen.summarize(rows, 1.0, "binary", "mixed", concurrency=2)
        self.assertEqual(summary["successes"], 2)
        self.assertEqual(summary["failures"], 1)
        self.assertGreater(summary["error_rate"], 0.001)
        self.assertFalse(summary["slo"]["within_slo"])

    def test_accuracy_uses_label_index(self):
        output = [0.0] * 1000
        output[7] = 10.0
        rows = [{"output": output, "label_index": 7}]
        result = loadgen.accuracy(rows)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["top1"], 1.0)
        self.assertEqual(result["top5"], 1.0)

    def test_build_specs_binary_and_json_request_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp)
            (data / "tensors").mkdir()
            tensor = struct.pack("<3f", 1.0, 2.0, 3.0)
            (data / "tensors" / "0000.bin").write_bytes(tensor)
            manifest = {
                "images": [
                    {"label_index": 0, "tensor_path": "tensors/0000.bin"},
                ]
            }
            binary = loadgen.build_specs(data, manifest, "unique", 1, "binary", "p")
            self.assertEqual(binary[0]["request_id"], "p-0")
            self.assertEqual(binary[0]["payload"], tensor)
            as_json = loadgen.build_specs(data, manifest, "unique", 1, "json", "q")
            body = json.loads(as_json[0]["body"])
            self.assertEqual(body["request_id"], "q-0")
            self.assertEqual(body["input_data"], [1.0, 2.0, 3.0])

    def test_cost_per_million(self):
        self.assertEqual(analyze.cost_per_million(10.0, 0.85), 23.6111)
        self.assertIsNone(analyze.cost_per_million(0.0, 0.85))

    def test_consistency_groups_repeated_outputs(self):
        output = [0.0] * 1000
        output[7] = 10.0
        rows = [
            {"success": True, "label_index": 7, "output": output, "node_id": "worker_1"},
            {"success": True, "label_index": 7, "output": output, "node_id": "worker_2"},
            {
                "success": True,
                "label_index": 8,
                "output": [0.0] * 1000,
                "node_id": "worker_1",
            },
        ]
        result = analyze.consistency(rows)
        self.assertEqual(result["samples"], 3)
        self.assertEqual(result["labels"], 2)
        self.assertEqual(result["top1_conflicts"], 0)
        self.assertEqual(result["multi_node_labels"], 1)
        self.assertEqual(result["max_abs_diff"], 0.0)

    def test_consistency_detects_a_top1_conflict(self):
        first = [0.0] * 1000
        first[1] = 10.0
        second = [0.0] * 1000
        second[2] = 10.0
        rows = [
            {"success": True, "label_index": 1, "output": first, "node_id": "worker_1"},
            {"success": True, "label_index": 1, "output": second, "node_id": "worker_2"},
        ]
        result = analyze.consistency(rows)
        self.assertEqual(result["top1_conflicts"], 1)
        self.assertGreater(result["max_abs_diff"], 0.0)


if __name__ == "__main__":
    unittest.main()
