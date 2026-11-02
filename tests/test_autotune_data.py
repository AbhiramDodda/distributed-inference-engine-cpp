import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_data", ROOT / "benchmark-aws" / "autotune" / "prepare_data.py"
)
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


class PrepareDataTests(unittest.TestCase):
    def test_selects_first_image_per_class_in_order(self):
        tree = [
            {"path": "ILSVRC2012_img_val_subset/0/b.JPEG"},
            {"path": "ILSVRC2012_img_val_subset/0/a.JPEG"},
            {"path": "ILSVRC2012_img_val_subset/1/c.JPEG"},
        ]
        self.assertEqual(
            prepare_data.select_images(tree, 2),
            [
                ("ILSVRC2012_img_val_subset/0/a.JPEG", 0),
                ("ILSVRC2012_img_val_subset/1/c.JPEG", 1),
            ],
        )

    def test_preprocess_writes_expected_tensor_size(self):
        image = Image.new("RGB", (320, 240), (120, 80, 40))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.bin"
            digest = prepare_data.write_tensor(image, path)
            self.assertEqual(path.stat().st_size, 3 * 224 * 224 * 4)
            self.assertEqual(len(digest), 64)

    def test_missing_class_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no image found"):
            prepare_data.select_images([], 1)


if __name__ == "__main__":
    unittest.main()
