"""合成データ生成器のテスト。実行: .venv/bin/python -m unittest discover -s tests -v"""
import re
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from plate_synth import DIFFICULTIES, charset, format_serial, generate_sample, sample_params, split_seed  # noqa: E402
from plate_synth.difficulty import CLEAN  # noqa: E402
from plate_synth.spec import REGIONS, SCENE_H, SCENE_W, SEED_STRIDE  # noqa: E402

SERIAL_RE = re.compile(r"^(\d\d-\d\d|・{0,3}\d{1,3})$")


class TestPlateSynth(unittest.TestCase):
    def test_same_seed_same_result(self):
        """同じシード・難易度なら、画像もラベルも完全に同じ（再現性）"""
        for d in DIFFICULTIES:
            a, b = generate_sample(123, d), generate_sample(123, d)
            self.assertEqual(a.label, b.label)
            self.assertTrue(np.array_equal(np.asarray(a.image), np.asarray(b.image)))

    def test_different_seed_different_image(self):
        a, b = generate_sample(1, "medium"), generate_sample(2, "medium")
        self.assertNotEqual(a.label["text"], b.label["text"])

    def test_label_format(self):
        """文字列の形式が仕様どおり（地名・分類番号3文字・かな1文字・一連番号）"""
        for i in range(60):
            lb = generate_sample(i, DIFFICULTIES[i % 3]).label
            self.assertIn(lb["region"], REGIONS)
            self.assertRegex(lb["class_no"], r"^[1-9]\d[0-9ACFHKLMPXY]$")
            self.assertEqual(len(lb["kana"]), 1)
            self.assertRegex(lb["serial"], SERIAL_RE)
            self.assertEqual(lb["text"], f"{lb['region']} {lb['class_no']} {lb['kana']} {lb['serial']}")

    def test_all_chars_in_charset(self):
        """ラベルの文字がすべて文字集合に含まれる（OCRで読める範囲）"""
        cs = set(charset())
        for i in range(100):
            lb = generate_sample(i, "easy").label
            for ch in lb["text"].replace(" ", ""):
                self.assertIn(ch, cs)

    def test_bbox_and_corners_inside_image(self):
        """板の四隅・外接矩形が画像の内側にある（全難易度）"""
        for d in DIFFICULTIES:
            for i in range(80):
                s = generate_sample(i, d)
                self.assertEqual(s.image.size, (SCENE_W, SCENE_H))
                x, y, w, h = s.label["bbox"]
                self.assertGreater(w, 5); self.assertGreater(h, 3)
                self.assertGreaterEqual(x, 0); self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + w, SCENE_W); self.assertLessEqual(y + h, SCENE_H)
                self.assertEqual(len(s.label["corners"]), 4)

    def test_serial_format(self):
        self.assertEqual(format_serial(1234), "12-34")
        self.assertEqual(format_serial(1000), "10-00")
        self.assertEqual(format_serial(123), "・123")
        self.assertEqual(format_serial(12), "・・12")
        self.assertEqual(format_serial(1), "・・・1")

    def test_difficulty_orders_degradation(self):
        """難しい難易度ほど、劣化が強い傾向（板が小さい・ノイズが大きい）"""
        def mean(key, d):
            return np.mean([sample_params(d, np.random.default_rng(i))[key] for i in range(200)])
        self.assertGreater(mean("scale", "easy"), mean("scale", "hard"))
        self.assertLess(mean("noise_sigma", "easy"), mean("noise_sigma", "hard"))
        self.assertLess(mean("blur_sigma", "easy"), mean("blur_sigma", "hard"))

    def test_overrides(self):
        """overridesで単独の劣化を指定できる。未知のキーはエラー"""
        p = sample_params("easy", np.random.default_rng(0), {"noise_sigma": 20.0})
        self.assertEqual(p["noise_sigma"], 20.0)
        with self.assertRaises(KeyError):
            sample_params("easy", np.random.default_rng(0), {"unknown": 1})
        self.assertEqual(set(CLEAN), set(p))

    def test_split_seeds_do_not_overlap(self):
        """train/val/test と難易度のシード範囲が重ならない（データリーク防止）"""
        ranges = []
        for sp in ("train", "val", "test", "sim"):
            for d in DIFFICULTIES:
                lo = split_seed(sp, d, 0)
                ranges.append((lo, lo + SEED_STRIDE - 1))
        ranges.sort()
        for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
            self.assertLess(a1, b0)
        with self.assertRaises(ValueError):
            split_seed("val", "easy", SEED_STRIDE)


if __name__ == "__main__":
    unittest.main()
