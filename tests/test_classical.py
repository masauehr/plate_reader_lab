"""古典的ベースラインと評価指標のテスト。実行: .venv/bin/python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from classical import detect_plate, read_text, rectify  # noqa: E402
from classical.detect import fill_holes, label_components  # noqa: E402
from plate_eval import align_ops, cer, edit_distance, exact_match, iou  # noqa: E402
from plate_synth import generate_sample, random_label, render_plate, split_seed  # noqa: E402


class TestMetrics(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 10, 10]), 50 / 150)

    def test_edit_distance_and_cer(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        self.assertEqual(edit_distance("", "abc"), 3)
        self.assertAlmostEqual(cer("沖縄 300 あ 12-34", "沖縄 300 あ 12-34"), 0.0)
        self.assertAlmostEqual(cer("沖縄 300 あ 12-35", "沖縄 300 あ 12-34"), 1 / 11)   # 空白は無視して11文字
        self.assertTrue(exact_match("沖縄 300 あ 12-34", "沖縄300あ12-34"))


class TestAlign(unittest.TestCase):
    def test_align_ops(self):
        self.assertEqual([o[0] for o in align_ops("123", "123")], ["match"] * 3)
        self.assertEqual(align_ops("6", "8"), [("sub", "6", "8")])
        ops = align_ops("沖縄300", "沖縄30")                      # 最後の「0」が脱落
        self.assertEqual(sorted(o[0] for o in ops), ["del"] + ["match"] * 4)       # 2つの「0」のどちらが脱落かは、等価（同じ編集距離）
        self.assertEqual([o for o in ops if o[0] == "del"], [("del", "0", None)])
        ops = align_ops("12", "132")                              # 余分な「3」
        self.assertEqual([o[0] for o in ops], ["match", "ins", "match"])
        self.assertEqual(ops[1], ("ins", None, "3"))
        n_err = sum(o[0] != "match" for o in align_ops("沖縄300あ12-34", "沖縄380あ12-3"))
        self.assertEqual(n_err, edit_distance("沖縄300あ12-34", "沖縄380あ12-3"))     # 誤りの数は編集距離と一致


class TestComponents(unittest.TestCase):
    def test_label_components(self):
        m = np.zeros((10, 10), bool)
        m[1:3, 1:4] = True           # 領域1
        m[6:9, 6:9] = True           # 領域2
        m[3, 3] = True               # 領域1につながる（4近傍）
        labels, n = label_components(m)
        self.assertEqual(n, 2)
        self.assertEqual(len(np.unique(labels[m])), 2)
        self.assertEqual(labels[1, 1], labels[3, 3])

    def test_fill_holes(self):
        m = np.zeros((9, 9), bool)
        m[2:7, 2:7] = True
        m[4, 4] = False              # 内側の穴
        self.assertTrue(fill_holes(m)[4, 4])


class TestClassicalOcr(unittest.TestCase):
    def test_clean_plates_are_read_correctly(self):
        """きれいな板（描画のばらつきあり）は、ほぼすべて完全に読める"""
        rng = np.random.default_rng(0)
        ok = 0
        for _ in range(60):
            lb = random_label(rng)
            ok += exact_match(read_text(render_plate(lb, rng).convert("RGB")).text, lb["text"])
        self.assertGreaterEqual(ok, 57)

    def test_easy_with_true_corners(self):
        """Easyでは、正解の四隅で補正すれば読める（OCR単独の力）"""
        ok = 0
        for i in range(40):
            s = generate_sample(split_seed("val", "easy", i), "easy")
            ok += exact_match(read_text(rectify(s.image, s.label["corners"])).text, s.label["text"])
        self.assertGreaterEqual(ok, 36)

    def test_detection_works_on_easy_and_fails_more_on_hard(self):
        """検出は Easy でおおむね成功し、Hard では大きく劣化する（古典的手法の限界）"""
        def rate(d):
            hit = 0
            for i in range(40):
                s = generate_sample(split_seed("val", d, i), d)
                det = detect_plate(s.image)
                hit += det is not None and iou(det.bbox, s.label["bbox"]) >= 0.5
            return hit / 40
        easy, hard = rate("easy"), rate("hard")
        self.assertGreaterEqual(easy, 0.75)
        self.assertLess(hard, easy - 0.2)


if __name__ == "__main__":
    unittest.main()
