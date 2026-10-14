"""OCR（Phase3）のテスト。学習済みの重みは使わない（高速）。"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_ocr.data import OcrSynthDataset, collate, jitter_corners, plate_to_tensor  # noqa: E402
from ai_ocr.evaluate import pretty, split_fields  # noqa: E402
from ai_ocr.model import BLANK, CHARS, CRNN, VOCAB, decode_greedy, encode  # noqa: E402
from plate_synth import generate_sample, random_label, render_plate  # noqa: E402


class TestCodec(unittest.TestCase):
    def test_encode_ignores_spaces_and_is_in_range(self):
        ids = encode("沖縄 300 あ 12-34")
        self.assertEqual(len(ids), 11)
        self.assertTrue(all(1 <= i < VOCAB for i in ids))

    def test_decode_collapses_repeats_and_blanks(self):
        """CTC: 同じ文字が続くときは、間に blank が要る。「22」は「2, blank, 2」で表す"""
        ids = encode("1223")
        seq = [ids[0], ids[0], BLANK, ids[1], BLANK, ids[2], ids[2], BLANK, ids[3]]    # 1 1 _ 2 _ 2 2 _ 3
        lp = torch.full((len(seq), 1, VOCAB), -20.0)
        for t, k in enumerate(seq):
            lp[t, 0, k] = 0.0
        text, conf = decode_greedy(lp)[0]
        self.assertEqual(text, "1223")
        self.assertEqual(len(conf), 4)
        self.assertTrue(all(c > 0.99 for c in conf))

    def test_charset_covers_labels(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            for ch in random_label(rng)["text"].replace(" ", ""):
                self.assertIn(ch, CHARS)


class TestFields(unittest.TestCase):
    def test_split_and_pretty(self):
        self.assertEqual(split_fields("沖縄300あ12-34"), dict(region="沖縄", class_no="300", kana="あ", serial="12-34"))
        self.assertEqual(split_fields("沖縄300あ・・12")["serial"], "・・12")
        self.assertEqual(pretty("沖縄300あ12-34"), "沖縄 300 あ 12-34")


class TestData(unittest.TestCase):
    def test_plate_to_tensor(self):
        t = plate_to_tensor(render_plate(random_label(np.random.default_rng(1))).convert("RGB"))
        self.assertEqual(tuple(t.shape), (3, 48, 512))
        self.assertTrue(0.0 <= float(t.min()) and float(t.max()) <= 1.0)

    def test_jitter_corners(self):
        c = generate_sample(3, "easy").label["corners"]
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(jitter_corners(c, 0.0, rng), np.asarray(c))
        self.assertGreater(np.abs(jitter_corners(c, 3.0, rng) - np.asarray(c)).max(), 0.1)

    def test_dataset_item_and_collate(self):
        ds = OcrSynthDataset(6, corner_sigma_max=3.0)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (3, 48, 512))
        self.assertIn(len(y), (10, 11))
        xb, yb, lens = collate([ds[i] for i in range(4)])
        self.assertEqual(tuple(xb.shape), (4, 3, 48, 512))
        self.assertEqual(int(lens.sum()), len(yb))
        self.assertTrue(torch.equal(ds[2][0], OcrSynthDataset(6, corner_sigma_max=3.0)[2][0]))     # 再現性


class TestModel(unittest.TestCase):
    def test_output_shape_and_ctc_loss_is_finite(self):
        m = CRNN().eval()
        ds = OcrSynthDataset(4)
        x, y, lens = collate([ds[i] for i in range(4)])
        with torch.no_grad():
            lp = m(x)
        self.assertEqual(tuple(lp.shape), (128, 4, VOCAB))
        self.assertTrue(torch.allclose(lp.exp().sum(2), torch.ones(128, 4), atol=1e-4))      # 確率の合計が1
        loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(lp, y, torch.full((4,), 128, dtype=torch.long), lens)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
