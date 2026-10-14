"""検出器（Phase2）のテスト。学習済みの重みは使わない（高速）。"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_detect.data import SynthDataset, corners_to_target, target_to_corners  # noqa: E402
from ai_detect.evaluate import corners_bbox, summarize  # noqa: E402
from ai_detect.model import make_model  # noqa: E402
from plate_synth import generate_sample, split_seed  # noqa: E402
from plate_synth.spec import SCENE_H, SCENE_W  # noqa: E402


class TestData(unittest.TestCase):
    def test_target_roundtrip(self):
        s = generate_sample(5, "medium")
        t = corners_to_target(s.label["corners"])
        self.assertEqual(t.shape, (8,))
        self.assertTrue(((t >= 0) & (t <= 1)).all())
        np.testing.assert_allclose(target_to_corners(t), np.asarray(s.label["corners"]), atol=1e-3)

    def test_synth_dataset_item(self):
        x, y = SynthDataset(6, offset=0)[1]
        self.assertEqual(tuple(x.shape), (3, SCENE_H, SCENE_W))
        self.assertEqual(tuple(y.shape), (8,))
        self.assertTrue(0.0 <= float(x.min()) and float(x.max()) <= 1.0)

    def test_dataset_is_fresh_and_deterministic(self):
        a1, _ = SynthDataset(6, offset=0)[2]
        a2, _ = SynthDataset(6, offset=0)[2]
        b, _ = SynthDataset(6, offset=100)[2]            # offsetが違えば別の画像
        self.assertTrue(torch.equal(a1, a2))
        self.assertFalse(torch.equal(a1, b))

    def test_train_seeds_do_not_overlap_eval(self):
        """学習用の画像のシードは、評価(val/test)のシードと重ならない"""
        k = 5
        d = ("easy", "medium", "hard")[k % 3]
        train_seed = split_seed("train", d, k // 3)
        self.assertNotIn(train_seed, {split_seed("val", d, k // 3), split_seed("test", d, k // 3)})


class TestModels(unittest.TestCase):
    def test_output_shape(self):
        x = torch.rand(2, 3, SCENE_H, SCENE_W)
        for name in ("small", "resnet18_scratch"):     # 事前学習の重みをダウンロードしないものだけ
            m = make_model(name).eval()
            with torch.no_grad():
                self.assertEqual(tuple(m(x).shape), (2, 8))

    def test_can_overfit_one_batch(self):
        """1バッチを繰り返し学習すると、loss が下がる（学習の配線が正しい）"""
        torch.manual_seed(0)
        ds = SynthDataset(8)
        x = torch.stack([ds[i][0] for i in range(8)]); y = torch.stack([ds[i][1] for i in range(8)])
        m = make_model("small"); opt = torch.optim.Adam(m.parameters(), 1e-3)
        loss_fn = torch.nn.SmoothL1Loss(beta=0.01)
        first = None
        for _ in range(25):
            loss = loss_fn(m(x), y); opt.zero_grad(); loss.backward(); opt.step()
            first = first or loss.item()
        self.assertLess(loss.item(), first * 0.5)


class TestEvaluate(unittest.TestCase):
    def test_corners_bbox(self):
        c = np.array([[10, 20], [50, 22], [48, 60], [12, 58]], dtype=float)
        self.assertEqual(corners_bbox(c), [10.0, 20.0, 40.0, 40.0])

    def test_summarize(self):
        df = pd.DataFrame(dict(difficulty=["easy"] * 4, iou=[1.0, 0.95, 0.6, 0.2], corner_err=[1, 2, 3, 4], exact=[True, True, False, False]))
        s = summarize(df)
        self.assertEqual(s.loc["easy", "Det@0.5"], 0.75)
        self.assertEqual(s.loc["easy", "Det@0.9"], 0.5)
        self.assertEqual(s.loc["easy", "読み取り完全一致"], 0.5)


if __name__ == "__main__":
    unittest.main()
