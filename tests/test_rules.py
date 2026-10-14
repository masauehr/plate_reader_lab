"""Phase4（書式検査・補正・確信度・振り分け）のテスト。学習済みモデルは使わない（高速）。"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai_ocr.infer import Emission  # noqa: E402
from ai_ocr.model import CHARS, VOCAB, encode  # noqa: E402
from plate_synth import make_eval_set  # noqa: E402
from rules.confidence import confidences, seq_prob  # noqa: E402
from rules.correct import correct  # noqa: E402
from rules.decision import REJECT, REVIEW, AUTO, apply, auroc, choose_threshold, choose_threshold_by_cost, decide, risk_coverage  # noqa: E402
from rules.format import is_valid, split, violations  # noqa: E402


def emissions_for(text: str, alts: dict | None = None, p: float = 0.95):
    """text の各文字を読んだ Emission を作る。alts={位置: [(文字, 確率), ...]} で次点の候補を与える"""
    out = []
    for i, ch in enumerate(text):
        cand = [(ch, p)] + (alts or {}).get(i, [])
        out.append(Emission(ch, p, p, i * 8, cand))
    return out


class TestFormat(unittest.TestCase):
    def test_valid_and_invalid(self):
        for t in ["沖縄300あ12-34", "沖縄300あ・・12", "品川72Fま・・・1", "沖縄 300 あ 12-34"]:
            self.assertTrue(is_valid(t), t)
        for t in ["沖縄3O0あ12-34", "沖縄300あ1234", "那覇A00あ12-34", "沖縄300あ・1・2", "沖縄300あ12-3", "東京300あ12-34", "沖縄300ん12-34"]:
            self.assertFalse(is_valid(t), t)

    def test_split_and_reasons(self):
        self.assertEqual(split("沖縄300あ12-34")["serial"], "12-34")
        self.assertIn("文字数", violations("沖縄300あ12")[0])          # 8文字（10か11が正しい）


class TestCorrect(unittest.TestCase):
    def test_valid_is_ok(self):
        t = "沖縄300あ12-34"
        c = correct(emissions_for(t), t)
        self.assertEqual((c.status, c.text, c.changes), ("ok", t, []))

    def test_class_letter_O_becomes_zero_when_candidate_exists(self):
        t = "沖縄3O0あ12-34"
        c = correct(emissions_for(t, {3: [("0", 0.3)]}), t)
        self.assertEqual((c.status, c.text), ("corrected", "沖縄300あ12-34"))
        self.assertEqual(c.changes, [(3, "O", "0")])

    def test_region_is_chosen_from_list(self):
        t = "沖偏300あ12-34"
        c = correct(emissions_for(t, {1: [("縄", 0.3)]}), t)
        self.assertEqual((c.status, c.text), ("corrected", "沖縄300あ12-34"))

    def test_length_error_cannot_be_fixed(self):
        t = "沖縄300あ12"                                    # 8文字: 脱落は、候補の入れ替えでは直せない
        c = correct(emissions_for(t), t)
        self.assertEqual(c.status, "invalid")
        self.assertEqual(c.text, t)

    def test_no_candidate_means_invalid(self):
        t = "沖縄3O0あ12-34"                                  # 次点の候補に数字が無い → 直せない
        self.assertEqual(correct(emissions_for(t), t).status, "invalid")

    def test_dot_after_digit_is_fixed(self):
        t = "沖縄300あ・1・2"
        c = correct(emissions_for(t, {8: [("5", 0.3)]}), t)        # 8番目（2つ目の「・」）の次点候補が「5」
        self.assertEqual(c.status, "corrected")
        self.assertTrue(is_valid(c.text))


def peaky_logp(text: str, T: int = 128) -> torch.Tensor:
    """text を、間隔をあけて自信満々に出力する対数確率（T, V）"""
    lp = torch.full((T, VOCAB), -12.0)
    lp[:, 0] = 0.0
    for i, k in enumerate(encode(text)):
        t = 5 + i * 10
        lp[t, 0] = -12.0; lp[t, k] = -0.001
    return lp.log_softmax(1)


class TestConfidence(unittest.TestCase):
    def test_seq_prob_prefers_the_emitted_text(self):
        lp = peaky_logp("沖縄300あ12-34")
        good, bad = seq_prob(lp, "沖縄300あ12-34"), seq_prob(lp, "沖縄300あ12-35")
        self.assertGreater(good, 0.9)
        self.assertLess(bad, 1e-3)
        self.assertEqual(seq_prob(lp, ""), 0.0)

    def test_confidence_keys_and_range(self):
        t = "沖縄300あ12-34"
        c = confidences(emissions_for(t, {0: [("那", 0.02)]}), peaky_logp(t), t)
        self.assertEqual(set(c), {"min_max", "min_sum", "margin", "seq"})
        self.assertTrue(all(0.0 <= v <= 1.0 for v in c.values()))
        self.assertAlmostEqual(c["margin"], 0.93, places=2)         # 0.95 - 0.02


class TestDecision(unittest.TestCase):
    def test_decide(self):
        self.assertEqual(decide(False, 0.99, 0.9), REJECT)
        self.assertEqual(decide(True, 0.5, 0.9), REVIEW)
        self.assertEqual(decide(True, 0.95, 0.9), AUTO)

    def test_risk_coverage_and_threshold(self):
        rng = np.random.default_rng(0)
        correct = rng.random(2000) < 0.8
        conf = np.where(correct, rng.beta(5, 1.5, 2000), rng.beta(2, 3, 2000))      # 正解のほうが確信度が高い
        valid = np.ones(2000, bool)
        cov, err, thr = risk_coverage(conf, correct, valid)
        self.assertTrue(np.all(np.diff(cov) > 0))                                  # カバレッジは増えていく
        self.assertLess(err[cov < 0.2].mean(), err[cov > 0.9].mean())               # 絞るほど誤りが少ない
        tau = choose_threshold(conf, correct, valid, 0.05)
        r = apply(conf, correct, valid, tau)
        self.assertLessEqual(r["自動処理の誤り率"], 0.05 + 1e-9)
        self.assertGreater(r["自動処理率"], 0.1)

    def test_invalid_never_auto(self):
        conf, correct = np.array([0.99, 0.99]), np.array([True, True])
        r = apply(conf, correct, np.array([True, False]), 0.5)
        self.assertEqual((r["自動処理率"], r["読み取り不可率"]), (0.5, 0.5))

    def test_auroc(self):
        self.assertEqual(auroc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]), 1.0)
        self.assertEqual(auroc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]), 0.0)
        self.assertAlmostEqual(auroc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]), 0.5)      # 全部同点なら、偶然と同じ
        self.assertAlmostEqual(auroc([0.9, 0.5, 0.5, 0.1], [True, True, False, False]), 0.875)  # 同点は半分として数える

    def test_cost_threshold_follows_error_cost(self):
        rng = np.random.default_rng(1)
        correct = rng.random(3000) < 0.85
        conf = np.where(correct, rng.beta(4, 1.5, 3000), rng.beta(2, 3, 3000))
        valid = np.ones(3000, bool)
        cheap = choose_threshold_by_cost(conf, correct, valid, 1.0, 1.0)         # 誤りが安い → 緩く自動処理
        dear = choose_threshold_by_cost(conf, correct, valid, 1.0, 200.0)        # 誤りが高い → 厳しく（要確認に回す）
        self.assertLess(cheap, dear)


class TestEvalSetSafety(unittest.TestCase):
    def test_zero_rows_is_refused(self):
        """n=0 で既存の labels.jsonl を空で上書きしない（過去に起きた事故の再発防止）"""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                make_eval_set(Path(d), "val", "easy", 0)
            self.assertFalse((Path(d) / "val").exists())

    def test_start_offset_gives_different_images(self):
        with tempfile.TemporaryDirectory() as d:
            a = make_eval_set(Path(d), "val", "easy", 2, start=0)
            b = make_eval_set(Path(d), "val", "easy", 2, start=500, out_split="valcal")
            self.assertNotEqual((a / "labels.jsonl").read_text(encoding="utf-8"), (b / "labels.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
