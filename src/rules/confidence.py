"""確信度の作り方（4通り）。Phase3の「文字ごとの最小確率」は粗かったので、いくつか比べる。

  min_max  … 文字ごとの確率(最大)の最小値（Phase3の方法）
  min_sum  … 文字ごとの確率（連続する同じ文字の合計）の最小値
  margin   … 1位と2位の確率の差の最小値（迷っている文字を見つける）
  seq      … CTCの「文字列全体の確率」（その文字列になる、すべての並べ方の確率の合計）
値が大きいほど、自信がある。
"""
import numpy as np
import torch
import torch.nn.functional as F

from ai_ocr.model import encode

NAMES = ["min_max", "min_sum", "margin", "seq"]


def seq_prob(logp: torch.Tensor, text: str) -> float:
    """CTCで「この文字列だった」確率。logp: (T, V) の対数確率"""
    if not text:
        return 0.0
    tgt = torch.tensor([encode(text)], dtype=torch.long)
    nll = F.ctc_loss(logp[:, None, :], tgt, torch.tensor([logp.shape[0]]), torch.tensor([tgt.shape[1]]),
                     blank=0, reduction="sum", zero_infinity=True)
    return float(torch.exp(-nll)) if torch.isfinite(nll) and float(nll) > 0 else 0.0


def confidences(emissions: list, logp: torch.Tensor, text: str, char_probs: list | None = None) -> dict:
    """emissions: 読み取りの各文字。text: 評価する文字列（補正後でもよい）。char_probs: 補正後の文字ごとの確率（あれば）"""
    if not emissions or len(emissions) != len(text):
        return dict(min_max=0.0, min_sum=0.0, margin=0.0, seq=seq_prob(logp, text) if text else 0.0)
    probs = char_probs if char_probs is not None else [e.prob for e in emissions]
    sums = [max(e.prob_sum, p) for e, p in zip(emissions, probs)]
    margins = []
    for e in emissions:
        ps = sorted((p for _, p in e.alts), reverse=True)
        margins.append(ps[0] - (ps[1] if len(ps) > 1 else 0.0))
    return dict(min_max=float(min(probs)), min_sum=float(min(sums)), margin=float(min(margins)), seq=seq_prob(logp, text))
