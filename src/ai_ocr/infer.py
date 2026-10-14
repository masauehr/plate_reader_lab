"""CRNNの出力を、文字ごとの「確率・位置・候補」つきで取り出す（Phase4の補正・確信度の材料）。"""
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image

from .data import plate_to_tensor
from .model import BLANK, CHARS
from classical.ocr import rectify


@dataclass
class Emission:
    """読み取った1文字ぶんの情報"""
    char: str
    prob: float                 # その文字が選ばれた位置での確率（連続する同じ文字のうち最大）
    prob_sum: float             # 連続する同じ文字の確率の合計（1で打ち切り）。確率が隣の位置に分かれても拾える
    frame: int                  # 出力した位置（0〜127）
    alts: list = field(default_factory=list)       # その位置での候補（文字, 確率）の上位5つ（blankを除く）


@dataclass
class Reading:
    text: str                   # 貪欲デコードの文字列（空白なし）
    emissions: list             # Emission のリスト（textの1文字ずつ）
    logp: torch.Tensor          # (128, 語彙数) の対数確率（系列全体の確率の計算に使う）


def read_logprobs(log_probs: torch.Tensor) -> list[Reading]:
    """log_probs: (T, B, V)。バッチの各画像について Reading を作る"""
    p = log_probs.exp().cpu().numpy()
    out = []
    for b in range(p.shape[1]):
        pb = p[:, b]
        best = pb.argmax(1)
        ems, prev, run = [], -1, []
        def flush():
            if prev > 0 and run:
                frames = [t for t, _ in run]
                probs = [q for _, q in run]
                t_best = frames[int(np.argmax(probs))]
                order = [k for k in np.argsort(-pb[t_best]) if k != BLANK][:5]
                ems.append(Emission(CHARS[prev - 1], float(max(probs)), float(min(1.0, sum(probs))), int(t_best),
                                    [(CHARS[k - 1], float(pb[t_best, k])) for k in order]))
        for t, k in enumerate(best):
            if k != prev:
                flush(); run = []
            run.append((t, float(pb[t, k]))); prev = k
        flush()
        out.append(Reading("".join(e.char for e in ems), ems, log_probs[:, b].detach().cpu()))
    return out


@torch.no_grad()
def read_plates(model, device, plates: list[torch.Tensor], batch: int = 128) -> list[Reading]:
    model.eval()
    out = []
    for i in range(0, len(plates), batch):
        out += read_logprobs(model(torch.stack(plates[i:i + batch]).to(device)))
    return out


def load_plates(items, corners) -> list[torch.Tensor]:
    """評価画像と四隅から、モデルへの入力（1行にした板）を作る。四隅が壊れて直せない場合は空の板にする"""
    out = []
    for (d, path, lab), c in zip(items, corners):
        try:
            out.append(plate_to_tensor(rectify(Image.open(path).convert("RGB"), c)))
        except Exception:
            out.append(torch.zeros(3, 48, 512))
    return out
