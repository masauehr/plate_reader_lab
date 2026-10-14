"""OCRの評価。四隅の出どころを選べる（正解の四隅＝読み取りの力そのもの／検出器の予測＝システム全体）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from plate_eval import cer, exact_match  # noqa: E402

from .data import EvalItems, plate_to_tensor  # noqa: E402
from .model import decode_greedy  # noqa: E402
from classical.ocr import rectify  # noqa: E402


def split_fields(s: str) -> dict:
    """空白なしの文字列を、地名(2)・分類番号(3)・かな(1)・一連番号(残り)に分ける"""
    return dict(region=s[:2], class_no=s[2:5], kana=s[5:6], serial=s[6:])


def pretty(s: str) -> str:
    f = split_fields(s)
    return f"{f['region']} {f['class_no']} {f['kana']} {f['serial']}"


@torch.no_grad()
def predict_corners(detector, items, device, batch: int = 64) -> np.ndarray:
    """検出器（Phase2）で、評価画像の四隅を予測する"""
    from ai_detect.data import to_tensor, target_to_corners
    detector.eval()
    out = []
    for i in range(0, len(items), batch):
        x = torch.stack([to_tensor(Image.open(p)) for _, p, _ in items[i:i + batch]]).to(device)
        out += [target_to_corners(q) for q in detector(x).cpu().numpy()]
    return np.stack(out)


@torch.no_grad()
def evaluate(model, device, corners="gt", detector=None, start: int = 100, stop: int = 500, split: str = "val") -> pd.DataFrame:
    """split の start〜stop 番目で評価する。corners="gt" なら正解の四隅、"pred" なら detector の予測で板を直す"""
    ev = EvalItems(split, start, stop)
    pc = predict_corners(detector, ev.items, device) if corners == "pred" else None
    model.eval()
    xs = []
    for k, (d, path, lab) in enumerate(ev.items):
        c = lab["corners"] if corners == "gt" else pc[k]
        try:
            xs.append(plate_to_tensor(rectify(Image.open(path).convert("RGB"), c)))
        except Exception:                                   # 四隅が壊れて補正できない場合は、空の画像で読む
            xs.append(torch.zeros(3, 48, 512))
    rows = []
    for i in range(0, len(xs), 128):
        lp = model(torch.stack(xs[i:i + 128]).to(device))
        for j, (txt, conf) in enumerate(decode_greedy(lp)):
            d, path, lab = ev.items[i + j]
            gt = lab["text"].replace(" ", "")
            pf, gf = split_fields(txt), {k: lab[k] for k in ("region", "class_no", "kana", "serial")}
            rows.append(dict(difficulty=d, image=lab["image"], gt=gt, pred=txt, exact=txt == gt, cer=cer(txt, gt),
                             conf_min=min(conf) if conf else 0.0, conf_mean=float(np.mean(conf)) if conf else 0.0,
                             n_chars=len(txt), **{f"f_{k}": pf[k] == gf[k] for k in gf}))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("difficulty", sort=False)
    out = pd.DataFrame({"枚数": g.size(), "完全一致": g.exact.mean(), "CER": g.cer.mean(), "地名": g.f_region.mean(),
                        "分類番号": g.f_class_no.mean(), "かな": g.f_kana.mean(), "一連番号": g.f_serial.mean()})
    return out.round(3).loc[[d for d in ("easy", "medium", "hard") if d in out.index]]
