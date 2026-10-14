"""検出器の評価。IoU・Det@0.5・Det@0.9・四隅の誤差、そして Phase1 のOCRで読んだ結果（検出→読み取り）まで測る。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classical import read_text, rectify  # noqa: E402
from plate_eval import cer, exact_match, iou  # noqa: E402

from .data import FileDataset, target_to_corners  # noqa: E402


def corners_bbox(c: np.ndarray) -> list:
    lo, hi = c.min(0), c.max(0)
    return [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]


@torch.no_grad()
def predict(model, ds: FileDataset, device, batch: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in DataLoader(ds, batch_size=batch):
        out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out)


def mean_iou_only(model, ds: FileDataset, device) -> float:
    """学習中のエポック選択用（OCRなしの速い評価）"""
    pred = predict(model, ds, device)
    return float(np.mean([iou(corners_bbox(target_to_corners(p)), lab["bbox"]) for p, (_, _, lab) in zip(pred, ds.items)]))


def evaluate(model, device, start: int = 100, stop: int = 500, split: str = "val", with_ocr: bool = True) -> pd.DataFrame:
    """split の start〜stop 番目で評価し、1枚ずつの結果を返す"""
    ds = FileDataset(split, start, stop)
    pred = predict(model, ds, device)
    rows = []
    for p, (d, path, lab) in zip(pred, ds.items):
        pc = target_to_corners(p)
        gc = np.asarray(lab["corners"], dtype=np.float32)
        r = dict(difficulty=d, image=lab["image"], iou=iou(corners_bbox(pc), lab["bbox"]),
                 corner_err=float(np.linalg.norm(pc - gc, axis=1).mean()), gt=lab["text"])
        if with_ocr:
            try:
                txt = read_text(rectify(Image.open(path).convert("RGB"), pc)).text
            except Exception:                         # 四隅が壊れていて補正できない場合
                txt = ""
            r.update(pred=txt, exact=exact_match(txt, lab["text"]) if txt else False, cer=cer(txt, lab["text"]) if txt else 1.0)
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("difficulty", sort=False)
    out = pd.DataFrame({
        "枚数": g.size(), "平均IoU": g.iou.mean(), "Det@0.5": g.iou.apply(lambda x: (x >= 0.5).mean()),
        "Det@0.9": g.iou.apply(lambda x: (x >= 0.9).mean()), "四隅誤差(px)": g.corner_err.mean(),
    })
    if "exact" in df:
        out["読み取り完全一致"] = g.exact.mean()
    return out.round(3).loc[[d for d in ("easy", "medium", "hard") if d in out.index]]
