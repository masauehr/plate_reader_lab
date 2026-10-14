"""Phase1: 古典的ベースライン（機械学習なし）の評価。検証セット(val)で、難易度別に測る。

使い方:
    .venv/bin/python src/eval_baseline.py            # val の先頭100枚を除く400枚/難易度（約1分）
    .venv/bin/python src/eval_baseline.py --skip 0   # 全500枚（検出のしきい値を決めた先頭100枚を含む）
出力: outputs/baseline/{summary.md, val_per_sample.csv, failures_*.png}

評価の方針（EVALUATION.md）:
  - 検出の手調整は val の先頭100枚で行った。そのため既定ではそれを除いて評価する。
  - test は使わない（最後に1回だけ）。
  - 3つの見方: 検出だけ / OCRだけ（正解の四隅で補正＝OCRの力そのもの）/ 全体（検出→OCR）
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classical import read_plate, read_text, rectify  # noqa: E402
from plate_eval import cer, exact_match, field_matches, iou  # noqa: E402
from plate_synth import DIFFICULTIES, load_labels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "outputs" / "baseline"


def parse_fields(text: str) -> dict:
    parts = text.split(" ")
    return dict(zip(("region", "class_no", "kana", "serial"), parts)) if len(parts) == 4 else {}


def evaluate_one(args):
    split, diff, lab = args
    img = Image.open(EVAL / split / diff / "images" / lab["image"]).convert("RGB")
    gt_text, gt_box = lab["text"], lab["bbox"]
    # (1) 全体: 検出 → 補正 → OCR
    res = read_plate(img)
    det_iou = iou(res.detection.bbox, gt_box) if res.detection else 0.0
    pred = res.text
    fm = field_matches(parse_fields(pred), lab) if pred else dict.fromkeys(("region", "class_no", "kana", "serial"), False)
    # (2) OCRだけ: 正解の四隅で補正してから読む
    o = read_text(rectify(img, lab["corners"]))
    return dict(
        difficulty=diff, image=lab["image"], gt=gt_text, pred=pred, detected=res.detection is not None, iou=det_iou,
        exact=exact_match(pred, gt_text) if pred else False, cer=cer(pred, gt_text) if pred else 1.0,
        conf=res.ocr.confidence if res.ocr else 0.0,
        **{f"f_{k}": v for k, v in fm.items()},
        oracle_pred=o.text, oracle_exact=exact_match(o.text, gt_text), oracle_cer=cer(o.text, gt_text), oracle_conf=o.confidence,
        gt_box=gt_box, det_box=res.detection.bbox if res.detection else None,
    )


def failure_gallery(df: pd.DataFrame, diff: str, kind: str, n: int = 8):
    """失敗例の画像を並べる。緑=正解の枠、赤=検出した枠"""
    if kind == "det":
        sub = df[(df.difficulty == diff) & (df.iou < 0.5)]
        title = "検出の失敗"
    else:
        sub = df[(df.difficulty == diff) & (df.iou >= 0.5) & (~df.exact)]
        title = "検出は成功・読み間違い"
    sub = sub.head(n)
    if sub.empty:
        return
    cols = 4
    rows = -(-len(sub) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, sub.iterrows()):
        im = Image.open(EVAL / "val" / diff / "images" / r.image).convert("RGB")
        ax.imshow(im)
        x, y, w, h = r.gt_box
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="lime", lw=1.5))
        if r.det_box is not None and not (isinstance(r.det_box, float)):
            x, y, w, h = r.det_box
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec="red", lw=1.5))
        ax.set_title(f"正:{r['gt']}\n読:{r['pred'] or '(未検出)'}", fontsize=7)
    fig.suptitle(f"{diff}: {title}（緑=正解の枠 / 赤=検出した枠）", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"failures_{kind}_{diff}.png", dpi=90)
    plt.close(fig)


def main():
    plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=100, help="先頭から除く枚数（しきい値を決めた分）")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for d in DIFFICULTIES:
            labs = load_labels(EVAL / "val" / d / "labels.jsonl")[a.skip:]
            rows += list(ex.map(evaluate_one, [("val", d, lab) for lab in labs], chunksize=8))
            print(f"{d}: {len(labs)}枚 完了")
    df = pd.DataFrame(rows)
    df.drop(columns=["gt_box", "det_box"]).to_csv(OUT / "val_per_sample.csv", index=False)

    lines = ["| 難易度 | 枚数 | 検出率(IoU≥0.5) | 平均IoU | OCRだけ 完全一致 | OCRだけ CER | 全体 完全一致 | 全体 CER |",
             "|---|---|---|---|---|---|---|---|"]
    for d in DIFFICULTIES:
        s = df[df.difficulty == d]
        lines.append(f"| {d} | {len(s)} | {(s.iou >= 0.5).mean():.3f} | {s.iou.mean():.3f} | {s.oracle_exact.mean():.3f} | "
                     f"{s.oracle_cer.mean():.3f} | {s.exact.mean():.3f} | {s.cer.mean():.3f} |")
    lines += ["", "**全体の結果の内訳**（各行の合計=100%）", "",
              "| 難易度 | 成功 | 未検出 | 検出ずれ(IoU<0.5) | 読み間違い(検出成功) |", "|---|---|---|---|---|"]
    for d in DIFFICULTIES:
        s = df[df.difficulty == d]
        ok = s.exact.mean()
        nodet = (~s.detected).mean()
        miss = (s.detected & (s.iou < 0.5)).mean()
        misread = ((s.iou >= 0.5) & ~s.exact).mean()
        lines.append(f"| {d} | {ok:.3f} | {nodet:.3f} | {miss:.3f} | {misread:.3f} |")
    lines += ["", "**フィールド別の一致率（全体）**", "", "| 難易度 | 地名 | 分類番号 | かな | 一連番号 |", "|---|---|---|---|---|"]
    for d in DIFFICULTIES:
        s = df[df.difficulty == d]
        lines.append(f"| {d} | {s.f_region.mean():.3f} | {s.f_class_no.mean():.3f} | {s.f_kana.mean():.3f} | {s.f_serial.mean():.3f} |")
    md = "\n".join(lines)
    (OUT / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    for d in DIFFICULTIES:
        failure_gallery(df, d, "det")
        failure_gallery(df, d, "ocr")


if __name__ == "__main__":
    main()
