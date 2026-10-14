"""Phase3: CRNN＋CTC の文字認識を学習する。

使い方:
    .venv/bin/python src/train_ocr.py --seed 0
    .venv/bin/python src/train_ocr.py --corner_sigma 0                # 正解の四隅だけで学習（位置ずれに弱くなるか）
    .venv/bin/python src/train_ocr.py --train_diffs easy              # 易しい条件だけで学習

  - 学習画像は、エポックごとに新しく生成する。四隅には「検出器のような誤差」を足す（--corner_sigma は上限px）。
  - エポックの選択は val の先頭100枚/難易度（正解の四隅）。評価は val の101枚目以降（Phase1・2と同じ）。test は使わない。
  - 評価は2通り: 正解の四隅（読み取りの力そのもの）／Phase2の検出器(ResNet18)が予測した四隅（システム全体）。
出力: outputs/ocr/{tag}_best.pt、{tag}_history.csv、{tag}_curve.png、{tag}_val_{gt,pred}.csv、results.csv に追記
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_detect.model import make_model as make_detector  # noqa: E402
from ai_ocr.data import OcrSynthDataset, collate  # noqa: E402
from ai_ocr.evaluate import evaluate, summarize  # noqa: E402
from ai_ocr.model import CRNN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "ocr"
DET = ROOT / "outputs" / "detect"


def load_detector(device, seed: int = 0):
    """Phase2で学習した、ResNet18（学習済み）の検出器を読む"""
    p = DET / f"resnet18_all_s{seed}_best.pt"
    if not p.exists():
        return None
    m = make_detector("resnet18").to(device)
    m.load_state_dict(torch.load(p, map_location=device))
    return m.eval()


def main():
    plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n_per_epoch", type=int, default=4500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_diffs", default="easy,medium,hard")
    ap.add_argument("--corner_sigma", type=float, default=3.0, help="学習時に四隅へ足す誤差の上限(px)。0=正解の四隅のみ")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    a.n_per_epoch = a.n_per_epoch // a.batch * a.batch
    diffs = tuple(a.train_diffs.split(","))
    tag = a.tag or f"crnn_{'-'.join(diffs) if len(diffs) < 3 else 'all'}_cs{a.corner_sigma:g}_s{a.seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = CRNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (a.n_per_epoch // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.15)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)            # CTCはMPS未対応のため、lossの計算だけCPUで行う

    per_diff = a.epochs * a.n_per_epoch // len(diffs)
    # 画像の通し番号（1難易度あたり10万まで）。収まらない場合（Easyだけで学習するなど）は、10万を超えた分は先頭に戻って再利用する
    offset = a.seed * 30000 if per_diff + a.seed * 30000 <= 100_000 else max(0, (100_000 - per_diff) * a.seed // 2)
    ds = OcrSynthDataset(a.epochs * a.n_per_epoch, offset=offset, diffs=diffs, corner_sigma_max=a.corner_sigma)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, drop_last=True, collate_fn=collate, prefetch_factor=4)
    spe = a.n_per_epoch // a.batch

    hist, best, t0, tot, n = [], (9.9, None), time.time(), 0.0, 0        # best=(最小のCER, エポック)
    model.train()
    for step, (x, y, lens) in enumerate(dl, 1):
        lp = model(x.to(device))                              # (T, B, V)
        T = lp.shape[0]
        loss = ctc(lp.cpu(), y, torch.full((x.shape[0],), T, dtype=torch.long), lens)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        tot += loss.item() * x.shape[0]; n += x.shape[0]
        if step % spe == 0:
            ep = step // spe
            sel = evaluate(model, device, "gt", start=0, stop=100)       # 選択用（val先頭100枚/難易度、正解の四隅）
            model.train()
            acc, vcer = float(sel.exact.mean()), float(sel.cer.mean())
            hist.append(dict(epoch=ep, train_loss=tot / n, val_exact=acc, val_cer=vcer))
            if vcer < best[0]:                                   # エポックの選択はCER（完全一致が0の間も選べる）
                best = (vcer, ep); torch.save(model.state_dict(), OUT / f"{tag}_best.pt")
            print(f"ep{ep:3d} loss {tot / n:.4f}  val先頭100枚 完全一致 {acc:.3f} CER {hist[-1]['val_cer']:.3f}  ({time.time() - t0:.0f}秒)", flush=True)
            tot, n = 0.0, 0

    with open(OUT / f"{tag}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ep_ = [h["epoch"] for h in hist]
    ax[0].plot(ep_, [h["train_loss"] for h in hist]); ax[0].set(title="学習 loss (CTC)", xlabel="epoch", yscale="log")
    ax[1].plot(ep_, [h["val_exact"] for h in hist]); ax[1].set(title="val先頭100枚の完全一致（正解の四隅）", xlabel="epoch", ylim=(0, 1))
    for x_ in ax: x_.grid(alpha=0.3)
    fig.suptitle(tag); fig.tight_layout(); fig.savefig(OUT / f"{tag}_curve.png", dpi=90)

    model.load_state_dict(torch.load(OUT / f"{tag}_best.pt", map_location=device))
    det = load_detector(device)
    res = OUT / "results.csv"
    new = not res.exists()
    with open(res, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "train_diffs", "corner_sigma", "seed", "epochs", "best_epoch", "corners", "difficulty", "exact", "cer", "f_region", "f_class_no", "f_kana", "f_serial"])
        for src in ("gt", "pred"):
            if src == "pred" and det is None:
                continue
            df = evaluate(model, device, src, det, start=100, stop=500)
            df.to_csv(OUT / f"{tag}_val_{src}.csv", index=False)
            s = summarize(df)
            print(f"\n四隅={'正解' if src == 'gt' else '検出器(ResNet18)の予測'}（最良エポック {best[1]}）\n{s}")
            for d, r in s.iterrows():
                w.writerow([tag, "+".join(diffs), a.corner_sigma, a.seed, a.epochs, best[1], src, d, r["完全一致"], r["CER"], r["地名"], r["分類番号"], r["かな"], r["一連番号"]])


if __name__ == "__main__":
    main()
