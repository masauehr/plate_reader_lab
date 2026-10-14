"""Phase2: 板の四隅を回帰するCNNを学習する。

使い方:
    .venv/bin/python src/train_detector.py --model small --epochs 20 --seed 0
    .venv/bin/python src/train_detector.py --model resnet18 --epochs 20 --seed 0
    .venv/bin/python src/train_detector.py --model resnet18 --train_diffs easy   # 易しい条件だけで学習

出力: outputs/detect/{tag}_best.pt（git管理外）、{tag}_history.csv、{tag}_curve.png、results.csv に追記
  - エポックの選択は val の先頭100枚/難易度（学習には使わない）。評価は val の101枚目以降（Phase1と同じ）。
  - 学習画像はエポックごとに新しく生成する（同じ画像を繰り返し見ない）。test は使わない。
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
from ai_detect.data import FileDataset, SynthDataset  # noqa: E402
from ai_detect.evaluate import evaluate, mean_iou_only, summarize  # noqa: E402
from ai_detect.model import make_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "detect"


def main():
    plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=["small", "resnet18", "resnet18_scratch"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--n_per_epoch", type=int, default=4500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_diffs", default="easy,medium,hard")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    a.n_per_epoch = a.n_per_epoch // a.batch * a.batch       # バッチの整数倍にそろえる
    diffs = tuple(a.train_diffs.split(","))
    lr = a.lr or (1e-3 if a.model == "small" else 3e-4)
    tag = a.tag or f"{a.model}_{'-'.join(diffs) if len(diffs) < 3 else 'all'}_s{a.seed}"
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = make_model(a.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps = a.epochs * (a.n_per_epoch // a.batch)           # 学習全体のステップ数（学習率の計画に使う）
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    loss_fn = nn.SmoothL1Loss(beta=0.01)
    sel = FileDataset("val", 0, 100)                       # エポック選択用（val先頭100枚/難易度）

    hist, best, t0 = [], (-1.0, None), time.time()
    # 学習全体で1つのDataLoader（ワーカーを作り直さない）。k番目の画像は、難易度を順番に混ぜ、通し番号は
    # シードごとにずらした範囲から取る（毎回新しい画像。1難易度あたり最大10万枚、シード0〜2で重ならない）
    per_diff = a.epochs * a.n_per_epoch // len(diffs)
    assert per_diff <= 100_000, "学習画像の通し番号が上限(10万/難易度)を超えます。epochs・n_per_epoch を減らしてください"
    if per_diff + a.seed * 30000 <= 100_000:
        offset = a.seed * 30000                              # 通常: シードごとに3万ずつずらす（シード0〜2で重ならない）
    else:
        offset = (100_000 - per_diff) * a.seed // 2          # 1難易度だけで学習するときなど: 上限に収まる範囲に等間隔で割り当てる
    ds = SynthDataset(a.epochs * a.n_per_epoch, offset=offset, diffs=diffs)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=a.workers, drop_last=True, prefetch_factor=4)
    steps_per_epoch = a.n_per_epoch // a.batch
    model.train()
    tot, n = 0.0, 0
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item() * len(y); n += len(y)
        if step % steps_per_epoch == 0:                       # 1エポックぶん学習したら、選択用の評価
            ep = step // steps_per_epoch
            val_iou = mean_iou_only(model, sel, device)
            model.train()
            hist.append(dict(epoch=ep, train_loss=tot / n, val_iou=val_iou))
            if val_iou > best[0]:
                best = (val_iou, ep)
                torch.save(model.state_dict(), OUT / f"{tag}_best.pt")
            print(f"ep{ep:3d} loss {tot / n:.5f}  val先頭100枚 平均IoU {val_iou:.3f}  ({time.time() - t0:.0f}秒)", flush=True)
            tot, n = 0.0, 0

    with open(OUT / f"{tag}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys()); w.writeheader(); w.writerows(hist)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist]); ax[0].set(title="学習 loss", xlabel="epoch", yscale="log")
    ax[1].plot([h["epoch"] for h in hist], [h["val_iou"] for h in hist]); ax[1].set(title="val先頭100枚の平均IoU", xlabel="epoch", ylim=(0, 1))
    for x_ in ax: x_.grid(alpha=0.3)
    fig.suptitle(tag); fig.tight_layout(); fig.savefig(OUT / f"{tag}_curve.png", dpi=90)

    model.load_state_dict(torch.load(OUT / f"{tag}_best.pt", map_location=device))      # 最良のエポックで評価
    df = evaluate(model, device, start=100, stop=500)
    df.to_csv(OUT / f"{tag}_val_per_sample.csv", index=False)
    s = summarize(df)
    print(f"\n最良エポック: {best[1]}（先頭100枚の平均IoU {best[0]:.3f}）／ 評価は val の101枚目以降\n{s}")
    res = OUT / "results.csv"
    new = not res.exists()
    with open(res, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["tag", "model", "train_diffs", "seed", "epochs", "n_per_epoch", "best_epoch", "difficulty", "mean_iou", "det50", "det90", "corner_err", "exact"])
        for d, r in s.iterrows():
            w.writerow([tag, a.model, "+".join(diffs), a.seed, a.epochs, a.n_per_epoch, best[1], d, r["平均IoU"], r["Det@0.5"], r["Det@0.9"], r["四隅誤差(px)"], r["読み取り完全一致"]])


if __name__ == "__main__":
    main()
