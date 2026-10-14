"""Phase6: 検証レポート（開発用データ val で行う。test は使わない）。

使い方:
    .venv/bin/python src/eval_report.py                 # 約5〜8分。先にPhase2〜4（モデル・較正セット）が必要
出力: outputs/report/{summary.md, system_val.csv, confusion_*.png, factor_ablation.{csv,png}, failures_*.png}

内容: ① システムの道のり（P1〜P4）の難易度別評価 ② 混同行列（どの文字を取り違えるか）
      ③ 劣化要因の単独アブレーション（どの劣化に弱いか。AI vs 古典）④ 失敗例集
"""
import argparse
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import system_eval as se  # noqa: E402
from ai_ocr.data import EvalItems  # noqa: E402
from classical import read_plate  # noqa: E402
from parking_sim.events import read_images  # noqa: E402
from plate_eval import align_ops, exact_match, iou  # noqa: E402
from plate_synth import generate_sample, split_seed  # noqa: E402
from plate_synth.difficulty import CLEAN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "report"
DIGITS, KANA, KANJI, LETTERS = list("0123456789"), list("あかさたなはまやらわ"), list("沖縄那覇品川横浜大阪"), list("ACFHKLMPXY")

# 劣化要因の単独アブレーション: (表示名, パラメータ名, 強さの段階。左がきれい → 右が強い)
FACTORS = [
    ("ノイズ σ", "noise_sigma", [0, 5, 10, 20, 30, 45]), ("ぼけ σ", "blur_sigma", [0, 0.5, 1, 1.5, 2, 3]), ("動きぶれ(px)", "motion_len", [0, 2, 4, 6, 8]),
    ("明るさ", "brightness", [1.0, 0.8, 0.6, 0.45, 0.3, 0.2]), ("コントラスト", "contrast", [1.0, 0.8, 0.6, 0.4, 0.3]),
    ("逆光・影の強さ", "grad_strength", [0, 0.3, 0.5, 0.7, 0.9]), ("汚れの数", "stain_count", [0, 1, 2, 4, 6]), ("雨滴の数", "rain_count", [0, 10, 25, 50, 80]),
    ("隠れ(線の数)", "occlusion_lines", [0, 1, 2, 3, 4]), ("低解像度(縮小率)", "lowres", [1, 1.5, 2, 3, 4]), ("JPEG品質", "jpeg_quality", [95, 60, 30, 15, 8]),
    ("回転(度)", "rot_deg", [0, 5, 10, 15, 25]), ("透視の歪み", "persp", [0, 0.05, 0.1, 0.15, 0.2]), ("板の大きさ(画面幅比)", "scale", [0.5, 0.4, 0.3, 0.2, 0.15]),
]


# ------------------------------------------------------------------ ② 混同行列
def confusion_tables(frames: pd.DataFrame):
    """gt と raw（AIの読み取り）を文字ごとに対応付け、(正解の文字, 読んだ文字) の回数を数える。難易度別"""
    out = {d: defaultdict(Counter) for d in ("medium", "hard")}
    stats = {d: Counter() for d in ("medium", "hard")}
    for d, g, p in zip(frames.difficulty, frames["gt"], frames["raw"].fillna("")):
        if d not in out:
            continue
        for op, a, b in align_ops(g, p):
            stats[d][op] += 1
            if op == "sub":
                out[d][a][b] += 1
            elif op == "del":
                out[d][a]["(脱落)"] += 1
            elif op == "ins":
                out[d]["(余分)"][b] += 1
    return out, stats


def plot_confusion(conf: dict, chars: list, title: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    cols = chars + ["(脱落)"]
    for ax, d in zip(axes, ("medium", "hard")):
        m = np.array([[conf[d][r].get(c, 0) for c in cols] for r in chars], dtype=float)
        tot = np.array([sum(conf[d][r].values()) + 0.0 for r in chars])        # 取り違え・脱落の合計（一致は含まない）
        ax.imshow(np.log1p(m), cmap="Reds")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8); ax.set_yticks(range(len(chars))); ax.set_yticklabels(chars, fontsize=8)
        for i in range(len(chars)):
            for j in range(len(cols)):
                if m[i, j] >= max(3, 0.08 * tot[i]):
                    ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=6.5)
        ax.set(title=f"{d}: 正解（縦）→ 読み取り（横）", xlabel="読んだ文字", ylabel="正解の文字")
    fig.suptitle(title + "（色=誤りの回数の対数。数字=回数）", fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)


# ------------------------------------------------------------------ ③ 劣化要因の単独アブレーション
def _gen(job):
    seed, ov = job
    return np.asarray(generate_sample(seed, "easy", overrides=ov).image)


def factor_ablation(det, ocr, device, n: int = 80) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for name, key, levels in FACTORS:
            for lv in levels:
                ov = {**CLEAN, "bg_type": "car", "scale": 0.4}
                ov[key] = lv
                seeds = [split_seed("val", "easy", 3000 + i) for i in range(n)]       # val の3000番目以降（較正セット・評価データ・testと重ならない）
                gts = [generate_sample(sd, "easy").label["text"].replace(" ", "") for sd in seeds]
                imgs = list(ex.map(_gen, [(sd, ov) for sd in seeds], chunksize=8))
                ai = read_images(imgs, det, ocr, device)
                p1 = [read_plate(Image.fromarray(a)).text.replace(" ", "") for a in imgs]
                rows.append(dict(要因=name, 値=lv, AI=float(np.mean([a["text"] == g for a, g in zip(ai, gts)])),
                                 古典=float(np.mean([exact_match(t, g) if t else False for t, g in zip(p1, gts)]))))
        print("劣化要因の単独アブレーション: 完了", flush=True)
    return pd.DataFrame(rows)


def plot_factors(df: pd.DataFrame, path: Path):
    n = len(FACTORS)
    fig, axes = plt.subplots(3, 5, figsize=(17, 8.5))
    for ax, (name, key, levels) in zip(axes.flat, FACTORS):
        x = df[df.要因 == name]
        ax.plot(range(len(levels)), x.AI, "o-", label="AI（検出＋CRNN）"); ax.plot(range(len(levels)), x.古典, "s--", label="古典（検出＋テンプレートOCR）")
        ax.set_xticks(range(len(levels))); ax.set_xticklabels([f"{v:g}" for v in levels], fontsize=8)
        ax.set(title=name, ylim=(-0.03, 1.03)); ax.grid(alpha=0.3)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7); fig.suptitle("劣化要因を1つずつ強くしたときの、読み取り（完全一致）。左がきれい → 右が強い（Easyの基準、板の大きさ40%）", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=85); plt.close(fig)


# ------------------------------------------------------------------ ④ 失敗例集
def gallery(df: pd.DataFrame, items: list, mask, title: str, path: Path, seed: int = 0, n: int = 8):
    sub = df[mask].head(n)
    if sub.empty:
        return 0
    paths = {(d, lab["image"]): p for d, p, lab in items}
    cols = 4
    fig, axes = plt.subplots(-(-len(sub) // cols), cols, figsize=(3.8 * cols, 3.5 * -(-len(sub) // cols)))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, r) in zip(axes, sub.iterrows()):
        ax.imshow(Image.open(paths[(r.difficulty, r.image)]).convert("RGB"))
        pc = np.array(r.corners_pred)
        ax.plot(*np.vstack([pc, pc[:1]]).T, "-", color="red", lw=1.2)
        ax.set_title(f"[{r.difficulty}] 正:{r['gt']}\n読:{r[f'p3_text_s{seed}'] or '(空)'}  確信度{r[f'conf_s{seed}']:.2f}", fontsize=7.5)
    fig.suptitle(title, fontsize=10); fig.tight_layout(); fig.savefig(path, dpi=88); plt.close(fig)
    return len(df[mask])


def envelope_table(fa: pd.DataFrame, thr: float) -> list[str]:
    """AIの完全一致が thr を下回る最初の強さ。運用できる範囲（カメラの仕様）の目安"""
    lines = [f"| 要因 | AI（{thr:.1f}を下回る最初の強さ） | 古典 | AIが{thr:.1f}以上を保つ最後の強さ |", "|---|---|---|---|"]
    for name, key, levels in FACTORS:
        x = fa[fa.要因 == name]
        first = lambda col: next((f"{v:g}" for v, p in zip(x.値, x[col]) if p < thr), "—")
        last_ok = "—"
        for v, p in zip(x.値, x.AI):
            if p >= thr:
                last_ok = f"{v:g}"
            else:
                break
        lines.append(f"| {name} | {first('AI')} | {first('古典')} | {last_ok} |")
    return lines


def main():
    plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_factors", action="store_true")
    ap.add_argument("--envelope_only", action="store_true", help="劣化要因のCSVから、運用できる範囲の表だけを summary.md に追記する（再計算しない）")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.envelope_only:
        fa = pd.read_csv(OUT / "factor_ablation.csv")
        txt = (OUT / "summary.md").read_text(encoding="utf-8")
        if "運用できる範囲" not in txt:
            extra = ["", "## 運用できる範囲の目安（AIの完全一致が0.9以上を保つ強さ。Easyの基準・板の大きさ40%）", ""] + envelope_table(fa, 0.9) + [""]
            (OUT / "summary.md").write_text(txt.rstrip("\n") + "\n" + "\n".join(extra), encoding="utf-8")
        print("\n".join(envelope_table(fa, 0.9)))
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    seeds = [0, 1, 2]
    det, ocrs = se.load_models(device)
    taus = se.thresholds(seeds)
    lines = [f"# 検証レポート（開発用データ val の101枚目以降、各400枚。test は使っていない）", "", f"自動処理のしきい値（較正セットで決定。目標の誤り率 {se.TARGET_RISK:.0%}・昼間中心）: " + "、".join(f"CRNNシード{s}: {t:.4f}" for s, t in taus.items()), ""]

    # ① システムの道のり
    items = EvalItems("val", 100, 500).items
    df = se.evaluate_system(items, det, ocrs, taus, device)
    df.to_csv(OUT / "system_val.csv", index=False)
    T = se.tables(df, seeds)
    lines += ["## ① システムの道のり（完全一致。P3は3シード平均）", "", se.md(T["path"]), "", "## 検出", "", se.md(T["det"]), "", "## フィールド別の正解率（P3・3シード平均）", "", se.md(T["fields"]),
              "", "## 振り分け（P4。難易度の混ぜ方別・3シード平均±標準偏差）", "", se.md(T["decision"]), "", "## 振り分け（難易度別・3シード平均）", "", se.md(T["decision_by_diff"]), ""]
    print("① 完了", flush=True)

    # ② 混同行列（較正セットの読み取り結果を使う。枚数が多く安定）
    cal = pd.concat([pd.read_csv(ROOT / "outputs" / "rules" / f"seed{s}_cal_samples.csv") for s in seeds], ignore_index=True)
    conf, stats = confusion_tables(cal)
    plot_confusion(conf, DIGITS + ["-", "・"], "数字・記号の混同行列", OUT / "confusion_digits.png")
    plot_confusion(conf, KANA, "かなの混同行列", OUT / "confusion_kana.png")
    plot_confusion(conf, KANJI + LETTERS, "地名の漢字・分類番号の英字の混同行列", OUT / "confusion_kanji_letters.png")
    lines += ["## ② 混同行列: よくある取り違え（較正セット・3シード、AIの読み取り）", ""]
    for d in ("medium", "hard"):
        c = Counter()
        for r, row in conf[d].items():
            for k, v in row.items():
                c[(r, k)] += v
        total_chars = sum(stats[d].values())
        lines += [f"**{d}**: 文字の対応付け {total_chars:,}文字のうち、一致 {stats[d]['match'] / total_chars:.1%}、取り違え {stats[d]['sub'] / total_chars:.2%}、脱落 {stats[d]['del'] / total_chars:.2%}、余分 {stats[d]['ins'] / total_chars:.2%}", "",
                  "| 順位 | 正解 → 読み取り | 回数 |", "|---|---|---|"]
        lines += [f"| {i + 1} | {r} → {k} | {v} |" for i, ((r, k), v) in enumerate(c.most_common(10))]
        lines.append("")
    print("② 完了", flush=True)

    # ③ 劣化要因の単独アブレーション
    if not a.skip_factors:
        fa = factor_ablation(det, ocrs[0], device)
        fa.to_csv(OUT / "factor_ablation.csv", index=False)
        plot_factors(fa, OUT / "factor_ablation.png")
        lines += ["## ③ 劣化要因の単独アブレーション（Easyの基準に、1つだけ劣化を足す。完全一致。80枚/点）", "", "| 要因 | 値 | AI | 古典 |", "|---|---|---|---|"]
        lines += [f"| {r.要因} | {r.値:g} | {r.AI:.3f} | {r.古典:.3f} |" for r in fa.itertuples()]
        # 「AIの完全一致が0.5を下回る最初の強さ」
        lines += ["", "**AIの完全一致が0.5を下回る最初の強さ**（—は、範囲内で下回らない）", "", "| 要因 | AI | 古典 |", "|---|---|---|"]
        for name, key, levels in FACTORS:
            x = fa[fa.要因 == name]
            f = lambda col: next((f"{v:g}" for v, p in zip(x.値, x[col]) if p < 0.5), "—")
            lines.append(f"| {name} | {f('AI')} | {f('古典')} |")
        lines += ["", "## 運用できる範囲の目安（AIの完全一致が0.9以上を保つ強さ。Easyの基準・板の大きさ40%）", ""] + envelope_table(fa, 0.9) + [""]
        print("③ 完了", flush=True)

    # ④ 失敗例集
    wrong = df["p3_text_s0"] != df["gt"]
    cats = [("failures_confident.png", "自信満々の誤り（自動処理と判断されたのに間違い）", (df.decision_s0 == "自動処理") & wrong),
            ("failures_review.png", "要確認に回った例（確信度が低い）", df.decision_s0 == "要確認"),
            ("failures_reject.png", "読み取り不可（書式が直せない）の例", df.decision_s0 == "読み取り不可"),
            ("failures_corner.png", "四隅の誤差が大きい検出（誤差10px以上）", df.corner_err >= 10)]
    lines += ["## ④ 失敗例集（画像は outputs/report/failures_*.png）", "", "| 種類 | 件数（評価データ1,200枚中） |", "|---|---|"]
    for fn, title, mask in cats:
        n = gallery(df, items, mask, title, OUT / fn)
        lines.append(f"| {title} | {n} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
