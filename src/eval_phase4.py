"""Phase4: ルール補正・確信度・振り分けの評価。

使い方:
    .venv/bin/python src/make_eval_sets.py --val 0 --test 0 --cal 1000   # 較正セットを作る（初回のみ）
    .venv/bin/python src/eval_phase4.py            # CRNN 3シード（Phase3の標準モデル）× 検出器の四隅（Phase2）。約1分
出力: outputs/rules/{seed*_samples.csv, summary.md, risk_coverage.png}

手順: val 0〜500番目/難易度を CRNN で読む → 書式検査・補正 → 確信度4通り。
  - しきい値は「val の先頭100枚/難易度」または「較正セット（valのシードの500番目以降、1,000枚/難易度）」で決め、評価は val の101枚目以降（Phase1〜3と同じ）。test は使わない。
  - 難易度の混ぜ方は2通り: 均等（1:1:1）／昼間中心（Easy 0.6・Medium 0.3・Hard 0.1）
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_detect.model import make_model as make_detector  # noqa: E402
from ai_ocr.data import EvalItems  # noqa: E402
from ai_ocr.evaluate import predict_corners  # noqa: E402
from ai_ocr.infer import load_plates, read_plates  # noqa: E402
from ai_ocr.model import CRNN  # noqa: E402
from rules import confidence as conf_mod  # noqa: E402
from rules.correct import correct  # noqa: E402
from rules.decision import apply, auroc, choose_threshold, choose_threshold_by_cost, risk_coverage  # noqa: E402
from rules.format import is_valid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "rules"
DIFFS = ["easy", "medium", "hard"]
MIXES = {"均等(1:1:1)": {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3}, "昼間中心(0.6:0.3:0.1)": {"easy": 0.6, "medium": 0.3, "hard": 0.1}}
N_SEL = 100                         # 各難易度の先頭100枚を、しきい値の選択に使う


def sample_table(model, device, items, corners, seed: int) -> pd.DataFrame:
    readings = read_plates(model, device, load_plates(items, corners))
    rows = []
    cnt = {d: 0 for d in DIFFS}
    for (d, path, lab), r in zip(items, readings):
        gt = lab["text"].replace(" ", "")
        c = correct(r.emissions, r.text)
        raw_c = conf_mod.confidences(r.emissions, r.logp, r.text)
        fin_c = raw_c if c.status == "ok" else (conf_mod.confidences(r.emissions, r.logp, c.text, c.probs) if c.status == "corrected" else raw_c)
        row = dict(seed=seed, difficulty=d, idx=cnt[d], image=lab["image"], gt=gt, raw=r.text, raw_valid=is_valid(r.text),
                   status=c.status, final=c.text, changes=";".join(f"{i}:{a}>{b}" for i, a, b in c.changes),
                   exact_raw=r.text == gt, exact_final=c.text == gt, valid_final=c.status != "invalid")
        row.update({f"raw_{k}": v for k, v in raw_c.items()})
        row.update({f"fin_{k}": v for k, v in fin_c.items()})
        rows.append(row)
        cnt[d] += 1
    return pd.DataFrame(rows)


def weights(df: pd.DataFrame, mix: dict) -> np.ndarray:
    n = df.difficulty.value_counts()
    return df.difficulty.map(lambda d: mix[d] / n[d]).to_numpy()


def policies():
    """比べる方式: (名前, 正解判定の列, 書式検査に使う列, 確信度の列)"""
    out = [("A: 補正なし・検査なし・最小確率（Phase3）", "exact_raw", None, "raw_min_max"),
           ("B: 書式検査のみ・最小確率", "exact_raw", "raw_valid", "raw_min_max"),
           ("B2: 書式検査のみ（補正しない）・seq", "exact_raw", "raw_valid", "raw_seq")]
    for k in conf_mod.NAMES:
        out.append((f"C: 補正＋検査・{k}", "exact_final", "valid_final", f"fin_{k}"))
    return out


def evaluate_policies(sel: pd.DataFrame, ev: pd.DataFrame, mixname: str, targets=(0.02, 0.01, 0.005), only=None) -> pd.DataFrame:
    """sel（しきい値を決めるデータ）でしきい値を選び、ev（評価データ）で実現した結果を測る"""
    ws, we = weights(sel, MIXES[mixname]), weights(ev, MIXES[mixname])
    rows = []
    for name, ok_col, valid_col, conf_col in policies():
        if only and name not in only:
            continue
        vs = sel[valid_col].to_numpy() if valid_col else np.ones(len(sel), bool)
        ve = ev[valid_col].to_numpy() if valid_col else np.ones(len(ev), bool)
        for tr in targets:
            tau = choose_threshold(sel[conf_col], sel[ok_col], vs, tr, ws)
            r = apply(ev[conf_col], ev[ok_col], ve, tau, we)
            rows.append(dict(方式=name, 目標の誤り率=tr, しきい値=round(tau, 4), **{k: round(v, 4) for k, v in r.items()}))
    return pd.DataFrame(rows)


def cost_table(cal: pd.DataFrame, ev: pd.DataFrame, mixname: str, cost_error: float, cost_review: float = 1.0) -> pd.DataFrame:
    """1台あたりの期待コスト（要確認・読み取り不可=人が対応=cost_review、自動処理の誤り=cost_error）"""
    wc, we = weights(cal, MIXES[mixname]), weights(ev, MIXES[mixname])
    rows = []
    for name, ok_col, valid_col, conf_col in policies():
        vc = cal[valid_col].to_numpy() if valid_col else np.ones(len(cal), bool)
        ve = ev[valid_col].to_numpy() if valid_col else np.ones(len(ev), bool)
        tau = choose_threshold_by_cost(cal[conf_col], cal[ok_col], vc, cost_review, cost_error, wc)
        r = apply(ev[conf_col], ev[ok_col], ve, tau, we)
        rows.append(dict(方式=name, しきい値=round(tau, 4), 自動処理率=r["自動処理率"], 自動処理の誤り率=r["自動処理の誤り率"],
                         期待コスト=(1 - r["自動処理率"]) * cost_review + r["誤自動処理率"] * cost_error))
    return pd.DataFrame(rows)


def main():
    plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    det = make_detector("resnet18").to(device)
    det.load_state_dict(torch.load(ROOT / "outputs" / "detect" / "resnet18_all_s0_best.pt", map_location=device))
    ev_items = EvalItems("val", 0, 500)
    cal_items = EvalItems("valcal", 0, 1000)                   # 較正用: val のシードの500番目以降（make_eval_sets.py --cal 1000）
    ev_c, cal_c = predict_corners(det, ev_items.items, device), predict_corners(det, cal_items.items, device)

    evs, cals = [], []
    for sd in a.seeds:
        m = CRNN().to(device)
        m.load_state_dict(torch.load(ROOT / "outputs" / "ocr" / f"crnn_all_cs3_s{sd}_best.pt", map_location=device))
        t = sample_table(m, device, ev_items.items, ev_c, sd)
        c = sample_table(m, device, cal_items.items, cal_c, sd)
        t.to_csv(OUT / f"seed{sd}_samples.csv", index=False); c.to_csv(OUT / f"seed{sd}_cal_samples.csv", index=False)
        evs.append(t); cals.append(c)
        print(f"seed{sd}: 完了", flush=True)
    allt = pd.concat(evs)
    e = allt[allt.idx >= N_SEL]
    lines = ["## 書式検査・補正の効果（評価: val の101枚目以降、3シード平均）", "",
             "| 難易度 | そのまま正しい書式 | 補正して正しい書式 | 直せない | 完全一致（補正前） | 完全一致（補正後） | 補正した画像の正解率 |", "|---|---|---|---|---|---|---|"]
    for d in DIFFS:
        x = e[e.difficulty == d]
        c = x[x.status == "corrected"]
        lines.append(f"| {d} | {(x.status == 'ok').mean():.3f} | {(x.status == 'corrected').mean():.3f} | {(x.status == 'invalid').mean():.3f} | "
                     f"{x.exact_raw.mean():.3f} | {x.exact_final.mean():.3f} | {c.exact_final.mean() if len(c) else float('nan'):.3f}（{len(c)}枚）|")
    # 確信度の分離能力（AUROC）: 較正セット（3,000枚）で測る（評価セットは誤りが少なく不安定なため）
    lines += ["", "## 確信度が、正解と誤りをどれくらい分けるか（AUROC。0.5=偶然、1=完全。較正セット1,000枚/難易度、補正後・書式が正しいものだけ、3シード平均）", "",
              "| 確信度 | " + " | ".join(DIFFS) + " | 全体 |", "|---|---|---|---|---|"]
    call = pd.concat(cals)
    for k in conf_mod.NAMES:
        vals = []
        for d in DIFFS + [None]:
            x = call[(call.difficulty == d) if d else slice(None)]
            x = x[x.valid_final]
            v = [auroc(x[x.seed == s][f"fin_{k}"], x[x.seed == s].exact_final) for s in a.seeds]
            vals.append(f"{np.nanmean(v):.3f}")
        lines.append(f"| {k} | " + " | ".join(vals) + " |")
    # しきい値の決め方: 先頭100枚（300枚）で決める vs 較正セット（3,000枚）で決める
    for mixname in MIXES:
        for label, sel_fn in (("S: 先頭100枚/難易度（300枚）で決める", lambda i: evs[i][evs[i].idx < N_SEL]), ("C: 較正セット（3,000枚）で決める", lambda i: cals[i])):
            res = [evaluate_policies(sel_fn(i), evs[i][evs[i].idx >= N_SEL], mixname, only=None).assign(seed=a.seeds[i]) for i in range(len(a.seeds))]
            r = pd.concat(res).groupby(["方式", "目標の誤り率"], sort=False).agg(
                自動処理率=("自動処理率", "mean"), 実現した誤り率=("自動処理の誤り率", "mean"), 要確認率=("要確認率", "mean"),
                読み取り不可率=("読み取り不可率", "mean"), 誤自動処理率=("誤自動処理率", "mean")).round(4).reset_index()
            lines += ["", f"## 振り分けの結果 [{mixname}] しきい値の決め方 = {label}（評価は101枚目以降、3シード平均）", "",
                      "| 方式 | 目標の誤り率 | 自動処理率 | 実現した誤り率 | 要確認率 | 読み取り不可率 | 全体に対する誤自動処理率 |", "|---|---|---|---|---|---|---|"]
            for _, x in r.iterrows():
                lines.append(f"| {x['方式']} | {x['目標の誤り率']:.3f} | {x['自動処理率']:.3f} | {x['実現した誤り率']:.4f} | {x['要確認率']:.3f} | {x['読み取り不可率']:.3f} | {x['誤自動処理率']:.4f} |")
    # コストで決める
    for mixname in MIXES:
        for ce in (5, 20, 100):
            r = pd.concat([cost_table(cals[i], evs[i][evs[i].idx >= N_SEL], mixname, ce).assign(seed=a.seeds[i]) for i in range(len(a.seeds))]).groupby("方式", sort=False).mean(numeric_only=True).round(4).reset_index()
            lines += ["", f"## コストで決める [{mixname}] 誤り1件のコスト = 要確認1件の{ce}倍（較正セットで決定、評価は101枚目以降、3シード平均）", "",
                      "| 方式 | しきい値 | 自動処理率 | 自動処理の誤り率 | 1台あたりの期待コスト |", "|---|---|---|---|---|"]
            for _, x in r.iterrows():
                lines.append(f"| {x['方式']} | {x['しきい値']:.4f} | {x['自動処理率']:.3f} | {x['自動処理の誤り率']:.4f} | {x['期待コスト']:.4f} |")
            lines.append(f"\n（参考: 全部を人が確認すると 1.0000／全部を自動処理すると、誤り率分×{ce} になる）")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    # リスク-カバレッジ曲線（評価セット、seed0）
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, mixname in zip(axes, MIXES):
        x = e[e.seed == a.seeds[0]]
        we = weights(x, MIXES[mixname])
        for name, ok_col, valid_col, conf_col in policies():
            v = x[valid_col].to_numpy() if valid_col else None
            cov, err, _ = risk_coverage(x[conf_col], x[ok_col], v, we)
            ax.plot(cov, err, label=name, lw=1.6)
        ax.set(title=f"カバレッジ曲線: {mixname}", xlabel="自動処理率（カバレッジ）", ylabel="自動処理した中の誤り率", ylim=(0, 0.3), xlim=(0, 1))
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(OUT / "risk_coverage.png", dpi=90)


if __name__ == "__main__":
    main()
