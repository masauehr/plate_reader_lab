"""Phase5: 入出庫シミュレーションの本実験。

使い方:
    .venv/bin/python src/run_phase5.py            # 4シナリオ × 3つの乱数 × 4つの照合方針（約15分。先にPhase2〜4が必要）
出力: outputs/sim/{events_*.csv, results.csv, summary.md, tradeoff.png}

  - 事象（入出庫の画像をAIで読んだ結果）は、シナリオ×乱数ごとに1回だけ作って保存し、4つの方針を同じ事象で比べる。
  - 自動処理のしきい値は、Phase4の方法（昼間中心の想定・目標の誤り率・較正セット）で決める。
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
from parking_sim.events import build_event_table, load_models  # noqa: E402
from parking_sim.policy import POLICIES  # noqa: E402
from parking_sim.setup import auto_threshold as threshold, to_events  # noqa: E402
from parking_sim.simulate import simulate  # noqa: E402
from parking_sim.world import MIX_DAY, MIX_EQUAL, make_world  # noqa: E402
from rules.decision import decide  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "sim"
SCENARIOS = {
    "S1 ランダムなナンバー・昼間中心": dict(similar_fraction=0.0, mix=MIX_DAY),
    "S2 似たナンバー30%・昼間中心": dict(similar_fraction=0.3, mix=MIX_DAY),
    "S3 似たナンバー30%・悪条件多め（均等）": dict(similar_fraction=0.3, mix=MIX_EQUAL),
    "S4 似たナンバー80%（車列など）・昼間中心【ストレス】": dict(similar_fraction=0.8, mix=MIX_DAY),
}
STRESS = "S4 似たナンバー80%（車列など）・昼間中心【ストレス】"


def main():
    plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2500, help="1回のシミュレーションの車の台数")
    ap.add_argument("--sims", type=int, default=3)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--cost_error", type=float, default=20.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    det = ocr = None
    tau = threshold(a.target)
    print(f"自動処理のしきい値 τ={tau:.4f}（目標の誤り率 {a.target:.1%}、Phase4の方法）", flush=True)

    tables, rows = {}, []
    for sname, cfg in SCENARIOS.items():
        for sd in range(a.sims):
            f = OUT / f"events_{sname[:2]}_s{sd}.csv"
            if f.exists():
                tab = pd.read_csv(f, keep_default_na=False)
            else:
                if det is None:
                    det, ocr = load_models(device)
                world = make_world(a.n, sd, cfg["similar_fraction"], cfg["mix"])
                tab = build_event_table(world, sd, det, ocr, device)
                tab.to_csv(f, index=False)
            tab["text"] = tab["text"].astype(str)
            tables[(sname, sd)] = tab
            ev = to_events(tab, tau)
            for pol in POLICIES:
                rows.append(dict(シナリオ=sname, sim=sd, 方針=pol.name, **simulate(ev, pol, a.cost_error)))
            print(f"{sname} sim{sd}: 完了", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)

    cols = ["自動精算率", "誤課金率", "要確認率", "読み取り不可_未照合率", "人手対応率", "期待コスト"]
    lines = [f"自動処理のしきい値 τ={tau:.4f}（目標の誤り率 {a.target:.1%}）／1回あたり {a.n} 台×{a.sims}回／誤課金1件のコスト={a.cost_error:g}（要確認1件=1）", ""]
    for sname in SCENARIOS:
        g = res[res.シナリオ == sname].groupby("方針", sort=False)
        m, sd_ = g[cols].mean(), g[cols].std()
        lines += [f"## {sname}", "", "| 方針 | 自動精算率 | **誤課金率** | 誤課金の件数 | 要確認率 | 読み取り不可・未照合率 | 人手対応率 | 期待コスト |", "|---|---|---|---|---|---|---|---|"]
        for p in m.index:
            xx = res[(res.シナリオ == sname) & (res.方針 == p)]
            cnt = int(round((xx.誤課金率 * xx.台数).sum()))
            lines.append(f"| {p} | {m.loc[p, '自動精算率']:.4f} | **{m.loc[p, '誤課金率']:.4f}** ±{sd_.loc[p, '誤課金率']:.4f} | {cnt}／{int(xx.台数.sum())}件 | {m.loc[p, '要確認率']:.4f} | "
                         f"{m.loc[p, '読み取り不可_未照合率']:.4f} | {m.loc[p, '人手対応率']:.4f} | {m.loc[p, '期待コスト']:.4f} |")
        x = res[res.シナリオ == sname]
        lines += ["", f"（入庫の自動処理率 {x.入庫の自動処理率.mean():.3f}、入庫で読み間違いを記録した率 {x.入庫の読み間違い記録率.mean():.4f}、最大同時駐車台数 {x.最大同時駐車台数.mean():.0f}台）", ""]
    # 読み取りレベルの誤り vs システムレベルの誤課金
    lines += ["## 読み取りの「自信満々の誤り」は、システムでどれだけ誤課金になったか（P1）", ""]
    for sname in (list(SCENARIOS)[1], STRESS):
        ex = pd.concat([tables[(sname, s_)] for s_ in range(a.sims)])
        ex = ex[ex["kind"] == "out"].copy()
        ex["decision"] = [decide(bool(v), float(c), tau) for v, c in zip(ex["valid"], ex["conf"])]
        auto = ex[ex["decision"] == "自動処理"]
        wrong = int((auto["text"] != auto["gt"]).sum())              # ※ 列名 gt は pandas のメソッド名と衝突するので、角括弧で取り出す
        bill = res[(res.シナリオ == sname) & res.方針.str.startswith("P1")]
        lines.append(f"- {sname}: 出庫の読み取りで「自動処理」と判断された {len(auto)} 件のうち、読み間違いは {wrong} 件（{wrong / len(auto):.2%}）。"
                     f"一方、システム全体の誤課金は {int(round((bill.誤課金率 * bill.台数).sum()))} 件（出庫 {int(bill.台数.sum())} 件に対して {bill.誤課金率.mean():.4%}）。")
    # 目標の誤り率を変えたときの、人手と誤課金のトレードオフ（誤課金が起きやすいストレスシナリオ）
    sweep = []
    for tr in (0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        t_ = threshold(tr)
        for pol in POLICIES:
            for sd in range(a.sims):
                r = simulate(to_events(tables[(STRESS, sd)], t_), pol, a.cost_error)
                sweep.append(dict(目標=tr, しきい値=t_, 方針=pol.name, 誤課金率=r["誤課金率"], 人手対応率=r["人手対応率"], 期待コスト=r["期待コスト"]))
    sw = pd.DataFrame(sweep).groupby(["方針", "目標"], sort=False).mean(numeric_only=True).reset_index()
    sw.to_csv(OUT / "sweep.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for pol in POLICIES:
        s_ = sw[sw.方針 == pol.name]
        ax[0].plot(s_.人手対応率, s_.誤課金率 * 100, "o-", label=pol.name)
        ax[1].plot(s_.目標 * 100, s_.期待コスト, "o-", label=pol.name)
    ax[0].set(xlabel="人手対応率（要確認＋読み取り不可・未照合）", ylabel="誤課金率（%）", title="S4(ストレス): 人手の負担と誤課金のトレードオフ"); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="自動処理の目標誤り率（%）", ylabel=f"1台あたりの期待コスト（誤課金={a.cost_error:g}）", title="S4(ストレス): 目標の誤り率と期待コスト", xscale="log"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "tradeoff.png", dpi=90)
    lines += ["", "## 目標の誤り率を変えたとき（S4ストレス、3回平均）", "", "| 方針 | 目標の誤り率 | しきい値 | 誤課金率 | 人手対応率 | 期待コスト |", "|---|---|---|---|---|---|"]
    for _, r in sw.iterrows():
        lines.append(f"| {r['方針']} | {r['目標']:.3f} | {r['しきい値']:.3f} | {r['誤課金率']:.4f} | {r['人手対応率']:.4f} | {r['期待コスト']:.4f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
