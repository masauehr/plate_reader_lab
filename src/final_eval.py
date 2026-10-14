"""最終評価。**test データを使うのは、プロジェクト全体で1回だけ。**

使い方:
    .venv/bin/python src/final_eval.py                           # 開発用の予行演習（val の101枚目以降）。何度でも実行できる → outputs/final_dev/
    .venv/bin/python src/final_eval.py --split test --confirm-final   # 最終評価（test）。1回だけ → outputs/final/

設計（testを守る仕組み）:
  - モデル・しきい値・方針は、実行前にすべて決まっている（検出器=ResNet18のシード0、OCR=CRNNのシード0〜2、しきい値=各シードの較正セットで決定）。
  - testの結果を見て、モデルやしきい値を調整しない。
  - 使用済みの記録（outputs/final/TEST_USED.json）を、評価が最後まで成功したときに書く。記録があれば、2回目は実行できない。
  - 予行演習（val）と最終評価（test）は、同じコードで動く（バグで test を無駄にしないため）。
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import system_eval as se  # noqa: E402
from ai_ocr.data import EvalItems  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT_FINAL, OUT_DEV = ROOT / "outputs" / "final", ROOT / "outputs" / "final_dev"
LOCK = OUT_FINAL / "TEST_USED.json"
SEEDS = [0, 1, 2]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--confirm-final", action="store_true", help="testでの最終評価（1回だけ）であることを理解した上で実行する")
    a = ap.parse_args()
    final = a.split == "test"
    if final:
        if LOCK.exists():
            info = json.loads(LOCK.read_text(encoding="utf-8"))
            sys.exit(f"✘ test は、すでに最終評価に使用済みです（{info['実行日時']}）。2回目の実行はできません。\n  結果: outputs/final/summary.md\n  （さらに評価したい場合は、新しいシードの test セット（例: testの1000番目以降）を作り、手順を新たに決めてください）")
        if not a.confirm_final:
            sys.exit("✘ test は、プロジェクト全体で1回だけ使います。実行するなら --confirm-final を付けてください。まず予行演習（--split val）で動作を確認しましょう。")
    out = OUT_FINAL if final else OUT_DEV
    out.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    det, ocrs = se.load_models(device, 0, SEEDS)
    taus = se.thresholds(SEEDS)                                           # 較正セットで決定（評価データは使わない）
    items = EvalItems(a.split, 0 if final else 100, 1000 if final else 500).items
    print(f"評価データ: {a.split} {len(items)}枚。しきい値: " + "、".join(f"s{k}={v:.4f}" for k, v in taus.items()), flush=True)
    df = se.evaluate_system(items, det, ocrs, taus, device)
    T = se.tables(df, SEEDS)
    df.to_csv(out / f"{a.split}_samples.csv", index=False)
    for k, v in T.items():
        v.to_csv(out / f"table_{k}.csv")
    label = "最終評価（test。1,000枚/難易度）" if final else "予行演習（val の101枚目以降。400枚/難易度）"
    lines = [f"# {label}", "", f"自動処理のしきい値（較正セットで決定。目標の誤り率 {se.TARGET_RISK:.0%}・昼間中心）: " + "、".join(f"CRNNシード{s}: {t:.4f}" for s, t in taus.items()), "",
             "## 道のり（完全一致。P3は3シード平均）", "", se.md(T["path"]), "", "## 検出", "", se.md(T["det"]), "", "## フィールド別の正解率（P3）", "", se.md(T["fields"]), "",
             "## 振り分け（P4。難易度の混ぜ方別・3シード平均±標準偏差）", "", se.md(T["decision"]), "", "## 振り分け（難易度別）", "", se.md(T["decision_by_diff"]), ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    if final:                                                             # 評価が最後まで成功したので、使用済みの記録を書く
        try:
            commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        except Exception:
            commit = ""
        LOCK.write_text(json.dumps({
            "実行日時": datetime.now().isoformat(timespec="seconds"), "評価データ": "test（難易度ごとに1,000枚）", "gitコミット": commit,
            "しきい値": {str(k): v for k, v in taus.items()}, "目標の誤り率": se.TARGET_RISK,
            "モデルのハッシュ": {**{"detector_resnet18_s0": sha(ROOT / "outputs/detect/resnet18_all_s0_best.pt")}, **{f"crnn_s{s}": sha(ROOT / f"outputs/ocr/crnn_all_cs3_s{s}_best.pt") for s in SEEDS}},
            "注意": "このファイルがある間、final_eval.py は test を評価しない。結果を見てモデルやしきい値を調整してはならない。"}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n✔ 最終評価を完了し、使用済みの記録を書きました: {LOCK}")


if __name__ == "__main__":
    main()
