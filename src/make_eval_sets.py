"""固定シードの評価セットを data/eval/ に生成する。

使い方:
    .venv/bin/python src/make_eval_sets.py                 # val 500枚 / test 1000枚 × 3難易度（約数分）
    .venv/bin/python src/make_eval_sets.py --val 50 --test 100   # 少量（動作確認用）
出力:
    data/eval/{val,test}/{easy,medium,hard}/images/*.png と labels.jsonl
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_synth import DIFFICULTIES, make_eval_set  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=500, help="valのクラス（難易度）あたり枚数")
    ap.add_argument("--test", type=int, default=1000, help="testの難易度あたり枚数")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "eval")
    a = ap.parse_args()
    t0 = time.time()
    for split, n in (("val", a.val), ("test", a.test)):
        for d in DIFFICULTIES:
            out = make_eval_set(a.out, split, d, n)
            print(f"{split:4s} {d:6s} {n}枚 → {out}  ({time.time() - t0:.0f}秒経過)")


if __name__ == "__main__":
    main()
