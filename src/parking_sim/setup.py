"""シミュレーションの準備（しきい値の決定・事象の作成）。run_phase5.py とNotebookの両方から使う。"""
from pathlib import Path

import pandas as pd

from rules.decision import choose_threshold, decide

from .simulate import Event
from .world import MIX_DAY

ROOT = Path(__file__).resolve().parent.parent.parent


def auto_threshold(target_risk: float) -> float:
    """Phase4の方法（較正セット・昼間中心・書式検査のみ＋seq確信度）で、自動処理のしきい値を決める"""
    cal = pd.read_csv(ROOT / "outputs" / "rules" / "seed0_cal_samples.csv")
    n = cal.difficulty.value_counts()
    w = cal.difficulty.map(lambda d: MIX_DAY[d] / n[d]).to_numpy()
    return choose_threshold(cal.raw_seq, cal.exact_raw, cal.raw_valid, target_risk, w)


def to_events(tab: pd.DataFrame, tau: float) -> list[Event]:
    """事象の表（AIの読み取り結果つき）を、シミュレーション用の Event に。判定はしきい値 tau で決める"""
    return [Event(int(r.vid), r.kind, float(r.t), r.gt, r.text if isinstance(r.text, str) else "", bool(r.valid), float(r.conf),
                  decide(bool(r.valid), float(r.conf), tau), r.difficulty) for r in tab.itertuples()]
