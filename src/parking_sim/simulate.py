"""シミュレーションの実行。入出庫の事象（AIが読んだ結果つき）を時刻順に処理し、指標を測る。

入庫: AIが自動処理と判断したら読み取り結果をそのまま記録（読み間違いも、そのまま記録される）。
      要確認・読み取り不可なら、人が正しいナンバーを入力する。
出庫: policy.py の方針で照合・精算する。
"""
import heapq
from dataclasses import dataclass

import pandas as pd

from rules.decision import AUTO

from .db import ParkingDB
from .fees import fee_yen
from .policy import A_AUTO, A_MANUAL, A_REVIEW, ExitPolicy, process_exit


@dataclass
class Event:
    vid: int
    kind: str                # "in" / "out"
    t: float
    gt: str                  # 正しいナンバー（空白なし）
    text: str                # AIの読み取り結果
    valid: bool
    conf: float
    decision: str
    difficulty: str


def simulate(events: list[Event], policy: ExitPolicy, cost_error: float = 20.0, detail: bool = False):
    """events: 入庫・出庫の事象（AIの読み取り結果つき）。戻り値: 指標（割合は、出庫した車の台数に対する割合）。
    detail=True なら、(指標, 出庫1件ごとの結果の表) を返す（人に回した理由 note などを分析できる）"""
    db = ParkingDB()
    heap = [(e.t, 0 if e.kind == "out" else 1, i, e) for i, e in enumerate(events)]        # 同時刻なら出庫を先に
    heapq.heapify(heap)
    rows, entry_stats, max_active = [], dict(n=0, auto=0, auto_wrong=0, human=0), 0
    while heap:
        _, _, _, e = heapq.heappop(heap)
        if e.kind == "in":
            entry_stats["n"] += 1
            if e.decision == AUTO:
                entry_stats["auto"] += 1; entry_stats["auto_wrong"] += e.text != e.gt
                db.enter(e.text, e.conf, e.decision, e.t, e.vid)             # 読み間違いも、そのまま記録される
            else:
                entry_stats["human"] += 1
                db.enter(e.gt, 1.0, "人が入力", e.t, e.vid)
            max_active = max(max_active, db.n_active())
        else:
            o = process_exit(db, e.text, e.valid, e.decision, e.conf, e.vid, e.t, policy)
            rows.append(dict(vid=e.vid, difficulty=e.difficulty, action=o.action, correct=o.correct, fee_error=abs(o.fee_billed - o.fee_true), note=o.note))
    db.close()
    df = pd.DataFrame(rows)
    n = len(df)
    auto, auto_wrong = df[df.action == A_AUTO], df[(df.action == A_AUTO) & ~df.correct]
    res = dict(
        台数=n, 自動精算率=(len(auto) - len(auto_wrong)) / n, 誤課金率=len(auto_wrong) / n,
        要確認率=(df.action == A_REVIEW).mean(), 読み取り不可_未照合率=(df.action == A_MANUAL).mean(),
        人手対応率=(df.action != A_AUTO).mean(), 誤課金時の平均料金誤差=float(auto_wrong.fee_error.mean()) if len(auto_wrong) else 0.0,
        入庫の自動処理率=entry_stats["auto"] / entry_stats["n"], 入庫の読み間違い記録率=entry_stats["auto_wrong"] / entry_stats["n"],
        最大同時駐車台数=max_active,
    )
    res["期待コスト"] = res["人手対応率"] * 1.0 + res["誤課金率"] * cost_error          # 要確認1件=1、誤課金1件=cost_error
    return (res, df) if detail else res
