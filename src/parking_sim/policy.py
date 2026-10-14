"""出庫の照合方針。AIの読み取り結果（文字列・書式の正しさ・確信度・Phase4の判定）と、入庫の記録を突き合わせ、
「自動精算」か「人の確認」かを決める。

  自動精算   … 人手なしで、入庫の記録を見つけて料金を計算する
  要確認     … 候補（近い記録）を添えて、人に確認してもらう
  読み取り不可・未照合 … 候補がない。人が手で探す
人が確認したときは、正しい記録に結びつく（人は車や画像を見て選べる想定。誤りは持ち込まない）。
"""
from dataclasses import dataclass

from rules.decision import AUTO, REJECT

from .fees import fee_yen

A_AUTO, A_REVIEW, A_MANUAL = "自動精算", "要確認", "読み取り不可・未照合"


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    auto_if_valid: bool = False          # True: 確信度が低くても、書式が正しく入庫記録に一意に一致すれば自動精算（照合を信頼する）
    neighbor_check: bool = False         # True: 一致した記録に「1文字違いの別の入庫中の記録」があれば、要確認に回す（取り違え対策）
    near_k: int = 1                      # 要確認の候補にする、編集距離


POLICIES = [
    ExitPolicy("P1 保守（AIが自動処理と判断＋一意に一致）"),
    ExitPolicy("P2 照合を信頼（書式OKで一意に一致なら自動）", auto_if_valid=True),
    ExitPolicy("P3 保守＋近傍チェック", neighbor_check=True),
    ExitPolicy("P4 照合を信頼＋近傍チェック", auto_if_valid=True, neighbor_check=True),
]


@dataclass
class ExitOutcome:
    action: str                 # 自動精算／要確認／読み取り不可・未照合
    billed_vehicle: int         # 実際に課金した記録の車（評価用）
    correct: bool               # 正しい車の記録に課金したか
    fee_billed: int
    fee_true: int
    note: str = ""


def process_exit(db, text: str, valid: bool, decision: str, conf: float, truth_vid: int, t_out: float, policy: ExitPolicy) -> ExitOutcome:
    """1台の出庫を処理して、記録を閉じる。truth_vid は評価用（人が確認する場面でだけ、正しい記録の選択に使う）"""
    true_rec = db.by_truth(truth_vid)                          # 評価用（正しい入庫記録）
    fee_true = fee_yen(t_out - true_rec[1]) if true_rec else 0

    def human(action: str, note: str) -> ExitOutcome:           # 人が確認して、正しい記録に結びつく
        if true_rec[3] == "in":
            db.exit(true_rec[0], t_out)
        else:                                                   # 先の誤課金で、この車の記録が誤って閉じられていた。人が復元して精算する
            note += "（先の誤課金で記録が閉じられていた）"; action = A_MANUAL
        return ExitOutcome(action, truth_vid, True, fee_true, fee_true, note)

    if decision == REJECT or not text:
        return human(A_MANUAL, "読み取り不可")
    exact = db.active_exact(text)
    if len(exact) == 1:
        rec = exact[0]
        trusted = decision == AUTO or (policy.auto_if_valid and valid)
        if trusted and policy.neighbor_check and db.active_near(text, 1, exclude_id=rec[0]):
            return human(A_REVIEW, "近い別の記録があるため確認")
        if trusted:
            fee = fee_yen(t_out - rec[1])
            db.exit(rec[0], t_out)
            return ExitOutcome(A_AUTO, rec[2], rec[2] == truth_vid, fee, fee_true, "自動精算")
        return human(A_REVIEW, "確信度が低いため確認")
    if len(exact) > 1:
        return human(A_REVIEW, "同じナンバーの記録が複数")
    near = db.active_near(text, policy.near_k)
    return human(A_REVIEW if near else A_MANUAL, "近い候補あり" if near else "候補なし（未照合）")
