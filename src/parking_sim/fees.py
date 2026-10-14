"""料金表（簡単なもの）: 最初の30分は無料、以降は30分ごとに100円、1日（24時間）あたり上限1,500円。"""
import math

FREE_MIN, UNIT_MIN, UNIT_YEN, DAY_CAP_YEN = 30, 30, 100, 1500


def fee_yen(minutes: float) -> int:
    """駐車時間（分）から料金（円）"""
    if minutes <= FREE_MIN:
        return 0
    full_days, rest = divmod(minutes, 24 * 60)
    rest_fee = min(DAY_CAP_YEN, max(0, math.ceil((rest - FREE_MIN) / UNIT_MIN)) * UNIT_YEN) if rest > FREE_MIN else 0
    return int(full_days * DAY_CAP_YEN + rest_fee)
