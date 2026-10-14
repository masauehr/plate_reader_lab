"""候補を使った補正（ルールベース）。

CRNNが各位置で持っていた「次点の候補」から、書式に合う文字を選び直す。
直せるのは「文字は合っているが、書式に合わない文字を読んだ」場合だけ。文字数が違う（脱落・余分）ものは直せない。
"""
import math
from dataclasses import dataclass, field

from plate_synth.spec import DOT, HYPHEN, REGIONS

from .format import allowed_chars, is_valid, violations

FLOOR = 1e-4                                           # 候補に無い文字の確率の下限（対数計算のため）


@dataclass
class Correction:
    text: str                       # 補正後の文字列（直せなければ元のまま）
    status: str                     # "ok"（そのまま正しい）/ "corrected"（補正して正しくなった）/ "invalid"（直せない）
    changes: list = field(default_factory=list)    # [(位置, 元の文字, 新しい文字)]
    probs: list = field(default_factory=list)      # 補正後の文字ごとの確率（候補にあれば）
    reasons: list = field(default_factory=list)    # 書式の違反（invalidのとき）


def _alt_prob(em, ch: str) -> float:
    for c, p in em.alts:
        if c == ch:
            return p
    return FLOOR


def _best_allowed(em, allowed: str):
    """その位置の候補のうち、許される文字で最も確率の高いもの。なければ None"""
    for c, p in sorted(em.alts, key=lambda x: -x[1]):
        if c in allowed:
            return c, p
    return None


def correct(emissions: list, text: str) -> Correction:
    """emissions: 読み取った1文字ごとの Emission。text: 貪欲デコードの文字列"""
    n = len(emissions)
    probs = [e.prob for e in emissions]
    if is_valid(text):
        return Correction(text, "ok", [], probs)
    if n not in (10, 11):                                # 文字の脱落・余分は、候補の入れ替えでは直せない
        return Correction(text, "invalid", [], probs, violations(text))
    chars, changes = list(text), []

    def fix(i, allowed):
        if chars[i] in allowed:
            return True
        r = _best_allowed(emissions[i], allowed)
        if r is None:
            return False
        changes.append((i, chars[i], r[0])); chars[i] = r[0]; probs[i] = r[1]
        return True

    # 地名: 2文字が一覧にない → 一覧の名前のうち、候補の確率の積が最大のものを選ぶ
    if "".join(chars[:2]) not in REGIONS:
        best = max(REGIONS, key=lambda r: sum(math.log(_alt_prob(emissions[i], r[i])) for i in range(2)))
        for i in range(2):
            if chars[i] != best[i]:
                changes.append((i, chars[i], best[i])); chars[i] = best[i]; probs[i] = _alt_prob(emissions[i], best[i])
    ok = True
    for i, (fld, idx) in enumerate([("class_no", 0), ("class_no", 1), ("class_no", 2)], start=2):
        ok &= fix(i, allowed_chars(fld, idx, n))
    ok &= fix(5, allowed_chars("kana", 0, n))
    for j in range(6, n):                                # 一連番号
        ok &= fix(j, allowed_chars("serial", j - 6, n))
    if n == 10 and ok:                                   # 「・」は先頭だけ: 数字のあとに「・」が来たら、数字に直す
        seen = False
        for j in range(6, n):
            if chars[j] == DOT and seen:
                r = _best_allowed(emissions[j], "0123456789")
                if r:
                    changes.append((j, DOT, r[0])); chars[j] = r[0]; probs[j] = r[1]
            if chars[j] != DOT:
                seen = True
    new = "".join(chars)
    if ok and is_valid(new):
        return Correction(new, "corrected", changes, probs)
    return Correction(text, "invalid", [], [e.prob for e in emissions], violations(text))
