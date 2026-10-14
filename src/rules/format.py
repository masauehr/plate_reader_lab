"""板の書式ルール（DATA_SPEC.md。学習用に簡略化したもの）と、検査。

  地名(2文字・一覧にある) ＋ 分類番号(3文字: [1-9][0-9][数字または英字]) ＋ かな(1文字・一覧にある) ＋ 一連番号
  一連番号: 「ab-cd」（数字4桁の中央にハイフン）か、「・」で上位を埋めた4文字（例: ・・12、・123、・・・1）
"""
import re

from plate_synth.spec import CLASS_LETTERS, DOT, HYPHEN, KANA, REGIONS

DIGITS = "0123456789"
CLASS_CHARS = DIGITS + "".join(CLASS_LETTERS)
SERIAL_HYPHEN = re.compile(r"^\d\d-\d\d$")
SERIAL_DOTS = re.compile(rf"^{DOT}{{1,3}}\d{{1,3}}$")        # 「・」は先頭にだけ並び、あとは数字（全部で4文字）


def split(text: str) -> dict:
    """空白なしの文字列を、地名(2)・分類番号(3)・かな(1)・一連番号(残り)に分ける"""
    t = text.replace(" ", "")
    return dict(region=t[:2], class_no=t[2:5], kana=t[5:6], serial=t[6:])


def violations(text: str) -> list[str]:
    """書式の違反を、理由の一覧で返す（空なら正しい書式）"""
    t = text.replace(" ", "")
    if len(t) not in (10, 11):
        return [f"文字数が{len(t)}（10か11が正しい）"]
    f, out = split(t), []
    if f["region"] not in REGIONS:
        out.append(f"地名「{f['region']}」が一覧にない")
    c = f["class_no"]
    if not (c[0] in DIGITS[1:] and c[1] in DIGITS and c[2] in CLASS_CHARS):
        out.append(f"分類番号「{c}」の形が違う")
    if f["kana"] not in KANA:
        out.append(f"かな「{f['kana']}」が一覧にない")
    s = f["serial"]
    if len(t) == 11 and not SERIAL_HYPHEN.match(s):
        out.append(f"一連番号「{s}」が「ab-cd」の形でない")
    if len(t) == 10 and not SERIAL_DOTS.match(s):
        out.append(f"一連番号「{s}」が「・」＋数字(4文字)の形でない")
    return out


def is_valid(text: str) -> bool:
    return not violations(text)


def allowed_chars(field: str, idx: int, n_total: int) -> str:
    """位置ごとに許される文字。n_total は全体の文字数（10か11）。一連番号の形は全体の長さで決まる"""
    if field == "region":
        return "".join(sorted({r[idx] for r in REGIONS}))
    if field == "class_no":
        return DIGITS[1:] if idx == 0 else (DIGITS if idx == 1 else CLASS_CHARS)
    if field == "kana":
        return "".join(KANA)
    if n_total == 11:                                     # ab-cd
        return HYPHEN if idx == 2 else DIGITS
    return DIGITS + DOT                                   # ・・12 など（「・」は先頭だけ。形は別に検査）
