"""VLMの返事（文字列）から、読み取り結果と自信を取り出し、表記ゆれをそろえる。"""
import json
import re

_DASH = re.compile(r"[ー−‐‑–—―－−]")             # ハイフンの表記ゆれ
_DOT = re.compile(r"[･·•●◦∙.．]")                       # 「・」の表記ゆれ
_FULL = str.maketrans("０１２３４５６７８９ＡＣＦＨＫＬＭＰＸＹ", "0123456789ACFHKLMPXY")


def normalize(text: str) -> str:
    """空白を除き、全角英数を半角に、ハイフン・中点の表記ゆれをそろえる（読み取りの内容は変えない）"""
    t = (text or "").translate(_FULL)
    t = _DASH.sub("-", t)
    t = _DOT.sub("・", t)
    return re.sub(r"\s+", "", t)


def parse_reply(raw: str) -> dict:
    """返事から {"text", "confidence", "parsed"} を作る。JSONが読めなければ parsed=False（文字列だけ拾う）"""
    raw = raw or ""
    m = re.search(r"\{.*?\}", raw, flags=re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            conf = d.get("confidence")
            conf = float(conf) if conf is not None and str(conf).replace(".", "", 1).isdigit() else None
            return dict(text=normalize(str(d.get("text", ""))), confidence=conf, parsed=True)
        except (json.JSONDecodeError, ValueError):
            pass
    first = raw.strip().splitlines()[0] if raw.strip() else ""
    return dict(text=normalize(first), confidence=None, parsed=False)
