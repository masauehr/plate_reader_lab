"""ナンバー風プレート画像の生成（ラベルの生成と描画）。"""
import numpy as np
from PIL import Image, ImageDraw

from .fonts import get_font
from .spec import (CLASS_LETTERS, DOT, HYPHEN, KANA, PLATE_H, PLATE_TYPES, PLATE_W, REGIONS)


def format_serial(n: int) -> str:
    """一連番号（1〜9999）を表示形式にする。4桁は中央にハイフン（例: 12-34）、3桁以下は上位を「・」で埋める（例: ・・12）。簡略化"""
    if n >= 1000:
        s = f"{n:04d}"
        return f"{s[:2]}{HYPHEN}{s[2:]}"
    return str(n).rjust(4, DOT)


def random_label(rng: np.random.Generator, plate_type: str = "private") -> dict:
    """ランダムなプレートの内容（実在の番号を狙わない）"""
    region = REGIONS[rng.integers(len(REGIONS))]
    # 分類番号: 3桁の数字（70%）か、末尾が英字（30%）
    d = [str(rng.integers(1, 10)), str(rng.integers(0, 10))]
    last = CLASS_LETTERS[rng.integers(len(CLASS_LETTERS))] if rng.random() < 0.3 else str(rng.integers(0, 10))
    class_no = "".join(d) + last
    kana = KANA[rng.integers(len(KANA))]
    serial = format_serial(int(rng.integers(1, 10000)))
    return dict(region=region, class_no=class_no, kana=kana, serial=serial, plate_type=plate_type,
                text=f"{region} {class_no} {kana} {serial}")


def _draw_chars(draw: ImageDraw.ImageDraw, text: str, x: float, cy: float, size: int, advance: float, fill):
    """1文字ずつ、決まった幅(advance)の中央に描く（文字間隔を安定させるため）"""
    font = get_font(size)
    for ch in text:
        draw.text((x + advance / 2, cy), ch, font=font, fill=fill, anchor="mm")
        x += advance
    return x


def render_plate(label: dict, rng: np.random.Generator | None = None) -> Image.Image:
    """プレート画像（RGBA, 384×192）を描く。rngを渡すと、位置・大きさ・色に小さなばらつきを加える"""
    t = PLATE_TYPES[label["plate_type"]]
    bg, fg = t["bg"], t["fg"]
    j = (lambda a: int(rng.integers(-a, a + 1))) if rng is not None else (lambda a: 0)

    im = Image.new("RGBA", (PLATE_W, PLATE_H), bg + (255,))
    d = ImageDraw.Draw(im)
    # 縁取り（角を丸めた枠）
    d.rounded_rectangle((7, 7, PLATE_W - 8, PLATE_H - 8), radius=14, outline=fg, width=3)
    # ボルト穴（左右。簡略化）
    for x in (16, PLATE_W - 17):
        d.ellipse((x - 6, PLATE_H // 2 - 6, x + 6, PLATE_H // 2 + 6), fill=tuple(int(c * 0.6 + 100 * 0.4) for c in fg))

    # 上段: 地名（左）・分類番号（右）
    top_cy = 56 + j(2)
    s_region = 46 + j(2)
    _draw_chars(d, label["region"], 30 + j(3), top_cy, s_region, s_region * 0.98, fg)
    s_class = 60 + j(2)
    adv_c = s_class * 0.62
    _draw_chars(d, label["class_no"], PLATE_W - 30 - adv_c * 3 + j(3), top_cy + 2, s_class, adv_c, fg)

    # 下段: かな（左）・一連番号（右、大きく）
    bot_cy = 138 + j(2)
    s_kana = 62 + j(2)
    _draw_chars(d, label["kana"], 28 + j(3), bot_cy, s_kana, s_kana * 1.0, fg)
    s_ser = 104 + j(3)
    adv_d, adv_h = s_ser * 0.56, s_ser * 0.36           # 数字・「・」の幅、ハイフンの幅
    ser = label["serial"]
    total = sum(adv_h if ch == HYPHEN else adv_d for ch in ser)
    x = PLATE_W - 24 - total + j(3)
    for ch in ser:
        adv = adv_h if ch == HYPHEN else adv_d
        _draw_chars(d, ch, x, bot_cy + 2, s_ser, adv, fg)
        x += adv
    return im
