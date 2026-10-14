"""古典的な文字認識: 板をまっすぐに直し、決まった位置の「文字セル」を、テンプレート（見本の文字）と照らし合わせる。

機械学習は使わない。ただし「板の書式（文字がどこに並ぶか）」を、人間が手で書いて持っておく必要がある。
手順（Notebook 01 で1段階ずつ見られる）:
  ① 板をまっすぐに直す（透視変換）→ 384×192
  ② 濃淡を「インクの濃さ」の画像にする（背景=0、文字=1）
  ③ 標準配置の各セルで、見本の文字を少しずらしながら重ね、いちばん似ている文字を選ぶ（正規化相互相関）
  ④ 一連番号は「4セル（・を含む）」と「5セル（ハイフン）」の2通りを試し、似ているほうを採用
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw

from plate_synth.fonts import get_font
from plate_synth.plate import nominal_cells
from plate_synth.scene import perspective_coeffs
from plate_synth.spec import CLASS_LETTERS, DOT, HYPHEN, KANA, PLATE_H, PLATE_W, REGIONS

DIGITS = list("0123456789")
REGION_CHARS = sorted({c for r in REGIONS for c in r})
SHIFT = 3            # 見本を重ねるときに試すずらし幅（半分の解像度で ±3px）
SCALE = 0.5          # 照合は半分の解像度で行う（速度のため）


@dataclass
class OcrResult:
    text: str
    fields: dict                       # region / class_no / kana / serial
    char_scores: list                  # 文字ごとの類似度（-1〜1。1に近いほど見本とよく一致）
    serial_hyphen: bool
    info: dict = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """板全体の確信度の目安（文字ごとの類似度の最小値）"""
        return float(min(self.char_scores)) if self.char_scores else 0.0


def rectify(img: Image.Image, corners) -> Image.Image:
    """① 四隅 corners（左上・右上・右下・左下）を、384×192 の長方形に直す"""
    dst = [(0, 0), (PLATE_W, 0), (PLATE_W, PLATE_H), (0, PLATE_H)]
    coeffs = perspective_coeffs(dst, [tuple(map(float, p)) for p in corners])
    return img.transform((PLATE_W, PLATE_H), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)


def ink_map(plate: Image.Image) -> np.ndarray:
    """② 「インクの濃さ」の画像（0=背景、1=文字）。明るさの違いに左右されないよう、パーセンタイルで正規化する"""
    g = np.asarray(plate.convert("L"), dtype=np.float32)
    inner = g[12:180, 23:363]
    lo, hi = np.percentile(inner, 3), np.percentile(inner, 90)
    ink = np.clip((hi - g) / (hi - lo + 1e-3), 0, 1)
    ink[:12], ink[180:], ink[:, :23], ink[:, 363:] = 0, 0, 0, 0     # 縁取り・ボルト穴を無視
    im = Image.fromarray((ink * 255).astype(np.uint8)).resize((int(PLATE_W * SCALE), int(PLATE_H * SCALE)), Image.Resampling.BOX)
    return np.asarray(im, dtype=np.float32) / 255.0


@lru_cache(maxsize=4096)
def _template(ch: str, size: int, w: int, h: int) -> np.ndarray:
    """見本の文字（セルの中央に描いた、インクの濃さ画像）。半分の解像度"""
    im = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im).text((w / 2, h / 2), ch, font=get_font(size), fill=255, anchor="mm")
    im = im.resize((max(1, int(w * SCALE)), max(1, int(h * SCALE))), Image.Resampling.BOX)
    return np.asarray(im, dtype=np.float32) / 255.0


def _candidates(cell: dict) -> list[str]:
    f = cell["field"]
    if f == "region":
        return REGION_CHARS
    if f == "class_no":
        return list("123456789") if cell["idx"] == 0 else (DIGITS if cell["idx"] == 1 else DIGITS + CLASS_LETTERS)
    if f == "kana":
        return KANA
    return [HYPHEN] if cell.get("slot") == "hyphen" else DIGITS + [DOT]


def _norm_rows(m: np.ndarray) -> np.ndarray:
    m = m - m.mean(1, keepdims=True)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-6)


def cell_scores(ink: np.ndarray, cell: dict) -> tuple[list[str], np.ndarray]:
    """③ 1つのセルで、見本の文字を ±SHIFT ずらしながら重ね、文字ごとの最高の類似度を返す（-1〜1）"""
    size, adv = cell["size"], cell["adv"]
    w, h = int(adv + 2 * 6), int(size * 1.25)               # セルの窓（少し余裕を持たせる）
    cx, cy = cell["cx"] * SCALE, cell["cy"] * SCALE
    chars = _candidates(cell)
    T = np.stack([_template(c, size, w, h).ravel() for c in chars])
    tw, th = _template(chars[0], size, w, h).shape[::-1]
    x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
    pad = SHIFT + 20                                          # 窓が板の外にはみ出しても、0で埋めて比較できるように
    padded = np.pad(ink, pad)
    patches = []
    for dy in range(-SHIFT, SHIFT + 1):
        for dx in range(-SHIFT, SHIFT + 1):
            patches.append(padded[y0 + dy + pad: y0 + dy + pad + th, x0 + dx + pad: x0 + dx + pad + tw].ravel())
    P = _norm_rows(np.stack(patches))
    S = P @ _norm_rows(T).T                                  # (ずらし数, 文字数)
    return chars, S.max(0)


def _match_cell(ink: np.ndarray, cell: dict) -> tuple[str, float]:
    """最も似ている文字と、その類似度"""
    chars, best = cell_scores(ink, cell)
    k = int(best.argmax())
    return chars[k], float(best[k])


def read_text(plate: Image.Image) -> OcrResult:
    """まっすぐにした板（384×192）から、文字列を読む"""
    ink = ink_map(plate)
    best = None
    for hyphen in (False, True):                             # ④ 4セルと5セルの両方を試す
        cells = nominal_cells(hyphen)
        res = [_match_cell(ink, c) for c in cells]
        mean = float(np.mean([s for _, s in res]))
        if best is None or mean > best[0]:
            best = (mean, hyphen, cells, res)
    _, hyphen, cells, res = best
    fields = {"region": "", "class_no": "", "kana": "", "serial": ""}
    for c, (ch, _) in zip(cells, res):
        fields[c["field"]] += ch
    text = f"{fields['region']} {fields['class_no']} {fields['kana']} {fields['serial']}"
    return OcrResult(text=text, fields=fields, char_scores=[s for _, s in res], serial_hyphen=hyphen)
