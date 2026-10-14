"""古典的な板の検出: 「明るくて色が薄い」領域を探し、形（縦横比・埋まり具合）と文字の色で絞る。

機械学習は使わない。手順（Notebook 01 で1段階ずつ見られる）:
  ① 明るく・彩度の低い画素を取り出す（白い板）
  ② すき間（文字の部分）を埋める（クロージング、穴埋め）
  ③ つながった領域（連結成分）を数える
  ④ 領域ごとに「回転した外接矩形」「縦横比」「埋まり率」「インクの緑さ」を調べ、板らしさで採点
"""
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

# しきい値などは、検証セットを見ながら手で決めたもの。古典的手法は、こうした手調整に頼る
PARAMS = dict(
    v_ratio=0.70,        # 「明るい」= 画像の最も明るい部分(99.5%点)の何割以上か
    s_max=0.25,          # 「色が薄い」= 彩度の上限
    smooth=1.0,          # ノイズ対策のぼかし（半径）
    close_size=7,        # クロージングの大きさ（奇数）
    open_size=15,        # オープニングの大きさ（0=なし）。細い橋でつながった別の物を切り離す
    area_min=0.004, area_max=0.40,   # 領域の面積（画像に対する割合）
    aspect=(1.3, 3.2),   # 縦横比（長辺÷短辺）
    fill_min=0.60,       # 外接矩形に対する埋まり率
    ink_ratio=(0.04, 0.60),          # 板の中の「穴」（文字）の割合
    green_min=0.05,      # インクが緑がかっている度合い（背景の白い看板との区別）
)


@dataclass
class Detection:
    corners: np.ndarray          # 四隅 左上・右上・右下・左下 (4,2)
    bbox: list                   # [x, y, 幅, 高さ]
    score: float
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------ 連結成分
def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """2値画像の連結成分ラベリング（4近傍）。行ごとの「連なり(run)」を併合する方式。戻り値: (ラベル画像, 個数)"""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    prev = []                                              # 直前の行の (開始, 終了, ラベル)
    for y in range(h):
        row = np.concatenate(([0], mask[y].astype(np.int8), [0]))
        d = np.diff(row)
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        cur = []
        for s, e in zip(starts, ends):
            near = [lab for ps, pe, lab in prev if s < pe and ps < e]
            if not near:
                parent.append(len(parent))
                lab = len(parent) - 1
            else:
                lab = min(find(n) for n in near)
                for n in near:                              # 併合
                    rn = find(n)
                    if rn != lab:
                        parent[rn] = lab
            labels[y, s:e] = lab
            cur.append((s, e, lab))
        prev = cur
    lut = np.array([find(i) for i in range(len(parent))], dtype=np.int32)
    labels = lut[labels]
    uniq = np.unique(labels[labels > 0])
    remap = np.zeros(lut.max() + 1, dtype=np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    return remap[labels], len(uniq)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """領域の内側の穴を埋める（画像の端につながっていない背景を、前景にする）"""
    bg, n = label_components(~mask)
    border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]])).tolist())
    filled = mask.copy()
    for lab in range(1, n + 1):
        if lab not in border:
            filled |= bg == lab
    return filled


def opening(mask: np.ndarray, size: int) -> np.ndarray:
    """オープニング（縮めてから膨らませる）: 細い橋や小さな点を取り除く"""
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MinFilter(size)).filter(ImageFilter.MaxFilter(size))
    return np.asarray(im) > 127


def close(mask: np.ndarray, size: int) -> np.ndarray:
    """クロージング（膨らませてから縮める）: 小さなすき間を埋める"""
    im = Image.fromarray((mask * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size))
    return np.asarray(im) > 127


# ------------------------------------------------------------------ 検出
def bright_low_saturation_mask(img: Image.Image, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """① 明るく、彩度の低い画素。戻り値: (マスク, ぼかした画像の配列)"""
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(p["smooth"])), dtype=np.float32)
    v = a.max(2)
    s = (v - a.min(2)) / (v + 1e-3)
    thr = p["v_ratio"] * np.percentile(v, 99.5)
    return (v > thr) & (s < p["s_max"]), a


def oriented_rect(xy: np.ndarray):
    """点の集まり(N,2)から、主成分方向に沿った外接矩形（回転した長方形）を求める。戻り値: (四隅, 長辺, 短辺)"""
    m = xy.mean(0)
    cov = np.cov((xy - m).T)
    vals, vecs = np.linalg.eigh(cov)
    u = vecs[:, 1]                                          # 長軸の向き
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    if v[1] < 0:
        v = -v
    pu, pv = (xy - m) @ u, (xy - m) @ v
    a0, a1, b0, b1 = pu.min(), pu.max(), pv.min(), pv.max()
    corners = np.array([m + a0 * u + b0 * v, m + a1 * u + b0 * v, m + a1 * u + b1 * v, m + a0 * u + b1 * v])
    return corners, a1 - a0 + 1, b1 - b0 + 1


def candidates(img: Image.Image, p: dict) -> tuple[list[dict], dict]:
    """板の候補をすべて採点して返す（デバッグ・可視化用に、途中経過も返す）"""
    mask, a = bright_low_saturation_mask(img, p)
    closed = close(mask, p["close_size"])
    filled = fill_holes(closed)
    if p["open_size"]:
        filled = opening(filled, p["open_size"])
    labels, n = label_components(filled)
    H, W = mask.shape
    out = []
    for lab in range(1, n + 1):
        comp = labels == lab
        area = int(comp.sum())
        if area < p["area_min"] * H * W or area > p["area_max"] * H * W:
            continue
        ys, xs = np.nonzero(comp)
        corners, long_, short_ = oriented_rect(np.stack([xs, ys], 1).astype(np.float64))
        aspect = long_ / max(short_, 1e-6)
        fill = area / (long_ * short_)
        holes = comp & ~mask                                  # 板の中で「明るくない」画素 = 文字（インク）
        ink = float(holes.sum() / area)
        if holes.sum() > 0:
            col = a[holes].mean(0)
            green = float((col[1] - col[0]) / (a[comp].max(1).mean() + 1e-3))   # 緑がかり
        else:
            green = 0.0
        ok = (p["aspect"][0] <= aspect <= p["aspect"][1] and fill >= p["fill_min"]
              and p["ink_ratio"][0] <= ink <= p["ink_ratio"][1] and green >= p["green_min"])
        score = fill * min(1.0, green / 0.15) * (1 - abs(aspect - 2.0) / 2.0) if ok else 0.0
        out.append(dict(corners=corners, area=area, aspect=aspect, fill=fill, ink=ink, green=green, ok=ok, score=score))
    return out, dict(mask=mask, closed=closed, filled=filled, labels=labels)


def detect_plate(img: Image.Image, params: dict | None = None) -> Detection | None:
    """板を1つ見つける。見つからなければ None（古典的手法は、条件が悪いとここで失敗する）"""
    p = {**PARAMS, **(params or {})}
    cands, _ = candidates(img, p)
    good = [c for c in cands if c["ok"]]
    if not good:
        return None
    best = max(good, key=lambda c: c["score"])
    xy = best["corners"]
    lo, hi = xy.min(0), xy.max(0)
    return Detection(corners=xy, bbox=[float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])],
                     score=float(best["score"]), info={k: best[k] for k in ("area", "aspect", "fill", "ink", "green")})
