"""シーン画像の合成: 背景を作り、プレートを傾け・歪めて貼り付ける。"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .spec import PLATE_H, PLATE_W, SCENE_H, SCENE_W


# ------------------------------------------------------------------ 背景
def _noise_texture(rng, strength: float) -> np.ndarray:
    n = rng.normal(0, strength, (SCENE_H, SCENE_W, 1)).astype(np.float32)
    # 少しぼかして「粒」を大きくする
    im = Image.fromarray(np.clip(n[..., 0] * 4 + 128, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))
    return (np.asarray(im, dtype=np.float32)[..., None] - 128) / 4


def make_background(rng: np.random.Generator, bg_type: str) -> Image.Image:
    """単純な背景（実写真は使わない）。wall=壁、gradient=グラデーション、car=車体に見立てた矩形"""
    base = rng.integers(60, 200, 3).astype(np.float32)
    canvas = np.tile(base, (SCENE_H, SCENE_W, 1))
    if bg_type == "gradient":
        other = rng.integers(40, 220, 3).astype(np.float32)
        t = np.linspace(0, 1, SCENE_H, dtype=np.float32)[:, None, None]
        canvas = base * (1 - t) + other * t
        canvas = np.broadcast_to(canvas, (SCENE_H, SCENE_W, 3)).copy()
    canvas += _noise_texture(rng, 3.0)
    im = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    if bg_type == "car":
        d = ImageDraw.Draw(im)
        asphalt = int(rng.integers(70, 110))
        d.rectangle((0, int(SCENE_H * 0.82), SCENE_W, SCENE_H), fill=(asphalt, asphalt, asphalt))
        body = tuple(int(c) for c in rng.choice([20, 60, 120, 180, 230], 3) * 0.5 + rng.integers(0, 80, 3))
        x0, x1 = int(rng.integers(5, 30)), SCENE_W - int(rng.integers(5, 30))
        d.rounded_rectangle((x0, int(SCENE_H * 0.22), x1, int(SCENE_H * 0.85)), radius=18, fill=body)
        hl = tuple(min(255, c + 40) for c in body)
        d.rectangle((x0 + 8, int(SCENE_H * 0.30), x1 - 8, int(SCENE_H * 0.34)), fill=hl)            # 反射の帯
        d.rectangle((x0 + 4, int(SCENE_H * 0.76), x1 - 4, int(SCENE_H * 0.85)), fill=tuple(c // 2 for c in body))  # バンパー
        for cx in (x0 + 40, x1 - 40):                                                                   # ライト
            d.ellipse((cx - 22, int(SCENE_H * 0.36), cx + 22, int(SCENE_H * 0.36) + 30), fill=(235, 235, 210))
        im = im.filter(ImageFilter.GaussianBlur(0.8))
    return im


def add_distractors(im: Image.Image, rng: np.random.Generator, n: int) -> None:
    """プレートに紛らわしい物（文字のような模様のある白い板）を描く。検出の難しさを上げる"""
    d = ImageDraw.Draw(im)
    for _ in range(n):
        w = int(rng.integers(40, 110)); h = int(w / rng.uniform(1.3, 3.0))
        x = int(rng.integers(0, SCENE_W - w)); y = int(rng.integers(0, SCENE_H - h))
        tone = int(rng.integers(200, 250))
        d.rectangle((x, y, x + w, y + h), fill=(tone, tone, tone - 5), outline=(60, 60, 60))
        for k in range(int(rng.integers(2, 6))):                      # 文字のような縦棒
            bx = x + 5 + k * (w - 10) // 6
            d.rectangle((bx, y + h // 4, bx + int(rng.integers(2, 6)), y + 3 * h // 4), fill=(40, 40, 40))


# ------------------------------------------------------------------ 透視変換
def perspective_coeffs(dst_pts, src_pts) -> list[float]:
    """PILのPERSPECTIVE変換の係数。出力座標(dst)から入力座標(src)への写像を、4点の対応から解く"""
    A, B = [], []
    for (X, Y), (x, y) in zip(dst_pts, src_pts):
        A.append([X, Y, 1, 0, 0, 0, -x * X, -x * Y]); B.append(x)
        A.append([0, 0, 0, X, Y, 1, -y * X, -y * Y]); B.append(y)
    return list(np.linalg.solve(np.array(A, dtype=np.float64), np.array(B, dtype=np.float64)))


def _is_convex(pts) -> bool:
    s = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        s.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    return all(v > 0 for v in s) or all(v < 0 for v in s)


def choose_corners(rng: np.random.Generator, scale: float, rot_deg: float, persp: float) -> np.ndarray:
    """板の四隅（左上・右上・右下・左下）の位置を決める。画面からはみ出さないよう、何度か試す"""
    w = scale * SCENE_W
    h = w * PLATE_H / PLATE_W * rng.uniform(0.97, 1.03)
    rect = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    th = np.deg2rad(rot_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for _ in range(60):
        jitter = np.clip(rng.uniform(-persp, persp, (4, 2)) * w, -0.15 * w, 0.15 * w)
        quad = rect @ R.T + jitter
        if not _is_convex(quad):
            continue
        lo, hi = quad.min(0), quad.max(0)
        cx_lo, cx_hi = 2 - lo[0], SCENE_W - 2 - hi[0]
        cy_lo, cy_hi = 2 - lo[1], SCENE_H - 2 - hi[1]
        if cx_lo >= cx_hi or cy_lo >= cy_hi:
            continue
        # 中央寄りに置く（車体の中央付近に板があるのが普通）。範囲内で一様に抽出
        c = np.array([rng.uniform(cx_lo, cx_hi), rng.uniform(cy_lo, cy_hi)])
        return quad + c
    # 見つからなければ、小さくして中央に置く
    return choose_corners(rng, scale * 0.8, 0.0, 0.0)


def paste_plate(scene: Image.Image, plate: Image.Image, corners: np.ndarray) -> Image.Image:
    """プレート画像を四隅 corners に合わせて変形し、シーンに貼り付ける"""
    src = [(0, 0), (PLATE_W, 0), (PLATE_W, PLATE_H), (0, PLATE_H)]
    coeffs = perspective_coeffs([tuple(p) for p in corners], src)
    warped = plate.transform((SCENE_W, SCENE_H), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)
    out = scene.convert("RGBA")
    out.alpha_composite(warped)
    return out.convert("RGB")


def bbox_of(corners: np.ndarray) -> list[float]:
    """四隅の外接矩形 [x, y, 幅, 高さ]"""
    lo, hi = corners.min(0), corners.max(0)
    return [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]
