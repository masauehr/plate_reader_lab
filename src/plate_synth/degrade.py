"""画像の劣化（DATA_SPEC.md 5）。パラメータは difficulty.sample_params で決まる。

どの劣化も「PIL画像 → PIL画像」。単独でも使えるので、Notebookで1つずつ見られる。
"""
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .spec import SCENE_H, SCENE_W


def _arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), dtype=np.float32)


def _img(a: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def lighting(im, rng, brightness: float, contrast: float, grad_strength: float):
    """明るさ・コントラスト・逆光/影（一方向に暗くなるグラデーション）"""
    a = _arr(im)
    mean = a.mean()
    a = (a - mean) * contrast + mean
    a = a * brightness
    if grad_strength > 0:
        th = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W].astype(np.float32)
        ramp = (xx - SCENE_W / 2) * np.cos(th) + (yy - SCENE_H / 2) * np.sin(th)
        ramp = (ramp - ramp.min()) / (ramp.max() - ramp.min() + 1e-6)          # 0〜1
        a = a * (1 - grad_strength * ramp)[..., None]
    return _img(a)


def stains(im, rng, count: int, bbox):
    """汚れ（板の付近にできる、ぼけた茶色の斑点）"""
    if count <= 0:
        return im
    x, y, w, h = bbox
    layer = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(layer)
    for _ in range(count):
        cx, cy = rng.uniform(x, x + w), rng.uniform(y, y + h)
        rx, ry = rng.uniform(0.05, 0.25) * w, rng.uniform(0.05, 0.3) * h
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=int(rng.integers(90, 200)))
    layer = layer.filter(ImageFilter.GaussianBlur(float(rng.uniform(2, 5))))
    dirt = Image.new("RGB", im.size, (70, 55, 40))
    return Image.composite(dirt, im, layer.point(lambda v: int(v * 0.7)))


def raindrops(im, rng, count: int):
    """雨滴（明るく、縁がぼけた丸）"""
    if count <= 0:
        return im
    layer = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(layer)
    for _ in range(count):
        cx, cy, r = rng.uniform(0, SCENE_W), rng.uniform(0, SCENE_H), rng.uniform(2, 7)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=int(rng.integers(80, 160)))
    layer = layer.filter(ImageFilter.GaussianBlur(1.5))
    return Image.composite(Image.new("RGB", im.size, (235, 240, 245)), im, layer.point(lambda v: int(v * 0.6)))


def occlusion(im, rng, n_lines: int, bbox):
    """部分的な隠れ（ワイパーや枠のような細い線が、板をまたぐ）"""
    if n_lines <= 0:
        return im
    im = im.copy()
    d = ImageDraw.Draw(im)
    x, y, w, h = bbox
    for _ in range(n_lines):
        width = int(rng.integers(1, 4))
        if rng.random() < 0.5:   # ほぼ縦
            px = rng.uniform(x, x + w)
            d.line((px + rng.uniform(-10, 10), y - 15, px + rng.uniform(-10, 10), y + h + 15), fill=(25, 25, 25), width=width)
        else:                    # ほぼ横
            py = rng.uniform(y, y + h)
            d.line((x - 15, py + rng.uniform(-6, 6), x + w + 15, py + rng.uniform(-6, 6)), fill=(25, 25, 25), width=width)
    return im


def blur(im, rng, sigma: float, motion_len: int):
    """ガウスぼかし（ピンぼけ）と、動きぶれ（ある向きに平均する）"""
    if sigma > 0:
        im = im.filter(ImageFilter.GaussianBlur(sigma))
    if motion_len and motion_len > 1:
        a = _arr(im)
        th = rng.uniform(0, np.pi)
        pad = motion_len
        ap = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        acc = np.zeros_like(a)
        for t in np.linspace(-motion_len / 2, motion_len / 2, motion_len + 1):
            dx, dy = int(round(t * np.cos(th))), int(round(t * np.sin(th)))
            acc += ap[pad + dy: pad + dy + SCENE_H, pad + dx: pad + dx + SCENE_W]
        im = _img(acc / (motion_len + 1))
    return im


def lowres(im, factor: float):
    """低解像度（縮小してから拡大）"""
    if factor <= 1.0:
        return im
    small = im.resize((max(8, int(SCENE_W / factor)), max(8, int(SCENE_H / factor))), Image.Resampling.BOX)
    return small.resize((SCENE_W, SCENE_H), Image.Resampling.BILINEAR)


def noise(im, rng, sigma: float):
    """ガウスノイズ（夜間・高感度撮影の想定）"""
    if sigma <= 0:
        return im
    a = _arr(im)
    a += rng.normal(0, sigma, a.shape).astype(np.float32)
    return _img(a)


def jpeg(im, quality: int):
    """JPEG圧縮のブロックノイズ"""
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_all(im, rng, p: dict, bbox) -> Image.Image:
    """パラメータ p に従って、劣化を順番に適用する（光 → 汚れ・雨・隠れ → ぼけ → 低解像度 → ノイズ → JPEG）"""
    im = lighting(im, rng, p["brightness"], p["contrast"], p["grad_strength"])
    im = stains(im, rng, p["stain_count"], bbox)
    im = raindrops(im, rng, p["rain_count"])
    im = occlusion(im, rng, p["occlusion_lines"], bbox)
    im = blur(im, rng, p["blur_sigma"], p["motion_len"])
    im = lowres(im, p["lowres"])
    im = noise(im, rng, p["noise_sigma"])
    return jpeg(im, p["jpeg_quality"])
