"""確認用の描画（Notebook用）。"""
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

plt.rcParams["font.family"] = ["Hiragino Sans", "AppleGothic", "DejaVu Sans"]


def draw_label(img: Image.Image, label: dict, color_box=(255, 0, 0), color_corner=(0, 200, 255)) -> Image.Image:
    """正解の枠（赤）と四隅（水色の点）を画像に描く"""
    im = img.copy()
    d = ImageDraw.Draw(im)
    x, y, w, h = label["bbox"]
    d.rectangle((x, y, x + w, y + h), outline=color_box, width=2)
    for cx, cy in label["corners"]:
        d.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=color_corner)
    return im


def show_grid(images, titles=None, cols=4, size=3.0):
    """画像を格子状に並べて表示する"""
    rows = -(-len(images) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(size * cols, (size * 0.8 + 0.4) * rows))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes:
        ax.axis("off")
    for i, (ax, im) in enumerate(zip(axes, images)):
        ax.imshow(im)
        if titles is not None:
            ax.set_title(titles[i], fontsize=9)
    fig.tight_layout()
    return fig
