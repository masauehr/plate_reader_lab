"""日本語フォントの検出。フォントファイルは再配布せず、手元のOS標準フォントを使う。"""
import os
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

# macOS標準の日本語フォント（太め）。環境変数 PLATE_FONT で上書きできる
CANDIDATES = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]


def find_font_path() -> str:
    env = os.environ.get("PLATE_FONT")
    for p in ([env] if env else []) + CANDIDATES:
        if p and Path(p).exists():
            return p
    raise FileNotFoundError(
        "日本語フォントが見つかりません。環境変数 PLATE_FONT にフォントファイル（.ttc/.ttf）のパスを指定してください。"
    )


@lru_cache(maxsize=64)
def get_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(find_font_path(), size)
