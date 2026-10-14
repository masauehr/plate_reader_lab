"""OCR用のデータ。

板の切り出し: 四隅で板をまっすぐに直した 384×192 の画像を、上段・下段に分けて横に並べ、1行（48×512）にする。
  → 読む順（地名・分類番号 → かな・一連番号）が、そのまま左から右の並びになる。
学習データ: その場で生成。四隅に「検出器のような誤差」を足して補正する（位置ずれに強くするため）。
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from classical.ocr import rectify  # noqa: E402
from plate_synth import DIFFICULTIES, generate_sample, load_labels, split_seed  # noqa: E402

from .model import encode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = ROOT / "data" / "eval"
STRIP_H, STRIP_W = 48, 256


def plate_to_tensor(plate: Image.Image) -> torch.Tensor:
    """まっすぐにした板(384×192) → (3, 48, 512)。上段と下段を横につなげる"""
    top = plate.crop((0, 8, 384, 92)).resize((STRIP_W, STRIP_H), Image.Resampling.BILINEAR)
    bot = plate.crop((0, 92, 384, 184)).resize((STRIP_W, STRIP_H), Image.Resampling.BILINEAR)
    line = Image.new("RGB", (STRIP_W * 2, STRIP_H))
    line.paste(top, (0, 0)); line.paste(bot, (STRIP_W, 0))
    return torch.from_numpy(np.asarray(line, dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def jitter_corners(corners, sigma: float, rng: np.random.Generator) -> np.ndarray:
    c = np.asarray(corners, dtype=np.float64)
    return c + rng.normal(0, sigma, c.shape) if sigma > 0 else c


class OcrSynthDataset(Dataset):
    """その場で生成する学習データ。corner_sigma_max: 四隅に足す誤差(px)の上限。画像ごとに 0〜この値 から選ぶ"""

    def __init__(self, n: int, offset: int = 0, diffs=tuple(DIFFICULTIES), corner_sigma_max: float = 3.0):
        self.n, self.offset, self.diffs, self.smax = n, offset, tuple(diffs), corner_sigma_max

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        d = self.diffs[k % len(self.diffs)]
        seed = split_seed("train", d, (self.offset + k // len(self.diffs)) % 100_000)    # 10万を超えたら先頭に戻る
        s = generate_sample(seed, d)
        rng = np.random.default_rng(seed + 7)
        sigma = rng.uniform(0, self.smax) if self.smax > 0 else 0.0
        plate = rectify(s.image, jitter_corners(s.label["corners"], sigma, rng))
        return plate_to_tensor(plate), torch.tensor(encode(s.label["text"]), dtype=torch.long)


def collate(batch):
    """文字列の長さがそろわないので、ラベルは連結して長さの一覧を付ける（CTCの入力形式）"""
    x = torch.stack([b[0] for b in batch])
    y = torch.cat([b[1] for b in batch])
    lens = torch.tensor([len(b[1]) for b in batch], dtype=torch.long)
    return x, y, lens


class EvalItems:
    """固定の評価データ（data/eval/）。画像と正解ラベルを読む"""

    def __init__(self, split: str, start: int, stop: int, diffs=tuple(DIFFICULTIES)):
        self.items = []
        for d in diffs:
            for lab in load_labels(EVAL / split / d / "labels.jsonl")[start:stop]:
                self.items.append((d, EVAL / split / d / "images" / lab["image"], lab))
