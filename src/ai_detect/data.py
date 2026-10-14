"""学習・評価用のデータ。

学習データは、エポックごとに**その場で新しく生成**する（シードの範囲は train 用。val/test と重ならない）。
評価データは、Phase0 で作った固定ファイル（data/eval/）を使う。
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from plate_synth import DIFFICULTIES, generate_sample, load_labels, split_seed
from plate_synth.spec import SCENE_H, SCENE_W

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL = ROOT / "data" / "eval"
_SCALE = np.array([SCENE_W, SCENE_H] * 4, dtype=np.float32)


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1).contiguous()


def corners_to_target(corners) -> torch.Tensor:
    """四隅(4,2)のピクセル座標 → 0〜1に正規化した8個の数"""
    return torch.from_numpy(np.asarray(corners, dtype=np.float32).reshape(-1) / _SCALE)


def target_to_corners(t) -> np.ndarray:
    """正規化した8個の数 → ピクセル座標(4,2)"""
    return (np.asarray(t, dtype=np.float32).reshape(-1) * _SCALE).reshape(4, 2)


class SynthDataset(Dataset):
    """その場で生成する学習データ。offset を進めれば、エポックごとに別の画像になる（難易度は順番に混ぜる）"""

    def __init__(self, n: int, offset: int = 0, diffs=tuple(DIFFICULTIES)):
        self.n, self.offset, self.diffs = n, offset, tuple(diffs)

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        d = self.diffs[k % len(self.diffs)]
        s = generate_sample(split_seed("train", d, self.offset + k // len(self.diffs)), d)
        return to_tensor(s.image), corners_to_target(s.label["corners"])


class FileDataset(Dataset):
    """固定の評価データ（data/eval/{split}/{難易度}/）。start〜stop 番目の画像"""

    def __init__(self, split: str, start: int, stop: int, diffs=tuple(DIFFICULTIES)):
        self.items = []
        for d in diffs:
            for lab in load_labels(EVAL / split / d / "labels.jsonl")[start:stop]:
                self.items.append((d, EVAL / split / d / "images" / lab["image"], lab))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        d, path, lab = self.items[k]
        return to_tensor(Image.open(path)), corners_to_target(lab["corners"]), k
