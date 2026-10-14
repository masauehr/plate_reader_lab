"""CRNN＋CTC。画像（1行にした板）を左から右へ読み、各位置で「どの文字か（または、何もない）」を出す。"""
import numpy as np
import torch
from torch import nn

from plate_synth.spec import charset

CHARS = charset()                       # 読める文字（数字・英字・かな・「・」「-」・地名の漢字）
BLANK = 0                               # CTCの「何もない」記号
VOCAB = len(CHARS) + 1                  # blank + 文字
_IDX = {c: i + 1 for i, c in enumerate(CHARS)}


def encode(text: str) -> list[int]:
    return [_IDX[c] for c in text.replace(" ", "")]


def decode_greedy(log_probs: torch.Tensor) -> list[tuple[str, list[float]]]:
    """各時刻でいちばん確からしい記号を選び、同じ記号の連続をまとめ、blankを除く。
    戻り値: バッチごとの (文字列, 文字ごとの確率)。log_probs: (T, B, VOCAB)"""
    p = log_probs.exp().cpu().numpy()                       # (T, B, V)
    out = []
    for b in range(p.shape[1]):
        best = p[:, b].argmax(1)
        conf = p[:, b].max(1)
        s, cs, prev, run = [], [], -1, []
        for t, k in enumerate(best):
            if k != prev:
                if prev > 0:
                    s.append(CHARS[prev - 1]); cs.append(float(max(run)))
                run = []
            run.append(conf[t]); prev = k
        if prev > 0:
            s.append(CHARS[prev - 1]); cs.append(float(max(run)))
        out.append(("".join(s), cs))
    return out


def _conv(i, o):
    return [nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """入力 (B, 3, 48, 512) → 出力 (T=128, B, VOCAB) の対数確率"""

    def __init__(self, hidden: int = 192):
        super().__init__()
        self.cnn = nn.Sequential(
            *_conv(3, 32), nn.MaxPool2d(2, 2),                     # 24 × 256
            *_conv(32, 64), nn.MaxPool2d(2, 2),                    # 12 × 128
            *_conv(64, 128), *_conv(128, 128), nn.MaxPool2d((2, 1)),   # 6 × 128（横は保つ）
            *_conv(128, 192), *_conv(192, 192), nn.MaxPool2d((2, 1)),  # 3 × 128
        )
        self.proj = nn.Linear(192 * 3, hidden)
        self.rnn = nn.LSTM(hidden, hidden, num_layers=2, bidirectional=True, batch_first=False, dropout=0.1)
        self.fc = nn.Linear(hidden * 2, VOCAB)

    def forward(self, x):
        f = self.cnn((x - 0.5) / 0.25)                       # (B, C, 3, 128)
        b, c, h, w = f.shape
        f = f.permute(3, 0, 1, 2).reshape(w, b, c * h)       # (T, B, C*H): 横方向を時間とみなす
        f = torch.relu(self.proj(f))
        f, _ = self.rnn(f)
        return self.fc(f).log_softmax(2)
