"""四隅を回帰するモデル。出力は (左上, 右上, 右下, 左下) の x, y を 0〜1 に正規化した8個の数。"""
import torch
from torch import nn
from torchvision import models

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)     # ImageNetの統計


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("m", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("s", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.m) / self.s


def _block(i, o, stride):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class SmallCorner(nn.Module):
    """自作の小さなCNN（ゼロから学習）。5回縮小して 8×10 の特徴マップにし、**位置を保ったまま**全結合で四隅を出す"""

    def __init__(self):
        super().__init__()
        self.norm = Normalize()
        self.features = nn.Sequential(_block(3, 32, 2), _block(32, 64, 2), _block(64, 128, 2), _block(128, 128, 2), _block(128, 256, 2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256 * 8 * 10, 256), nn.ReLU(inplace=True), nn.Linear(256, 8))

    def forward(self, x):
        return self.head(self.features(self.norm(x)))


class ResNetCorner(nn.Module):
    """ImageNetで学習済みのResNet18を借りる（転移学習）。最後の平均化は使わず、位置を保ったまま四隅を出す"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.norm = Normalize()
        r = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.backbone = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.reduce = nn.Sequential(nn.Conv2d(512, 64, 1), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 10, 256), nn.ReLU(inplace=True), nn.Linear(256, 8))

    def forward(self, x):
        return self.head(self.reduce(self.backbone(self.norm(x))))


def make_model(name: str) -> nn.Module:
    if name == "small":
        return SmallCorner()
    if name == "resnet18":
        return ResNetCorner(True)
    if name == "resnet18_scratch":
        return ResNetCorner(False)
    raise ValueError(name)
