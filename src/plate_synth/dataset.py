"""1枚の生成・評価セットの作成。すべてシードで再現できる。"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import degrade, scene
from .difficulty import sample_params
from .plate import random_label, render_plate
from .spec import DIFFICULTIES, SCENE_H, SCENE_W, split_seed


@dataclass
class Sample:
    image: Image.Image     # シーン画像（RGB, 320×240）
    label: dict            # 正解ラベル（DATA_SPEC.md 7）
    plate: Image.Image     # きれいなプレート画像（OCR学習や確認用）


def generate_sample(seed: int, difficulty: str = "easy", overrides: dict | None = None,
                    plate_type: str = "private") -> Sample:
    """シードと難易度から、シーン画像と正解ラベルを1つ作る。同じ引数なら必ず同じ結果"""
    rng = np.random.default_rng(seed)
    label = random_label(rng, plate_type)
    plate = render_plate(label, rng)
    p = sample_params(difficulty, rng, overrides)

    bg = scene.make_background(rng, p["bg_type"])
    scene.add_distractors(bg, rng, p["distractors"])
    corners = scene.choose_corners(rng, p["scale"], p["rot_deg"], p["persp"])
    img = scene.paste_plate(bg, plate, corners)
    bbox = scene.bbox_of(corners)
    img = degrade.apply_all(img, rng, p, bbox)

    label.update(
        bbox=[round(v, 2) for v in bbox],
        corners=[[round(float(x), 2), round(float(y), 2)] for x, y in corners],
        difficulty=difficulty, degradations={k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()},
        seed=seed, image_size=[SCENE_W, SCENE_H],
    )
    return Sample(img, label, plate)


def make_eval_set(root: Path, split: str, difficulty: str, n: int, start: int = 0, out_split: str | None = None) -> Path:
    """固定シードの評価セットを root/(out_split または split)/difficulty/ に書き出す（画像PNG＋labels.jsonl）。
    start番目から n 枚。シードは split の範囲から取る（例: split="val", start=500 → valの500番目以降＝しきい値決定用の較正セット）"""
    if n <= 0:
        raise ValueError("n は1以上にしてください（0だと既存のlabels.jsonlを空で上書きしてしまう）")
    out = Path(root) / (out_split or split) / difficulty
    (out / "images").mkdir(parents=True, exist_ok=True)
    with open(out / "labels.jsonl", "w", encoding="utf-8") as f:
        for i in range(n):
            s = generate_sample(split_seed(split, difficulty, start + i), difficulty)
            name = f"{i:05d}.png"
            s.image.save(out / "images" / name)          # PNGで保存（再圧縮を避ける）
            f.write(json.dumps({"image": name, **s.label}, ensure_ascii=False) + "\n")
    return out


def load_labels(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
