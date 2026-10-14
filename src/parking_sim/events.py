"""入出庫の画像を作り、AI（検出→四隅→OCR→書式検査→確信度）で読む。学習済みモデルが必要（Phase2・3）。

方針（Phase4の結論）: 補正は使わず、書式検査（読み取り不可の判定）と seq 確信度を使う。
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai_detect.data import target_to_corners, to_tensor  # noqa: E402
from ai_detect.model import make_model as make_detector  # noqa: E402
from ai_ocr.data import plate_to_tensor  # noqa: E402
from ai_ocr.infer import read_plates  # noqa: E402
from ai_ocr.model import CRNN  # noqa: E402
from classical.ocr import rectify  # noqa: E402
from plate_synth import generate_sample, split_seed  # noqa: E402
from rules.confidence import seq_prob  # noqa: E402
from rules.format import is_valid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
SEED_STRIDE_SIM = 30_000          # シミュレーションのシードごとに、画像の通し番号をずらす幅（難易度あたり10万まで）


def _gen(job):
    seed, diff, label = job
    return np.asarray(generate_sample(seed, diff, label=label).image)


def load_models(device, det_seed: int = 0, ocr_seed: int = 0):
    det = make_detector("resnet18").to(device)
    det.load_state_dict(torch.load(ROOT / "outputs" / "detect" / f"resnet18_all_s{det_seed}_best.pt", map_location=device)); det.eval()
    ocr = CRNN().to(device)
    ocr.load_state_dict(torch.load(ROOT / "outputs" / "ocr" / f"crnn_all_cs3_s{ocr_seed}_best.pt", map_location=device)); ocr.eval()
    return det, ocr


@torch.no_grad()
def read_images(images: list[np.ndarray], det, ocr, device) -> list[dict]:
    """シーン画像 → 四隅 → 板 → 文字列。書式の正しさと確信度(seq)つき"""
    pil = [Image.fromarray(a) for a in images]
    corners = []
    for i in range(0, len(pil), 64):
        x = torch.stack([to_tensor(im) for im in pil[i:i + 64]]).to(device)
        corners += [target_to_corners(q) for q in det(x).cpu().numpy()]
    plates = []
    for im, c in zip(pil, corners):
        try:
            plates.append(plate_to_tensor(rectify(im, c)))
        except Exception:
            plates.append(torch.zeros(3, 48, 512))
    out = []
    for r in read_plates(ocr, device, plates):
        out.append(dict(text=r.text, valid=is_valid(r.text), conf=seq_prob(r.logp, r.text) if r.text else 0.0))
    return out


def build_event_table(world, sim_seed: int, det, ocr, device, workers: int = 6, chunk: int = 600) -> pd.DataFrame:
    """全車の入庫・出庫の画像を作って読み、事象の表（1行=1事象）にする"""
    jobs, meta = [], []
    for v in world:
        for k, (kind, diff, t) in enumerate((("in", v.diff_in, v.t_in), ("out", v.diff_out, v.t_out))):
            idx = sim_seed * SEED_STRIDE_SIM + v.vid * 2 + k
            jobs.append((split_seed("sim", diff, idx), diff, v.label))
            meta.append(dict(vid=v.vid, kind=kind, t=t, difficulty=diff, gt=v.label["text"].replace(" ", "")))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i in range(0, len(jobs), chunk):
            imgs = list(ex.map(_gen, jobs[i:i + chunk], chunksize=20))
            for m, r in zip(meta[i:i + chunk], read_images(imgs, det, ocr, device)):
                rows.append({**m, **r})
    return pd.DataFrame(rows)
