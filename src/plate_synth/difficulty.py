"""難易度ごとのパラメータ（DATA_SPEC.md 6）。値の範囲は初期案で、実験しながら調整する。"""
import numpy as np

# 劣化・変形をすべて切った「きれいな」基準。単独の劣化を試すときの土台にする
CLEAN = dict(
    scale=0.45, rot_deg=0.0, persp=0.0,                  # 板の大きさ(画面幅比)、回転、透視の歪み
    brightness=1.0, contrast=1.0, grad_strength=0.0,     # 明るさ、コントラスト、逆光・影
    stain_count=0, rain_count=0, occlusion_lines=0,      # 汚れ、雨滴、部分的な隠れ
    blur_sigma=0.0, motion_len=0, lowres=1.0,            # ぼけ、動きぶれ、低解像度
    noise_sigma=0.0, jpeg_quality=95,                    # ノイズ、JPEG圧縮
    distractors=0, bg_type="wall",                       # 紛らわしい物の数、背景の種類
)

# 各項目の (最小, 最大) 。整数の項目は整数で抽出する。確率つきの項目は (確率, 最小, 最大)
RANGES = {
    "easy": dict(
        scale=(0.40, 0.55), rot_deg=(-2, 2), persp=(0.0, 0.02),
        brightness=(0.9, 1.1), contrast=(0.9, 1.1), grad=(0.0, 0.0, 0.0),
        stain=(0.0, 0, 0), rain=(0.0, 0, 0), occlusion=(0.0, 0, 0),
        blur_sigma=(0.0, 0.5), motion=(0.0, 0, 0), lowres=(1.0, 1.0),
        noise=(0.0, 3.0), jpeg=(85, 95), distractors=(0, 0),
    ),
    "medium": dict(
        scale=(0.30, 0.45), rot_deg=(-6, 6), persp=(0.02, 0.06),
        brightness=(0.7, 1.2), contrast=(0.7, 1.2), grad=(0.3, 0.1, 0.4),
        stain=(0.3, 1, 2), rain=(0.2, 8, 20), occlusion=(0.1, 1, 1),
        blur_sigma=(0.3, 1.2), motion=(0.3, 2, 3), lowres=(1.0, 1.5),
        noise=(3.0, 8.0), jpeg=(60, 85), distractors=(0, 1),
    ),
    "hard": dict(
        scale=(0.20, 0.35), rot_deg=(-12, 12), persp=(0.06, 0.12),
        brightness=(0.4, 1.0), contrast=(0.5, 1.1), grad=(0.6, 0.3, 0.6),
        stain=(0.7, 2, 4), rain=(0.5, 15, 40), occlusion=(0.4, 1, 2),
        blur_sigma=(0.8, 2.0), motion=(0.5, 3, 5), lowres=(1.5, 3.0),
        noise=(8.0, 18.0), jpeg=(30, 60), distractors=(1, 3),
    ),
}

BG_TYPES = ["wall", "gradient", "car"]


def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _prob_int(rng, spec) -> int:
    """(確率, 最小, 最大) から整数を抽出。確率で外れたら0"""
    p, lo, hi = spec
    return int(rng.integers(lo, hi + 1)) if (hi > 0 and rng.random() < p) else 0


def sample_params(difficulty: str, rng: np.random.Generator, overrides: dict | None = None) -> dict:
    """難易度から、1枚ぶんの具体的なパラメータを抽出する。overridesで一部を上書きできる"""
    r = RANGES[difficulty]
    p = dict(CLEAN)
    p.update(
        scale=_u(rng, *r["scale"]), rot_deg=_u(rng, *r["rot_deg"]), persp=_u(rng, *r["persp"]),
        brightness=_u(rng, *r["brightness"]), contrast=_u(rng, *r["contrast"]),
        blur_sigma=_u(rng, *r["blur_sigma"]), lowres=_u(rng, *r["lowres"]),
        noise_sigma=_u(rng, *r["noise"]), jpeg_quality=int(rng.integers(r["jpeg"][0], r["jpeg"][1] + 1)),
        distractors=int(rng.integers(r["distractors"][0], r["distractors"][1] + 1)),
        bg_type=BG_TYPES[rng.integers(len(BG_TYPES))],
    )
    pg, glo, ghi = r["grad"]
    p["grad_strength"] = _u(rng, glo, ghi) if (ghi > 0 and rng.random() < pg) else 0.0
    p["stain_count"] = _prob_int(rng, r["stain"])
    p["rain_count"] = _prob_int(rng, r["rain"])
    p["occlusion_lines"] = _prob_int(rng, r["occlusion"])
    pm, mlo, mhi = r["motion"]
    p["motion_len"] = int(rng.integers(mlo, mhi + 1)) if (mhi > 0 and rng.random() < pm) else 0
    if overrides:
        unknown = set(overrides) - set(CLEAN)
        if unknown:
            raise KeyError(f"未知のパラメータ: {sorted(unknown)}")
        p.update(overrides)
    return p
