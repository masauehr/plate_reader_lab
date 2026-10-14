"""シミュレーションの世界: 車（ナンバー・入庫時刻・出庫時刻・撮影条件）を作る。

似たナンバーの車（車列のように、連番や1文字違いの車が同時にいる状況）を、割合 similar_fraction で混ぜられる。
ここが、誤課金が起こりうる状況（別の車と取り違える）になる。
"""
from dataclasses import dataclass

import numpy as np

from plate_synth.plate import random_label
from rules.format import is_valid

# Phase3で多かった取り違え（形の似た文字）。似たナンバーは、この取り違えで入れ替わる位置に作る
CONFUSABLE = {"6": "85", "8": "69", "5": "69", "9": "58", "1": "7", "7": "1", "0": "6", "2": "37", "3": "28", "4": "9"}
MIX_DAY = {"easy": 0.6, "medium": 0.3, "hard": 0.1}                   # 昼間中心
MIX_EQUAL = {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3}


@dataclass
class Vehicle:
    vid: int
    label: dict
    t_in: float                  # 入庫時刻（分）
    t_out: float                 # 出庫時刻（分）
    diff_in: str                 # 入庫時の撮影条件
    diff_out: str                # 出庫時の撮影条件


def label_from_text(text: str, base: dict) -> dict:
    t = text.replace(" ", "")
    lab = dict(base)
    lab.update(region=t[:2], class_no=t[2:5], kana=t[5:6], serial=t[6:], text=f"{t[:2]} {t[2:5]} {t[5:6]} {t[6:]}")
    return lab


def neighbor_label(label: dict, rng: np.random.Generator) -> dict | None:
    """label の数字を1文字だけ、形の似た数字に替えたナンバー。書式が正しいものだけ。作れなければ None"""
    t = label["text"].replace(" ", "")
    pos = [i for i, ch in enumerate(t) if ch in CONFUSABLE]            # 数字の位置（地名・かな・ハイフン・「・」は対象外）
    rng.shuffle(pos)
    for i in pos:
        for alt in rng.permutation(list(CONFUSABLE[t[i]])):
            new = t[:i] + alt + t[i + 1:]
            if new != t and is_valid(new):
                return label_from_text(new, label)
    return None


def make_world(n: int, seed: int, similar_fraction: float = 0.0, mix: dict = MIX_DAY, arrival_per_min: float = 0.5) -> list[Vehicle]:
    """n台の車。入庫は一定の割合（1分あたり arrival_per_min 台）で到着し、駐車時間は短・中・長の混合"""
    rng = np.random.default_rng(10_000 + seed)
    diffs, probs = list(mix), np.array(list(mix.values()), float)
    t, vs, seen, recent = 0.0, [], set(), []
    for vid in range(n):
        t += rng.exponential(1 / arrival_per_min)
        lab = None
        if recent and rng.random() < similar_fraction:               # 直近に入庫した車と、似たナンバーにする
            lab = neighbor_label(recent[rng.integers(len(recent))], rng)
        if lab is None or lab["text"] in seen:
            while True:
                lab = random_label(rng)
                if lab["text"] not in seen:
                    break
        seen.add(lab["text"]); recent = (recent + [lab])[-6:]
        kind = rng.random()
        stay = rng.lognormal(np.log(60), 0.6) if kind < 0.70 else (rng.lognormal(np.log(240), 0.5) if kind < 0.95 else rng.lognormal(np.log(1500), 0.4))
        stay = max(10.0, stay)
        vs.append(Vehicle(vid, lab, t, t + stay, str(rng.choice(diffs, p=probs / probs.sum())), str(rng.choice(diffs, p=probs / probs.sum()))))
    return vs
