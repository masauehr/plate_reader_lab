"""plate_synth — ナンバー風の合成画像と正解ラベルの生成。

使い方:
    from plate_synth import generate_sample
    s = generate_sample(seed=0, difficulty="medium")
    s.image   # シーン画像（PIL）
    s.label   # 正解ラベル（dict）
"""
from .dataset import Sample, generate_sample, load_labels, make_eval_set
from .difficulty import CLEAN, sample_params
from .plate import format_serial, random_label, render_plate
from .spec import DIFFICULTIES, charset, split_seed

__all__ = ["Sample", "generate_sample", "make_eval_set", "load_labels", "sample_params", "CLEAN",
           "random_label", "render_plate", "format_serial", "DIFFICULTIES", "charset", "split_seed"]
