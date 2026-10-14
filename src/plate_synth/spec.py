"""プレート・シーンの仕様（定数）。詳細は DATA_SPEC.md を参照。"""

# 地名・かな・分類番号の英字（学習時間のため限定。DATA_SPEC.md 2-2）
REGIONS = ["沖縄", "那覇", "品川", "横浜", "大阪"]
KANA = list("あかさたなはまやらわ")
CLASS_LETTERS = list("ACFHKLMPXY")
DOT = "・"      # 一連番号の上位の0を表す記号
HYPHEN = "-"

# プレートの種別: 地の色(bg)・文字の色(fg)。数値は学習用の簡略化（実物の色とは異なる）
PLATE_TYPES = {
    "private":      dict(name="自家用",     bg=(247, 247, 240), fg=(0, 98, 60)),
    "business":     dict(name="事業用",     bg=(0, 98, 60),     fg=(247, 247, 240)),
    "kei_private":  dict(name="軽自家用",   bg=(250, 213, 40),  fg=(20, 20, 20)),
    "kei_business": dict(name="軽事業用",   bg=(25, 25, 25),    fg=(250, 213, 40)),
}

# 描画サイズ（縦横比 2:1）とシーンのサイズ
PLATE_W, PLATE_H = 384, 192
SCENE_W, SCENE_H = 320, 240

DIFFICULTIES = ["easy", "medium", "hard"]

# データ分割のシード範囲。重ならないようにベースを離す（難易度ごとに10万ずつ）
SEED_BASE = {"train": 1_000_000, "val": 2_000_000, "test": 3_000_000}
SEED_STRIDE = 100_000


def split_seed(split: str, difficulty: str, index: int) -> int:
    """分割・難易度・通し番号から、生成用のシードを決める（分割間でシードが重ならない）"""
    if not 0 <= index < SEED_STRIDE:
        raise ValueError(f"index は 0〜{SEED_STRIDE - 1} の範囲にしてください")
    return SEED_BASE[split] + DIFFICULTIES.index(difficulty) * SEED_STRIDE + index


def charset() -> list[str]:
    """OCRで読む文字の一覧（Phase3で使う）。空白は含まない"""
    chars = list("0123456789") + CLASS_LETTERS + KANA + [DOT, HYPHEN]
    for r in REGIONS:
        chars += [c for c in r if c not in chars]
    return chars
