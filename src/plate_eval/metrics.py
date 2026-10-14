"""評価指標の実装。定義は EVALUATION.md を参照。"""


def iou(a, b) -> float:
    """2つの枠 [x, y, 幅, 高さ] の IoU（重なり ÷ 合わせた面積）。0〜1"""
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def edit_distance(a: str, b: str) -> int:
    """編集距離（置換・挿入・削除の最小回数）"""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred: str, gt: str) -> float:
    """文字誤り率 = 編集距離 ÷ 正解の文字数（空白は無視）"""
    p, g = pred.replace(" ", ""), gt.replace(" ", "")
    return edit_distance(p, g) / max(1, len(g))


def exact_match(pred: str, gt: str) -> bool:
    """完全一致（空白は無視）"""
    return pred.replace(" ", "") == gt.replace(" ", "")


def field_matches(pred: dict, gt: dict) -> dict:
    """フィールド（地名・分類番号・かな・一連番号）ごとの一致"""
    return {k: pred.get(k) == gt[k] for k in ("region", "class_no", "kana", "serial")}
