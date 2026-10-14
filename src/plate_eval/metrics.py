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


def align_ops(gt: str, pred: str) -> list[tuple]:
    """正解と読み取りを、編集距離が最小になるように対応付ける（混同行列の材料）。
    戻り値: [(操作, 正解の文字, 読んだ文字)]。操作は "match"（一致）/ "sub"（取り違え）/ "del"（脱落＝読めなかった）/ "ins"（余分に読んだ）"""
    g, p = gt.replace(" ", ""), pred.replace(" ", "")
    n, m = len(g), len(p)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (g[i - 1] != p[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (g[i - 1] != p[j - 1]):
            ops.append(("match" if g[i - 1] == p[j - 1] else "sub", g[i - 1], p[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("del", g[i - 1], None)); i -= 1
        else:
            ops.append(("ins", None, p[j - 1])); j -= 1
    return ops[::-1]
