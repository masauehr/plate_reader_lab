"""振り分け（自動処理／要確認／読み取り不可）、しきい値の決め方、カバレッジ曲線。"""
import numpy as np

AUTO, REVIEW, REJECT = "自動処理", "要確認", "読み取り不可"


def decide(valid: bool, conf: float, tau_auto: float) -> str:
    """書式が正しく、確信度がしきい値以上なら自動処理。書式が直せなければ読み取り不可。それ以外は要確認（候補を添えて人へ）"""
    if not valid:
        return REJECT
    return AUTO if conf >= tau_auto else REVIEW


def risk_coverage(conf, correct, valid=None, w=None):
    """確信度の高い順に自動処理していったときの、(自動処理率, 自動処理の誤り率, そのときのしきい値) の曲線。
    書式が直せないもの(valid=False)は、最初から自動処理しない。w は画像ごとの重み（難易度の混ぜ方の想定）"""
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, bool)
    w = np.ones(len(conf)) if w is None else np.asarray(w, float)
    c = np.where(np.asarray(valid, bool), conf, -1.0) if valid is not None else conf
    order = np.argsort(-c, kind="stable")
    ww, ok = w[order], correct[order]
    cov = np.cumsum(ww) / ww.sum()                                  # 自動処理率（重み付き）
    err = np.cumsum(ww * ~ok) / np.cumsum(ww)                       # 自動処理した中の誤り率
    keep = c[order] >= 0
    return cov[keep], err[keep], c[order][keep]


def choose_threshold(conf, correct, valid, target_risk: float, w=None) -> float:
    """検証データで、自動処理の誤り率が target_risk 以下のうち、自動処理率が最大になるしきい値を選ぶ。
    どこまで絞っても誤りが残るなら、何も自動処理しないしきい値（2.0）を返す"""
    cov, err, thr = risk_coverage(conf, correct, valid, w)
    good = np.where(err <= target_risk)[0]
    return float(thr[good.max()]) if len(good) else 2.0


def choose_threshold_by_cost(conf, correct, valid, cost_review: float, cost_error: float, w=None) -> float:
    """1台あたりの期待コスト = 要確認等の率×cost_review + 自動処理して誤った率×cost_error を最小にするしきい値"""
    conf, correct, valid = np.asarray(conf, float), np.asarray(correct, bool), np.asarray(valid, bool)
    w = np.ones(len(conf)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    best, best_cost = 2.0, np.inf
    for t in np.unique(np.concatenate([conf[valid], [2.0]])):
        auto = valid & (conf >= t)
        cost = (w[~auto].sum() * cost_review) + (w[auto & ~correct].sum() * cost_error)
        if cost < best_cost:
            best, best_cost = float(t), float(cost)
    return best


def apply(conf, correct, valid, tau: float, w=None) -> dict:
    """しきい値 tau を使ったときの結果（重み付きの割合）"""
    conf, correct, valid = np.asarray(conf, float), np.asarray(correct, bool), np.asarray(valid, bool)
    w = np.ones(len(conf)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    auto = valid & (conf >= tau)
    wa = w[auto].sum()
    return dict(自動処理率=float(wa), 自動処理の誤り率=float(w[auto & ~correct].sum() / wa) if wa > 0 else float("nan"),
                要確認率=float(w[valid & ~auto].sum()), 読み取り不可率=float(w[~valid].sum()),
                誤自動処理率=float(w[auto & ~correct].sum()))      # 全体に対する「自動処理して間違えた」割合


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """順位（1から）。同じ値は、順位の平均を与える（同点の扱い）"""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(conf, correct) -> float:
    """確信度が、正解と誤りをどれくらい分けるか（0.5=偶然、1=完全）。ランダムに選んだ正解の確信度が、誤りの確信度より高い確率（同点は半分）"""
    conf, correct = np.asarray(conf, float), np.asarray(correct, bool)
    pos, neg = conf[correct], conf[~correct]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _average_ranks(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
