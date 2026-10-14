"""システム全体（検出→読み取り→書式検査→確信度→振り分け）の評価。Phase6の検証レポートと、最終評価(test)の両方で使う。

比べる4つの構成（プロジェクトの道のり）:
  P1 古典的な検出 ＋ 古典的なOCR（機械学習なし）
  P2 AIの検出 ＋ 古典的なOCR
  P3 AIの検出 ＋ CRNN（OCR）
  P4 AIの検出 ＋ CRNN ＋ 書式検査・確信度による振り分け（自動処理／要確認／読み取り不可）
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai_detect.model import make_model as make_detector  # noqa: E402
from ai_ocr.data import EvalItems  # noqa: E402
from ai_ocr.evaluate import predict_corners  # noqa: E402
from ai_ocr.infer import load_plates, read_plates  # noqa: E402
from ai_ocr.model import CRNN  # noqa: E402
from classical import read_plate, read_text, rectify  # noqa: E402
from plate_eval import cer, exact_match, iou  # noqa: E402
from rules.confidence import seq_prob  # noqa: E402
from rules.decision import choose_threshold, decide  # noqa: E402
from rules.format import is_valid, split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DIFFS = ["easy", "medium", "hard"]
MIX = {"均等(1:1:1)": {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3}, "昼間中心(0.6:0.3:0.1)": {"easy": 0.6, "medium": 0.3, "hard": 0.1}}
TARGET_RISK = 0.01                       # 自動処理の誤り率の目標（Phase4・5と同じ）


def corners_bbox(c) -> list:
    c = np.asarray(c)
    lo, hi = c.min(0), c.max(0)
    return [float(lo[0]), float(lo[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])]


def weights(df: pd.DataFrame, mix: dict) -> np.ndarray:
    n = df.difficulty.value_counts()
    return df.difficulty.map(lambda d: mix[d] / n[d]).to_numpy()


def thresholds(seeds, target: float = TARGET_RISK) -> dict:
    """各CRNNシードのしきい値。Phase4・5と同じ方法: 較正セット・昼間中心・書式検査のみ＋seq確信度。**評価データ（val/test）は使わない**"""
    out = {}
    for s in seeds:
        cal = pd.read_csv(ROOT / "outputs" / "rules" / f"seed{s}_cal_samples.csv")
        out[s] = float(choose_threshold(cal.raw_seq, cal.exact_raw, cal.raw_valid, target, weights(cal, MIX["昼間中心(0.6:0.3:0.1)"])))
    return out


def load_models(device, det_seed: int = 0, ocr_seeds=(0, 1, 2)):
    det = make_detector("resnet18").to(device)
    det.load_state_dict(torch.load(ROOT / "outputs" / "detect" / f"resnet18_all_s{det_seed}_best.pt", map_location=device)); det.eval()
    ocrs = {}
    for s in ocr_seeds:
        m = CRNN().to(device)
        m.load_state_dict(torch.load(ROOT / "outputs" / "ocr" / f"crnn_all_cs3_s{s}_best.pt", map_location=device)); m.eval()
        ocrs[s] = m
    return det, ocrs


def evaluate_system(items: list, det, ocrs: dict, taus: dict, device) -> pd.DataFrame:
    """items: [(難易度, 画像パス, ラベル)]。1枚=1行。CRNNのシードごとの列は _s{シード} を付ける"""
    pc = predict_corners(det, items, device)
    plates_pred = load_plates(items, pc)
    plates_gt = load_plates(items, [lab["corners"] for _, _, lab in items])
    read_pred = {s: read_plates(m, device, plates_pred) for s, m in ocrs.items()}
    read_gt = {s: read_plates(m, device, plates_gt) for s, m in ocrs.items()}
    rows = []
    for k, (d, path, lab) in enumerate(items):
        img = Image.open(path).convert("RGB")
        gt = lab["text"].replace(" ", "")
        r = dict(difficulty=d, image=lab["image"], gt=gt, iou=iou(corners_bbox(pc[k]), lab["bbox"]), corners_pred=[[round(float(x), 1), round(float(y), 1)] for x, y in pc[k]],
                 corner_err=float(np.linalg.norm(np.asarray(pc[k]) - np.asarray(lab["corners"]), axis=1).mean()))
        c = read_plate(img)                                                                # P1: 古典的な検出＋古典的なOCR
        r.update(p1_text=c.text.replace(" ", ""), p1_iou=iou(c.detection.bbox, lab["bbox"]) if c.detection else 0.0)
        try:
            r["p2_text"] = read_text(rectify(img, pc[k])).text.replace(" ", "")           # P2: AIの検出＋古典的なOCR
        except Exception:
            r["p2_text"] = ""
        for s in ocrs:
            rp, rg = read_pred[s][k], read_gt[s][k]
            valid, conf = is_valid(rp.text), (seq_prob(rp.logp, rp.text) if rp.text else 0.0)
            r.update({f"gtc_text_s{s}": rg.text, f"p3_text_s{s}": rp.text, f"valid_s{s}": valid, f"conf_s{s}": conf,
                      f"decision_s{s}": decide(valid, conf, taus[s])})
        rows.append(r)
    return pd.DataFrame(rows)


def tables(df: pd.DataFrame, seeds) -> dict:
    """評価結果の集計表（Markdown用のDataFrame）"""
    out = {}
    # 道のり: 4つの構成の完全一致・CER
    rows = []
    for d in DIFFS:
        x = df[df.difficulty == d]
        r = {"難易度": d, "枚数": len(x),
             "P1 古典（検出＋OCR）": np.mean([exact_match(t, g) if t else False for t, g in zip(x.p1_text, x["gt"])]),
             "P2 AI検出＋古典OCR": np.mean([exact_match(t, g) if t else False for t, g in zip(x.p2_text, x["gt"])])}
        p3 = [np.mean([t == g for t, g in zip(x[f"p3_text_s{s}"], x["gt"])]) for s in seeds]
        p3gt = [np.mean([t == g for t, g in zip(x[f"gtc_text_s{s}"], x["gt"])]) for s in seeds]
        r["P3 AI検出＋CRNN"] = float(np.mean(p3)); r["(参考)CRNN・正解の四隅"] = float(np.mean(p3gt)); r["P3の標準偏差"] = float(np.std(p3, ddof=1)) if len(seeds) > 1 else 0.0
        rows.append(r)
    out["path"] = pd.DataFrame(rows).astype({"枚数": int}).set_index("難易度")
    # 検出
    out["det"] = pd.DataFrame([{"難易度": d, "検出率(IoU≥0.5)": (df[df.difficulty == d].iou >= .5).mean(), "Det@0.9": (df[df.difficulty == d].iou >= .9).mean(),
                                "平均IoU": df[df.difficulty == d].iou.mean(), "四隅誤差(px)": df[df.difficulty == d].corner_err.mean(),
                                "古典の検出率(IoU≥0.5)": (df[df.difficulty == d].p1_iou >= .5).mean()} for d in DIFFS]).set_index("難易度")
    # フィールド別（P3）
    rows = []
    for d in DIFFS:
        x = df[df.difficulty == d]
        acc = {f: [] for f in ("region", "class_no", "kana", "serial")}
        for s in seeds:
            for g, t in zip(x["gt"], x[f"p3_text_s{s}"]):
                gf, tf = split(g), split(t)
                for f in acc:
                    acc[f].append(gf[f] == tf[f])
        rows.append({"難易度": d, "地名": np.mean(acc["region"]), "分類番号": np.mean(acc["class_no"]), "かな": np.mean(acc["kana"]), "一連番号": np.mean(acc["serial"])})
    out["fields"] = pd.DataFrame(rows).set_index("難易度")
    # 振り分け（P4）: 難易度の混ぜ方ごと
    rows = []
    for mixname, mix in MIX.items():
        per = []
        for s in seeds:
            w = weights(df, mix); w = w / w.sum()
            dec, ok = df[f"decision_s{s}"].to_numpy(), (df[f"p3_text_s{s}"] == df["gt"]).to_numpy()
            auto = dec == "自動処理"
            wa = w[auto].sum()
            per.append(dict(自動処理率=wa, 自動処理の誤り率=w[auto & ~ok].sum() / wa if wa else np.nan, 要確認率=w[dec == "要確認"].sum(), 読み取り不可率=w[dec == "読み取り不可"].sum(),
                            人手対応率=w[~auto].sum(), 全体の完全一致率=(w * ok).sum()))
        m = pd.DataFrame(per)
        rows.append({"難易度の混ぜ方": mixname, **{c: f"{m[c].mean():.4f}" + (f" ±{m[c].std(ddof=1):.4f}" if len(seeds) > 1 else "") for c in m.columns}})
    out["decision"] = pd.DataFrame(rows).set_index("難易度の混ぜ方")
    # 難易度別の振り分け
    rows = []
    for d in DIFFS:
        x = df[df.difficulty == d]
        per = []
        for s in seeds:
            dec, ok = x[f"decision_s{s}"].to_numpy(), (x[f"p3_text_s{s}"] == x["gt"]).to_numpy()
            auto = dec == "自動処理"
            per.append(dict(自動処理率=auto.mean(), 自動処理の誤り率=(~ok[auto]).mean() if auto.any() else np.nan, 要確認率=(dec == "要確認").mean(), 読み取り不可率=(dec == "読み取り不可").mean()))
        m = pd.DataFrame(per)
        rows.append({"難易度": d, **{c: f"{m[c].mean():.4f}" for c in m.columns}})
    out["decision_by_diff"] = pd.DataFrame(rows).set_index("難易度")
    return out


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """DataFrame → Markdown表（整数の列は整数のまま表示する）"""
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples():
        cells = [f"{v:{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v) for v in row[1:]]
        lines.append("| " + " | ".join([str(row[0])] + cells) + " |")
    return "\n".join(lines)
