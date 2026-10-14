"""Phase7: 学習なしのAI（VLM）と、自作のシステムを、同じ画像で比べる。

使い方:
    # 1) VLMに読ませる（結果は outputs/vlm/raw/ にキャッシュ。途中で止まっても再開できる）
    .venv/bin/python src/run_phase7.py ask --split test2 --n 60 --models ollama:qwen3.8:27b-mlx --prompts format naive
    .venv/bin/python src/run_phase7.py ask --split test2 --n 60 --models claude:haiku claude:sonnet --prompts format
    # 2) 集計（自作のシステムも、同じ画像で評価する）→ outputs/vlm/summary.md
    .venv/bin/python src/run_phase7.py report --split test2 --n 60

評価データ: test2 = testのシードの1000番目以降から作った、**一度も使っていない新しいセット**（make_eval_sets.py --test2 100）。
  開発・動作確認は val（--split val）で行い、test2 は本番の比較で1回だけ使う。
  claude: は画像が Anthropic に送信される（**合成画像のみ**）。ollama: はローカル（画像は外に出ない）。
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_eval import cer  # noqa: E402
from rules.decision import auroc  # noqa: E402
from rules.format import is_valid, split  # noqa: E402
from vlm.clients import ClaudeCliClient, OllamaClient  # noqa: E402
from vlm.parse import parse_reply  # noqa: E402
from vlm.prompts import PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
OUT = ROOT / "outputs" / "vlm"
DIFFS = ["easy", "medium", "hard"]


def items(split_name: str, n: int, start: int = 0) -> list[tuple]:
    """各難易度の start から n 枚: [(難易度, 画像パス, ラベル)]"""
    out = []
    for d in DIFFS:
        for line in open(EVAL / split_name / d / "labels.jsonl", encoding="utf-8").readlines()[start:start + n]:
            lab = json.loads(line)
            out.append((d, EVAL / split_name / d / "images" / lab["image"], lab))
    return out


def make_client(spec: str):
    kind, model = spec.split(":", 1)
    return OllamaClient(model) if kind == "ollama" else ClaudeCliClient(model)


def raw_path(split_name: str, spec: str, prompt: str) -> Path:
    return OUT / "raw" / f"{split_name}__{spec.replace(':', '_').replace('/', '_')}__{prompt}.jsonl"


def ask(a):
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    its = items(a.split, a.n, a.start)
    for spec in a.models:
        client = make_client(spec)
        workers = 1 if spec.startswith("ollama") else a.workers          # ローカルのモデルは1つずつ（メモリ・GPUの都合）
        for pr in a.prompts:
            f = raw_path(a.split, spec, pr)
            done = {json.loads(l)["key"] for l in open(f, encoding="utf-8")} if f.exists() else set()
            todo = [(d, p, lab) for d, p, lab in its if f"{d}/{lab['image']}" not in done]
            print(f"{spec} / {pr}: 残り {len(todo)}枚（済み {len(done)}）", flush=True)

            def one(t):
                d, p, lab = t
                for attempt in range(3):
                    try:
                        reply, meta = client.generate(PROMPTS[pr], p)
                        return dict(key=f"{d}/{lab['image']}", difficulty=d, image=lab["image"], reply=reply, **meta)
                    except Exception as e:                                  # 一時的な失敗は再試行
                        err = str(e)[:200]
                        time.sleep(3)
                return dict(key=f"{d}/{lab['image']}", difficulty=d, image=lab["image"], reply="", error=err, wall_s=0.0, in_tok=0, out_tok=0, usd=0.0)

            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=workers) as ex, open(f, "a", encoding="utf-8") as fh:
                for k, r in enumerate(ex.map(one, todo), 1):
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
                    if k % 20 == 0:
                        print(f"  {k}/{len(todo)} ({time.monotonic() - t0:.0f}秒)", flush=True)


def load_vlm(split_name: str, spec: str, pr: str, its) -> pd.DataFrame:
    f = raw_path(split_name, spec, pr)
    raw = {r["key"]: r for r in map(json.loads, open(f, encoding="utf-8"))} if f.exists() else {}
    rows = []
    for d, p, lab in its:
        r = raw.get(f"{d}/{lab['image']}")
        if r is None:
            continue
        pr_ = parse_reply(r["reply"])
        gt = lab["text"].replace(" ", "")
        rows.append(dict(system=f"{spec} [{pr}]", difficulty=d, image=lab["image"], gt=gt, pred=pr_["text"], conf=pr_["confidence"], parsed=pr_["parsed"],
                         wall_s=r.get("wall_s", 0.0), in_tok=r.get("in_tok", 0), out_tok=r.get("out_tok", 0), usd=r.get("usd", 0.0), error=bool(r.get("error"))))
    return pd.DataFrame(rows)


def ours(split_name: str, n: int, start: int = 0) -> pd.DataFrame:
    """自作のシステム（検出ResNet18＋CRNN3シード＋書式検査・確信度。Phase6の最終構成）を、同じ画像で評価する"""
    import torch
    import system_eval as se
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    det, ocrs = se.load_models(device)
    taus = se.thresholds([0, 1, 2])
    its = items(split_name, n, start)
    t0 = time.monotonic()
    df = se.evaluate_system(its, det, ocrs, taus, device)
    per_img = (time.monotonic() - t0) / len(its)
    rows = []
    for s in (0, 1, 2):
        rows.append(pd.DataFrame(dict(system=f"自作（検出ResNet18＋CRNN）seed{s}", difficulty=df.difficulty, image=df.image, gt=df["gt"], pred=df[f"p3_text_s{s}"],
                                      conf=df[f"conf_s{s}"], parsed=True, wall_s=per_img, in_tok=0, out_tok=0, usd=0.0, error=False, decision=df[f"decision_s{s}"])))
    return pd.concat(rows, ignore_index=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """割合の95%信頼区間（ウィルソン区間）。標本が少ないとき、差が誤差の範囲かを判断するのに使う"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def metrics(df: pd.DataFrame) -> dict:
    g = lambda x: dict(
        n=len(x), 完全一致=float((x.pred == x["gt"]).mean()), CER=float(np.mean([cer(p, t) if p else 1.0 for p, t in zip(x.pred, x["gt"])])),
        書式OK=float(np.mean([is_valid(p) for p in x.pred])), 解析OK=float(x.parsed.mean()),
        地名=float(np.mean([split(p)["region"] == split(t)["region"] for p, t in zip(x.pred, x["gt"])])),
        分類番号=float(np.mean([split(p)["class_no"] == split(t)["class_no"] for p, t in zip(x.pred, x["gt"])])),
        かな=float(np.mean([split(p)["kana"] == split(t)["kana"] for p, t in zip(x.pred, x["gt"])])),
        一連番号=float(np.mean([split(p)["serial"] == split(t)["serial"] for p, t in zip(x.pred, x["gt"])])))
    return g(df)


def report(a):
    its = items(a.split, a.n, a.start)
    frames = []
    for spec in a.models:
        for pr in a.prompts:
            d = load_vlm(a.split, spec, pr, its)
            if len(d):
                frames.append(d)
    vlm = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    mine = ours(a.split, a.n, a.start)
    allx = pd.concat([mine, vlm], ignore_index=True)
    allx.to_csv(OUT / f"compare_{a.split}_{a.name}.csv", index=False)

    L = [f"# Phase7: VLM（学習なし）と自作システムの比較（{a.split}・各難易度{a.n}枚。合成画像のみ）", ""]
    # ① 精度
    systems = list(dict.fromkeys(list(vlm.system.unique()) if len(vlm) else []))
    mine_sys = [s for s in mine.system.unique()]
    L += ["## ① 精度（完全一致）", "", "| システム | Easy | Medium | Hard | 全体 | 書式OK率 | 解析OK率 |", "|---|---|---|---|---|---|---|"]
    mm = mine.assign(sys_="自作（検出ResNet18＋CRNN）3シード平均")
    row = {d: np.mean([(mine[(mine.system == s) & (mine.difficulty == d)].pred == mine[(mine.system == s) & (mine.difficulty == d)]["gt"]).mean() for s in mine_sys]) for d in DIFFS}
    L.append(f"| **自作（検出ResNet18＋CRNN）3シード平均** | {row['easy']:.3f} | {row['medium']:.3f} | {row['hard']:.3f} | {np.mean(list(row.values())):.3f} | "
             f"{np.mean([metrics(mine[mine.system == s])['書式OK'] for s in mine_sys]):.3f} | 1.000 |")
    for s in systems:
        x = vlm[vlm.system == s]
        per = {d: metrics(x[x.difficulty == d]) if (x.difficulty == d).any() else None for d in DIFFS}
        L.append(f"| {s} | " + " | ".join(f"{per[d]['完全一致']:.3f}（{per[d]['n']}枚）" if per[d] else "—" for d in DIFFS) + f" | {np.mean([per[d]['完全一致'] for d in DIFFS if per[d]]):.3f} | {metrics(x)['書式OK']:.3f} | {metrics(x)['解析OK']:.3f} |")
    # ② フィールド別・CER
    L += ["", "## ② フィールド別の正解率・CER（全難易度込み）", "", "| システム | 地名 | 分類番号 | かな | 一連番号 | CER |", "|---|---|---|---|---|---|"]
    ms = [(f"自作 seed{s[-1]}", metrics(mine[mine.system == s])) for s in mine_sys]
    avg = {k: np.mean([m[k] for _, m in ms]) for k in ("地名", "分類番号", "かな", "一連番号", "CER")}
    L.append(f"| **自作（3シード平均）** | {avg['地名']:.3f} | {avg['分類番号']:.3f} | {avg['かな']:.3f} | {avg['一連番号']:.3f} | {avg['CER']:.3f} |")
    for s in systems:
        m = metrics(vlm[vlm.system == s])
        L.append(f"| {s} | {m['地名']:.3f} | {m['分類番号']:.3f} | {m['かな']:.3f} | {m['一連番号']:.3f} | {m['CER']:.3f} |")
    # ③ 速度・コスト・プライバシー
    L += ["", "## ③ 速度・コスト・プライバシー", "", "| システム | 1枚あたりの時間(秒) | 入力トークン/枚 | 出力トークン/枚 | 1,000枚あたりの金額（報告値） | 画像の外部送信 |", "|---|---|---|---|---|---|"]
    L.append(f"| **自作（3シード分まとめて）** | {mine.wall_s.iloc[0] * 1000:.0f}ミリ秒 | — | — | 0円（自分のMacで動く） | **なし** |")
    for s in systems:
        x = vlm[vlm.system == s]
        ext = "**あり（Anthropicへ）**" if s.startswith("claude") else "なし（ローカル）"
        L.append(f"| {s} | {x.wall_s.mean():.1f} | {x.in_tok.mean():,.0f} | {x.out_tok.mean():,.0f} | {x.usd.mean() * 1000:,.0f}ドル相当 | {ext} |")
    # ④ 自信の使いやすさ
    L += ["", "## ④ 「自信」で、正解と誤りを分けられるか（AUROC。0.5=偶然、1=完全。書式が正しいものだけ・全難易度込み）", "", "| システム | AUROC | 自信が0.9以上の割合 | 自信0.9以上の中の誤り率 |", "|---|---|---|---|"]
    for s in mine_sys:
        x = mine[(mine.system == s) & [is_valid(p) for p in mine[mine.system == s].pred]] if False else mine[mine.system == s]
        x = x[[is_valid(p) for p in x.pred]]
        a_ = x[x.decision == "自動処理"] if "decision" in x else x
        L.append(f"| 自作 seed{s[-1]}（確信度=seq） | {auroc(x.conf, x.pred == x['gt']):.3f} | 自動処理 {len(a_) / len(mine[mine.system == s]):.3f} | 自動処理の誤り率 {(a_.pred != a_['gt']).mean():.3f} |")
    for s in systems:
        x = vlm[(vlm.system == s) & vlm.conf.notna()]
        x = x[[is_valid(p) for p in x.pred]]
        hi = x[x.conf >= 0.9]
        L.append(f"| {s}（自己申告） | {auroc(x.conf, x.pred == x['gt']):.3f} | {len(hi) / max(1, len(x)):.3f} | {(hi.pred != hi['gt']).mean() if len(hi) else float('nan'):.3f} |")
    # ⑤ 信頼区間（全体・難易度別）: 差が誤差の範囲かを判断する
    L += ["", "## ⑤ 完全一致の95%信頼区間（ウィルソン区間。標本が小さいので、差は誤差の範囲のことがある）", "", "| システム | Easy | Medium | Hard |", "|---|---|---|---|"]
    def ci_row(label, frame):
        cells = []
        for d in DIFFS:
            x = frame[frame.difficulty == d]
            k, n_ = int((x.pred == x["gt"]).sum()), len(x)
            lo, hi = wilson(k, n_)
            cells.append(f"{k / max(n_, 1):.3f} [{lo:.2f}〜{hi:.2f}]（n={n_}）" if n_ else "—")
        return f"| {label} | " + " | ".join(cells) + " |"
    L.append(ci_row("自作（seed0。3シードの代表）", mine[mine.system == mine_sys[0]]))
    for s_ in systems:
        L.append(ci_row(s_, vlm[vlm.system == s_]))
    # ⑥ プロンプトの効果（同じ画像で比べる）
    L += ["", "## ⑥ 書式を教えるかどうか（同じ画像での比較。素の質問と同じ枚数に、書式つきを絞る）", "", "| モデル | 画像 | 素の質問 | 書式つき |", "|---|---|---|---|"]
    for m_ in ("ollama:qwen3.8:27b-mlx", "claude:sonnet"):
        nv, fm = vlm[vlm.system == f"{m_} [naive]"], vlm[vlm.system == f"{m_} [format]"]
        if len(nv) and len(fm):
            keys = set(zip(nv.difficulty, nv.image))
            fm2 = fm[[(d, i) in keys for d, i in zip(fm.difficulty, fm.image)]]
            L.append(f"| {m_} | {len(nv)}枚 | {(nv.pred == nv['gt']).mean():.3f}（書式OK {np.mean([is_valid(p) for p in nv.pred]):.3f}） | {(fm2.pred == fm2['gt']).mean():.3f}（書式OK {np.mean([is_valid(p) for p in fm2.pred]):.3f}） |")
    # ⑦ 自信の使い方（難易度別）: 固定のしきい値0.9（調整しない）
    L += ["", "## ⑦ 自信の使い方（難易度別。VLMは自己申告0.9以上、自作はPhase4の「自動処理」）", "", "| システム | 難易度 | 平均の自信 | 自動処理（自信が高い）の割合 | その中の誤り率 |", "|---|---|---|---|---|"]
    for s_ in systems:
        for d in DIFFS:
            x = vlm[(vlm.system == s_) & (vlm.difficulty == d) & vlm.conf.notna()]
            hi = x[(x.conf >= 0.9).to_numpy() & np.array([is_valid(p) for p in x.pred], dtype=bool)]
            if len(x):
                L.append(f"| {s_} | {d} | {x.conf.mean():.2f} | {len(hi) / len(x):.3f} | {(hi.pred != hi['gt']).mean() if len(hi) else float('nan'):.3f} |")
    for d in DIFFS:
        x = mine[mine.difficulty == d]
        auto = x[x.decision == "自動処理"]
        L.append(f"| 自作（3シード込み・seq確信度） | {d} | {x.conf.mean():.2f} | {len(auto) / len(x):.3f} | {(auto.pred != auto['gt']).mean() if len(auto) else float('nan'):.3f} |")
    # ⑧ 併用（探索的）: 自作が自動処理と判断したものはそのまま。それ以外は、VLMの自信が0.9以上なら、VLMの答えを採用
    L += ["", "## ⑧ 併用の試み（探索的。自作が自動処理→そのまま／それ以外でVLMの自信≥0.9かつ書式OK→VLMの答え／それ以外は人へ。しきい値0.9は調整していない）", "",
          "※ test2 を見ながら考えた方式なので、**参考値**（最終的な性能の保証ではない）。", "", "| 方式 | 難易度の混ぜ方 | 自動処理率 | 自動処理の誤り率 |", "|---|---|---|---|"]
    for v_name in ("claude:sonnet [format]", "ollama:qwen3.8:27b-mlx [format]"):
        v = vlm[vlm.system == v_name].set_index(["difficulty", "image"])
        for mixname, mix in {"均等": {"easy": 1 / 3, "medium": 1 / 3, "hard": 1 / 3}, "昼間中心": {"easy": 0.6, "medium": 0.3, "hard": 0.1}}.items():
            res = {"only": [], "hyb": []}
            for s_ in mine_sys:
                m_ = mine[mine.system == s_].copy()
                m_["w"] = m_.difficulty.map(lambda d: mix[d] / (m_.difficulty == d).sum())
                vv = v.reindex(list(zip(m_.difficulty, m_.image)))
                vok = (vv.conf.fillna(0).to_numpy() >= 0.9) & np.array([is_valid(p) if isinstance(p, str) else False for p in vv.pred])
                ours_auto = (m_.decision == "自動処理").to_numpy()
                pred_h = np.where(ours_auto, m_.pred, vv.pred.fillna("").to_numpy())
                auto_h = ours_auto | vok
                ok_o, ok_h = (m_.pred == m_["gt"]).to_numpy(), pred_h == m_["gt"].to_numpy()
                w = m_.w.to_numpy()
                for key, auto, ok in (("only", ours_auto, ok_o), ("hyb", auto_h, ok_h)):
                    res[key].append((w[auto].sum() / w.sum(), w[auto & ~ok].sum() / max(w[auto].sum(), 1e-9)))
            for key, lab in (("only", "自作だけ"), ("hyb", f"自作＋{v_name.split(' ')[0]}")):
                r_ = np.mean(res[key], axis=0)
                L.append(f"| {lab} | {mixname} | {r_[0]:.3f} | {r_[1]:.4f} |")
    (OUT / f"{a.name}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ask", "report"])
    ap.add_argument("--split", default="val", choices=["val", "test2"])
    ap.add_argument("--n", type=int, default=5, help="各難易度の枚数")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=["ollama:qwen3.8:27b-mlx"])
    ap.add_argument("--prompts", nargs="+", default=["format"], choices=list(PROMPTS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="summary", help="出力するmdのファイル名（拡張子なし）")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (ask if a.cmd == "ask" else report)(a)


if __name__ == "__main__":
    main()
