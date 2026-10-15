# plate_reader_lab

**ナンバープレート風の合成画像**を読み取る、駐車場精算システムの**ミニ版**を作るプロジェクト。
「画像認識（AI）」「ルールによる補正」「確信度と人への引き継ぎ」「入出庫の照合・料金計算」「評価・検証」を、**1つのシステムとして**組み立てて学ぶ。

詳しくは [MANUAL.md](MANUAL.md) / [plate-reader-lab.md](plate-reader-lab.md)（pc_docs運用マニュアルのコピー）を参照。

> 🔒 **実在のナンバーや、実際の車の写真は使わない。** すべて**自分で生成した合成画像**で行う（個人情報・プライバシー配慮）。
> 詳しくは [CLAUDE.md](CLAUDE.md) の「倫理・プライバシー」を参照。

## なぜこのプロジェクトか

[ai_core](../ai_core/README.md) で、駐車場のナンバー読み取り精算システムを分解した。結論は次のとおり。

- AIが担うのは「**目**」（ナンバー板の検出と文字認識）だけ。
- 残りの大半は、**ルール補正・データベース・料金計算・失敗時の人への引き継ぎ**といった従来のソフトウェアと運用。
- つまり、**AIはシステムの一部品**。

このプロジェクトは、それを**自分の手で作って確かめる**ためのもの。あわせて、[ai_core](../ai_core/PROJECT_MAP.md) で「未経験」だった分野（**物体検出・OCR・予測の検証**）を学ぶ。

## 作るもの（パイプライン）

```
合成画像の生成 ─▶ ①ナンバー板の検出 ─▶ ②文字の読み取り(OCR) ─▶ ③書式チェック・補正
   ─▶ ④確信度で判定（自動処理 / 要確認） ─▶ ⑤入出庫の照合・料金計算 ─▶ ⑥評価・検証レポート
```

| 工程 | 内容 | 種類 | 学ぶこと |
|---|---|---|---|
| 合成画像の生成 | ナンバー画像と正解（位置・文字）を自動生成 | プログラム | **教師データを「生成」で増やす**（拡張の次の手段） |
| ① 検出 | 画像中のナンバー板の位置を見つける | 機械学習（物体検出） | 物体検出、IoU |
| ② OCR | 切り出した板から文字列を読む | 機械学習（CRNN＋CTC） | OCR、系列認識 |
| ③ 補正 | 書式ルールで誤読を直す・弾く | ルールベース | AIとルールの組み合わせ |
| ④ 判定 | 確信度が低ければ「要確認」にする | しきい値設計 | 確信度、見逃しと誤検出のトレードオフ |
| ⑤ 入出庫 | 入庫を記録し、出庫時に照合して課金 | 従来のソフトウェア（SQLite） | システム全体の設計 |
| ⑥ 評価 | 難易度別の精度、失敗分析、誤課金率 | 検証 | **予測・検証システム** |

## 現状と使い方

**Phase0（合成データ生成器）・Phase1（非AIのベースライン）・Phase2（学習する検出器）・Phase3（学習する文字認識）・Phase4（ルール補正・確信度・要確認）・Phase5（入出庫シミュレーション）・Phase6（評価・検証レポート、最終評価）完了。** **発展の Phase7（学習なしのVLMとの比較）も完了。** **最終レポートは [REPORT.md](REPORT.md)**。

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests     # 生成器・ベースライン・検出器・評価指標のテスト（74件）
.venv/bin/python src/make_eval_sets.py             # 評価セット4,500枚を data/eval/ に生成（約45秒）
.venv/bin/python src/eval_baseline.py             # 古典的ベースラインの評価（約20秒）→ outputs/baseline/
.venv/bin/python src/train_detector.py --model resnet18 --epochs 20   # 検出器の学習（約9分。run_phase2.shで本実験を一括）
.venv/bin/python src/train_ocr.py                                    # OCR(CRNN+CTC)の学習（約10分。run_phase3.shで本実験を一括。先に検出器が必要）
.venv/bin/python src/make_eval_sets.py --val 0 --test 0 --cal 1000   # 較正セット（しきい値決定用）を追加で作る
.venv/bin/python src/eval_phase4.py                  # 書式検査・確信度・振り分けの評価（約1分。先にPhase2・3の学習が必要）
.venv/bin/python src/run_phase7.py ask --split test2 --n 60 --models ollama:qwen3.8:27b-mlx --prompts format   # VLMに読ませる（合成画像のみ。claude: は画像が外部に送信される）
.venv/bin/python src/run_phase7.py report --split test2 --n 60 --models ollama:qwen3.8:27b-mlx claude:sonnet --prompts format   # 自作との比較 → outputs/vlm/
.venv/bin/python src/eval_report.py                    # 検証レポート（混同行列・劣化要因アブレーション・失敗例。約7分）→ outputs/report/
.venv/bin/python src/final_eval.py                     # 最終評価の予行演習（val）。test は --split test --confirm-final（1回だけ・実施済み）
.venv/bin/python src/run_phase5.py                     # 入出庫シミュレーション（約15分。先にPhase2〜4が必要）→ outputs/sim/
.venv/bin/jupyter lab notebooks/                   # 解説付きNotebook（カーネル: Python (plate_reader_lab)）
```

```python
from plate_synth import generate_sample
s = generate_sample(seed=0, difficulty="medium")   # easy / medium / hard
s.image    # シーン画像（PIL, 320×240）
s.label    # 正解ラベル（板の位置・文字・劣化の強さなど）
```

各Notebookの**実行結果（図・表・読み取れること・演習）のまとめ**は [notebooks/README.md](notebooks/README.md) にある（Notebookを開く前の予習・実行後の答え合わせに使える）。

| Notebook | 内容 |
|---|---|
| [00_synth_data](notebooks/00_synth_data.ipynb) | Phase0: 合成データを「作る」。プレート4種、正解ラベル、背景、劣化を1つずつ、難易度3段階、再現性、データ分割 |
| [01_classical_baseline](notebooks/01_classical_baseline.ipynb) | Phase1: 「AIなし」でどこまで読めるか。検出と文字認識を1段階ずつ見て、難易度別に測り、失敗の原因を分析する |
| [02_detection_cnn](notebooks/02_detection_cnn.ipynb) | Phase2: 「例から学ぶAI」で板の四隅を見つける。その場で生成する学習、古典的手法との比較、四隅誤差と読み取りの関係、転移学習、分布シフト、「板が無い」と言えない問題 |
| [03_ocr_crnn](notebooks/03_ocr_crnn.ipynb) | Phase3: 「例から読み方を学ぶAI」(CRNN+CTC)。板を1行にする、CTCの仕組みと「位置は対応しない」発見、四隅の誤差の混ぜ方、分布シフト、読み間違いの分析、確信度の予告、左右反転がダメな理由 |
| [04_rules_confidence](notebooks/04_rules_confidence.ipynb) | Phase4: AIは間違える前提で作る。書式検査（弾く）・補正（効果ほぼゼロ）・確信度4通り・自動処理／要確認／読み取り不可の振り分け・カバレッジ曲線・しきい値の決め方（データ量・コスト）・残る誤り |
| [05_parking_system](notebooks/05_parking_system.ipynb) | Phase5: AIを「システム」の一部品にする。SQLite・料金・照合方針4通り・似たナンバーの世界・誤課金と二次被害・人手の内訳・損益分岐（誤課金の重さ） |
| [06_final_report](notebooks/06_final_report.ipynb) | Phase6: プロジェクトの総括。最終評価(test)、混同行列、劣化要因アブレーション（運用できる範囲）、失敗例、学び、限界 |
| [07_vlm_comparison](notebooks/07_vlm_comparison.ipynb) | Phase7: 作るか借りるか。学習なしのVLM（ローカル・Claude）と自作を比較。書式を教える効果、「分からない」と言える自信、速度・コスト・プライバシー、併用 |

- カーネルが無い場合: `.venv/bin/python -m ipykernel install --user --name plate_reader_lab --display-name "Python (plate_reader_lab)"`
- 日本語フォントは macOS 標準のヒラギノ角ゴシックを使う（環境変数 `PLATE_FONT` で変更可）。

## 進め方

詳細は [PLAN.md](PLAN.md)。

0. 環境・合成データ生成器
1. 非AIのベースライン（古典的な画像処理）で「AIなしでどこまでできるか」を知る
2. 検出（物体検出）
3. 文字認識（OCR）
4. ルール補正・確信度・「要確認」
5. 入出庫シミュレーション
6. 評価・検証レポート
7. （発展）VLM（学習なしのAI）との比較

## ドキュメント

| ファイル | 内容 |
|---|---|
| **[REPORT.md](REPORT.md)** | **最終レポート**（最終評価の結果、主要な発見、運用できる範囲、限界、再現手順） |
| [PLAN.md](PLAN.md) | フェーズ計画・進捗・未決事項 |
| [MANUAL.md](MANUAL.md) | 設計（プレートの簡略仕様、パイプラインの詳細、各工程の手法と判断） |
| [DATA_SPEC.md](DATA_SPEC.md) | 合成データの仕様（文字・色・背景・劣化・難易度・正解ラベル） |
| [EVALUATION.md](EVALUATION.md) | 評価指標の定義と、検証の進め方 |
| [CLAUDE.md](CLAUDE.md) | Claude Code向けプロジェクト固有ルール |

## ディレクトリ構成（予定）

```
plate_reader_lab/
├── README.md / REPORT.md / CLAUDE.md / PLAN.md / MANUAL.md / DATA_SPEC.md / EVALUATION.md
├── report_images/       # 最終レポートの図
├── plate-reader-lab.md   # pc_docs運用マニュアルのコピー
├── requirements.txt
├── data/        # 生成した評価セット（git管理外。`make_eval_sets.py` で再生成できる）
├── outputs/     # 学習曲線・評価結果・失敗例（git管理外）
├── notebooks/   # 解説付き学習Notebook（フェーズごと）
├── src/
│   ├── plate_synth/      # 合成データ生成器（Phase0 完成）
│   ├── classical/        # 古典的ベースライン: detect（検出）/ ocr（文字認識）/ pipeline（Phase1 完成）
│   ├── ai_detect/        # 学習する検出器: model（小さなCNN/ResNet18）/ data（その場で生成）/ evaluate（Phase2 完成）
│   ├── train_detector.py # 検出器の学習 ／ run_phase2.sh（本実験の一括実行）
│   ├── ai_ocr/           # 学習する文字認識: model（CRNN+CTC）/ data（板を1行に・四隅の誤差を混ぜる）/ evaluate（Phase3 完成）
│   ├── train_ocr.py      # OCRの学習 ／ run_phase3.sh（本実験の一括実行）
│   ├── rules/            # 書式検査・補正・確信度（4通り）・振り分け・カバレッジ曲線（Phase4 完成）
│   ├── eval_phase4.py    # 振り分けの評価（しきい値の決め方・コスト・難易度の混ぜ方）
│   ├── parking_sim/      # 入出庫シミュレーション: 料金・SQLite・照合方針・世界・AIで読む事象（Phase5 完成）
│   ├── run_phase5.py     # 入出庫シミュレーションの本実験（4場面×4方針）
│   ├── vlm/              # VLM（学習なし）との比較: プロンプト・解析・クライアント（Phase7 完成。ローカルのみ／Claude CLI）
│   ├── run_phase7.py     # VLM比較の実行（ask=問い合わせ／report=集計）
│   ├── plate_eval/       # 評価指標（IoU・CER・完全一致など）
│   ├── eval_baseline.py  # ベースラインの評価（難易度別・失敗例の画像）
│   ├── make_eval_sets.py # 固定シードの評価セットを生成
│   └── （detect / ocr / rules / parking_sim / eval は今後）
└── tests/       # 生成器のテスト
```

## 関連プロジェクト

- [ai_core](../ai_core/README.md) — AI全体の地図。事例分解は [CASE_STUDIES.md](../ai_core/CASE_STUDIES.md)、このプロジェクトの位置づけは [PLAN.md](../ai_core/PLAN.md)
- [image_augment_lab](../image_augment_lab/README.md) — 画像のデータ拡張・転移学習。**拡張の知見を応用**する（ただし**左右反転は文字が鏡像になるため禁止**。「ラベルを壊す拡張は使えない」の実例）。次のステップ「予測・検証システム化」とも合流する

## Notebook を読むとき
はじめての方は、まず [notebooks/00_basics.ipynb](notebooks/00_basics.ipynb)（基礎知識・用語集・読む順）から。各Notebookの冒頭には「やさしい導入」があります。
