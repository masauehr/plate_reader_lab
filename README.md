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

**Phase0（合成データ生成器）完了。** 以降は [PLAN.md](PLAN.md) の Phase1（非AIのベースライン）から。

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests     # 生成器のテスト（9件）
.venv/bin/python src/make_eval_sets.py             # 評価セット4,500枚を data/eval/ に生成（約45秒）
.venv/bin/jupyter lab notebooks/                   # 解説付きNotebook（カーネル: Python (plate_reader_lab)）
```

```python
from plate_synth import generate_sample
s = generate_sample(seed=0, difficulty="medium")   # easy / medium / hard
s.image    # シーン画像（PIL, 320×240）
s.label    # 正解ラベル（板の位置・文字・劣化の強さなど）
```

| Notebook | 内容 |
|---|---|
| [00_synth_data](notebooks/00_synth_data.ipynb) | Phase0: 合成データを「作る」。プレート4種、正解ラベル、背景、劣化を1つずつ、難易度3段階、再現性、データ分割 |

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
| [PLAN.md](PLAN.md) | フェーズ計画・進捗・未決事項 |
| [MANUAL.md](MANUAL.md) | 設計（プレートの簡略仕様、パイプラインの詳細、各工程の手法と判断） |
| [DATA_SPEC.md](DATA_SPEC.md) | 合成データの仕様（文字・色・背景・劣化・難易度・正解ラベル） |
| [EVALUATION.md](EVALUATION.md) | 評価指標の定義と、検証の進め方 |
| [CLAUDE.md](CLAUDE.md) | Claude Code向けプロジェクト固有ルール |

## ディレクトリ構成（予定）

```
plate_reader_lab/
├── README.md / CLAUDE.md / PLAN.md / MANUAL.md / DATA_SPEC.md / EVALUATION.md
├── plate-reader-lab.md   # pc_docs運用マニュアルのコピー
├── requirements.txt
├── data/        # 生成した評価セット（git管理外。`make_eval_sets.py` で再生成できる）
├── outputs/     # 学習曲線・評価結果・失敗例（git管理外）
├── notebooks/   # 解説付き学習Notebook（フェーズごと）
├── src/
│   ├── plate_synth/      # 合成データ生成器（Phase0 完成）
│   ├── make_eval_sets.py # 固定シードの評価セットを生成
│   └── （detect / ocr / rules / parking_sim / eval は今後）
└── tests/       # 生成器のテスト
```

## 関連プロジェクト

- [ai_core](../ai_core/README.md) — AI全体の地図。事例分解は [CASE_STUDIES.md](../ai_core/CASE_STUDIES.md)、このプロジェクトの位置づけは [PLAN.md](../ai_core/PLAN.md)
- [image_augment_lab](../image_augment_lab/README.md) — 画像のデータ拡張・転移学習。**拡張の知見を応用**する（ただし**左右反転は文字が鏡像になるため禁止**。「ラベルを壊す拡張は使えない」の実例）。次のステップ「予測・検証システム化」とも合流する
