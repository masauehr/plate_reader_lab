#!/bin/bash
# Phase3: OCR（CRNN+CTC）の本実験。40エポック×4,500枚（毎回新しく生成）、バッチ32（約10分/回、計約60分）
# 使い方: bash src/run_phase3.sh   ※ 先に Phase2（bash src/run_phase2.sh）で検出器を学習しておく
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# (1) 標準: 全難易度で学習、四隅に誤差（σ≦3px）を足す。3シード
for s in 0 1 2; do $PY src/train_ocr.py --seed $s; done
# (2) 四隅の誤差を足さない（正解の四隅だけで学習）→ 検出器の誤差に弱くなるか
$PY src/train_ocr.py --seed 0 --corner_sigma 0
# (3) 四隅の誤差を大きく足す（σ≦8px）→ 足しすぎは逆効果か
$PY src/train_ocr.py --seed 0 --corner_sigma 8
# (4) 易しい条件(Easy)だけで学習 → 難しい条件で評価（分布シフト）
$PY src/train_ocr.py --seed 0 --train_diffs easy
echo "PHASE3_DONE"
