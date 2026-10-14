#!/bin/bash
# Phase2: 検出器の本実験。20エポック×4,500枚（毎回新しく生成）、3シード
# 使い方: bash src/run_phase2.sh   （約50分）
cd "$(dirname "$0")/.."
PY=.venv/bin/python
# (1) モデルの比較: 自作の小さなCNN と ImageNet学習済みResNet18（転移学習）
for s in 0 1 2; do
    $PY src/train_detector.py --model small --epochs 20 --seed $s
    $PY src/train_detector.py --model resnet18 --epochs 20 --seed $s
done
# (2) 学習データの難易度: 易しい条件(Easy)だけで学習 → 難しい条件で評価（分布シフト）
for s in 0 1 2; do
    $PY src/train_detector.py --model small --epochs 20 --seed $s --train_diffs easy
done
echo "PHASE2_DONE"
