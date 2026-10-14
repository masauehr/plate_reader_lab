"""Phase2: 学習する検出器（板の四隅を回帰するCNN）。

model.py … モデル（自作の小さなCNN／ImageNet学習済みResNet18）
data.py … 学習データ（エポックごとにその場で生成）と、固定の評価データ
evaluate.py … IoU・Det@0.9・四隅の誤差・読み取り（Phase1のOCR）まで含む評価
"""
