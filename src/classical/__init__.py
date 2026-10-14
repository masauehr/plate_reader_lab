"""古典的な画像処理による、ナンバー読み取りのベースライン（機械学習を使わない）。

detect.py … 色と形で板を探す  / ocr.py … テンプレートマッチで文字を読む / pipeline.py … 両者をつなぐ
"""
from .detect import Detection, detect_plate
from .ocr import OcrResult, read_text, rectify
from .pipeline import ClassicalResult, read_plate

__all__ = ["Detection", "detect_plate", "OcrResult", "read_text", "rectify", "ClassicalResult", "read_plate"]
