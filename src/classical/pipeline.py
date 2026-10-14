"""検出 → 補正 → 文字認識 をつなぐ。どこで失敗したか（検出か、読み取りか）が分かるように結果を返す。"""
from dataclasses import dataclass

from PIL import Image

from .detect import Detection, detect_plate
from .ocr import OcrResult, read_text, rectify


@dataclass
class ClassicalResult:
    detection: Detection | None
    ocr: OcrResult | None
    plate: Image.Image | None          # まっすぐにした板（確認用）

    @property
    def text(self) -> str:
        return self.ocr.text if self.ocr else ""


def read_plate(img: Image.Image) -> ClassicalResult:
    det = detect_plate(img)
    if det is None:
        return ClassicalResult(None, None, None)
    plate = rectify(img, det.corners)
    return ClassicalResult(det, read_text(plate), plate)
