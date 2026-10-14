"""Phase7（VLM比較）のうち、外部サービスを使わない部分（返事の解析・正規化・安全装置）のテスト。"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vlm.clients import OllamaClient  # noqa: E402
from vlm.parse import normalize, parse_reply  # noqa: E402
from vlm.prompts import PROMPTS  # noqa: E402


class TestParse(unittest.TestCase):
    def test_json_in_code_fence(self):
        r = parse_reply('```json\n{"text": "沖縄 300 あ 12-34", "confidence": 0.9}\n```')
        self.assertEqual((r["text"], r["confidence"], r["parsed"]), ("沖縄300あ12-34", 0.9, True))

    def test_integer_confidence_and_extra_text(self):
        r = parse_reply('読み取り結果です。{"text": "品川 500 か ・・12", "confidence": 1} 以上')
        self.assertEqual((r["text"], r["confidence"]), ("品川500か・・12", 1.0))

    def test_normalize_variants(self):
        self.assertEqual(normalize("品川 ５００ か １２‐３４"), "品川500か12-34")           # 全角数字・ハイフンの表記ゆれ
        self.assertEqual(normalize("沖縄 300 あ ･･12"), "沖縄300あ・・12")                 # 中点の表記ゆれ
        self.assertEqual(normalize(" 大阪 1 2 "), "大阪12")

    def test_normalize_does_not_change_content(self):
        """表記ゆれだけをそろえ、読み取りの内容は変えない（6と8などは、そのまま）"""
        self.assertEqual(normalize("沖縄 380 あ"), "沖縄380あ")
        self.assertNotEqual(normalize("沖縄 380 あ"), normalize("沖縄 360 あ"))

    def test_unparsable_falls_back_to_first_line(self):
        r = parse_reply("沖縄 300 あ 12-34\n(自信はありません)")
        self.assertEqual((r["text"], r["confidence"], r["parsed"]), ("沖縄300あ12-34", None, False))
        self.assertEqual(parse_reply("")["text"], "")

    def test_broken_json(self):
        r = parse_reply('{"text": "沖縄 300 あ 12-34", "confidence": }')
        self.assertFalse(r["parsed"])


class TestSafety(unittest.TestCase):
    def test_ollama_is_local_only(self):
        """ローカル以外のホストには、画像を送らない"""
        OllamaClient("m")                                              # 既定（127.0.0.1）は通る
        with self.assertRaises(AssertionError):
            OllamaClient("m", host="http://example.com:11434")

    def test_prompts(self):
        self.assertIn("合成画像", PROMPTS["format"])                   # 合成画像であることを明記
        for p in PROMPTS.values():
            self.assertIn('"text"', p); self.assertIn("confidence", p)
        self.assertIn("沖縄", PROMPTS["format"]); self.assertNotIn("沖縄・那覇", PROMPTS["naive"])   # 素の質問には、書式の知識を入れない


if __name__ == "__main__":
    unittest.main()
