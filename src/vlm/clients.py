"""VLMの呼び出し。どちらも、1枚の画像と質問を渡し、返事の文字列とメタ情報（時間・トークン）を返す。

OllamaClient  … ローカル（127.0.0.1）。画像は外部に出ない。課金なし。
ClaudeCliClient … Claude Code CLI（サブスク経由）。画像は Anthropic に送信される（合成画像のみ送る）。
"""
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

CLAUDE_BIN = os.path.expanduser("~/.local/bin/claude")


class OllamaClient:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434"):
        assert host.startswith("http://127.0.0.1") or host.startswith("http://localhost"), "ローカル以外には送らない"
        self.model, self.host, self.kind = model, host, "local"

    def generate(self, prompt: str, image: Path) -> tuple[str, dict]:
        body = {"model": self.model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 300},
                "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(Path(image).read_bytes()).decode()]}]}
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
        return d["message"]["content"], dict(wall_s=time.monotonic() - t0, in_tok=d.get("prompt_eval_count", 0), out_tok=d.get("eval_count", 0), usd=0.0)


class ClaudeCliClient:
    def __init__(self, model: str):
        self.model, self.kind = model, "external"
        self._cwd = tempfile.mkdtemp(prefix="vlm_cli_")                  # プロジェクトのCLAUDE.mdなどを読み込まない、空のディレクトリ

    def generate(self, prompt: str, image: Path) -> tuple[str, dict]:
        image = Path(image).resolve()
        env = os.environ.copy()
        env.pop("ANTHROPIC_API_KEY", None)                                # サブスク認証を使う（APIの課金を発生させない）
        env.pop("ANTHROPIC_BASE_URL", None)
        full = f"{prompt}\n\n画像ファイル: {image}\nReadツールでこの画像を開いて読み取ってください。"
        cmd = [CLAUDE_BIN, "--print", "--model", self.model, "--output-format", "json", "--tools", "Read", "--allowedTools", "Read",
               "--add-dir", str(image.parent), "--no-session-persistence", "--disable-slash-commands", "--strict-mcp-config"]
        t0 = time.monotonic()
        p = subprocess.run(cmd, input=full, text=True, capture_output=True, timeout=600, env=env, cwd=self._cwd)
        wall = time.monotonic() - t0
        if p.returncode != 0:
            raise RuntimeError(f"claude CLI 終了コード {p.returncode}: {(p.stderr or p.stdout)[:300]}")
        d = json.loads(p.stdout)
        u = d.get("usage") or {}
        tok_in = int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0))
        return d.get("result") or "", dict(wall_s=wall, in_tok=tok_in, out_tok=int(u.get("output_tokens", 0)), usd=float(d.get("total_cost_usd") or 0.0))
