"""Local stand-in for the Anthropic and OpenAI HTTP APIs, used only by formulation runs.

There are no API keys on this machine, so LLM-calling evaluators would always fail. The stub
answers `/v1/messages`, `/v1/chat/completions` and `/v1/responses` with a short canned reply, so
validity checks whether the evaluator code runs, not model quality.

Never put the stub env into this process's os.environ: concurrent `claude -p` calls would inherit
ANTHROPIC_API_KEY=stub. Pass `stub_env()` to the formulation subprocess only.
"""
from __future__ import annotations

import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# JSON first (extraction evaluators parse `{...}`), then a judge-style score line.
CANNED_TEXT = '{"answer": "ok", "label": "ok"}\nSCORE: 5/10'

# Loaded automatically by every Python child whose PYTHONPATH includes its directory.
# Rewrites hardcoded api.anthropic.com / api.openai.com urllib calls to the stub.
SITECUSTOMIZE = '''\
import os as _os
_stub = _os.environ.get("OA_STUB_URL")
if _stub:
    import urllib.parse as _up
    import urllib.request as _ur
    _HOSTS = {"api.anthropic.com", "api.openai.com"}
    _orig_open = _ur.OpenerDirector.open

    def _rewrite(url):
        p = _up.urlsplit(url)
        if p.hostname in _HOSTS:
            s = _up.urlsplit(_stub)
            return _up.urlunsplit((s.scheme, s.netloc, p.path, p.query, p.fragment))
        return url

    def _open(self, fullurl, *args, **kwargs):
        if isinstance(fullurl, str):
            fullurl = _rewrite(fullurl)
        else:
            new = _rewrite(fullurl.full_url)
            if new != fullurl.full_url:
                fullurl.full_url = new
                fullurl.headers.pop("Host", None)
        return _orig_open(self, fullurl, *args, **kwargs)

    _ur.OpenerDirector.open = _open
'''


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # silence
        pass

    def _send(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("content-length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            req = {}
        model = req.get("model", "stub-model")
        path = self.path.split("?")[0].rstrip("/")
        usage = {"input_tokens": 10, "output_tokens": 10}
        if path.endswith("/messages"):
            self._send(200, {"id": "msg_stub", "type": "message", "role": "assistant", "model": model,
                             "content": [{"type": "text", "text": CANNED_TEXT}],
                             "stop_reason": "end_turn", "stop_sequence": None, "usage": usage})
        elif path.endswith("/chat/completions"):
            self._send(200, {"id": "chatcmpl-stub", "object": "chat.completion", "created": 0,
                             "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": CANNED_TEXT}}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}})
        elif path.endswith("/responses"):
            self._send(200, {"id": "resp_stub", "object": "response", "status": "completed",
                             "model": model, "output_text": CANNED_TEXT,
                             "output": [{"type": "message", "role": "assistant", "content": [
                                 {"type": "output_text", "text": CANNED_TEXT, "annotations": []}]}],
                             "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}})
        else:
            self._send(404, {"error": {"type": "not_found", "message": f"stub: no route {path}"}})

    def do_GET(self) -> None:  # noqa: N802
        self._send(404, {"error": {"type": "not_found", "message": "stub: GET not supported"}})


class StubServer:
    """Background-thread stub. Use as a context manager or call start()/stop()."""

    def __init__(self, site_dir: Path):
        self.site_dir = site_dir
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None, "stub not started"
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> StubServer:
        if self._server is None:
            self.site_dir.mkdir(parents=True, exist_ok=True)
            if any(p.name == ".cache" for p in self.site_dir.parents):
                from .llm_cache import ensure_ignored
                ensure_ignored(self.site_dir)
            (self.site_dir / "sitecustomize.py").write_text(SITECUSTOMIZE)
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            self._server.daemon_threads = True
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> StubServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def env(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Environment for a formulation subprocess: stub keys + base URLs + sitecustomize."""
        env = dict(os.environ if base is None else base)
        pp = env.get("PYTHONPATH", "")
        env.update({
            "OA_STUB_URL": self.url,
            "ANTHROPIC_BASE_URL": self.url,
            "OPENAI_BASE_URL": f"{self.url}/v1",
            "OA_OPENAI_BASE_URL": f"{self.url}/v1",
            "ANTHROPIC_API_KEY": "stub",
            "OPENAI_API_KEY": "stub",
            "OA_BACKEND": "mock",
            "PYTHONPATH": str(self.site_dir) + (os.pathsep + pp if pp else ""),
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        return env
