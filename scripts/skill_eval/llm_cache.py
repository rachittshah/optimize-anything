"""Disk-cached, cost-tracked LLM calls through the engine's LLM layer.

Cache key = sha256(model, effort, system, prompt), so a different SKILL.md in `system` never
collides. Errors and empty replies are never cached. Hits cost nothing and are counted per role.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from optimize_anything.core.cost import CostTracker, TrackedLLM
from optimize_anything.llm import LLMClient, LLMError, LLMResponse


class LockedCostTracker(CostTracker):
    """CostTracker that is safe to record into from worker threads."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self.cache_hits: dict[str, int] = {}
        self.quota_error: str | None = None  # set once a call fails on a usage/rate limit

    def record(self, role: str, response: LLMResponse) -> None:
        with self._lock:
            super().record(role, response)

    def hit(self, role: str) -> None:
        with self._lock:
            self.cache_hits[role] = self.cache_hits.get(role, 0) + 1

    def to_dict(self) -> dict:
        with self._lock:
            d = super().to_dict()
            d["cache_hits"] = dict(self.cache_hits)
            return d


QUOTA_RE = re.compile(r"hit your .{0,20}limit|usage limit|rate[ _-]?limit|quota|429", re.I)


def ensure_ignored(path: Path) -> None:
    """Self-ignoring cache: drop a `*` .gitignore at the .cache root (or `path` if it is one)."""
    root = next((p for p in (path, *path.parents) if p.name == ".cache"), path)
    root.mkdir(parents=True, exist_ok=True)
    ignore = root / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n")


def cache_key(model: str, effort: str | None, system: str | None, prompt: str) -> str:
    raw = json.dumps([model, effort, system, prompt], ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()


class CachedLLM:
    """LLMClient for one role: disk cache in front of TrackedLLM(llm, tracker, role)."""

    def __init__(self, llm: LLMClient, tracker: LockedCostTracker, role: str,
                 cache_dir: Path | None):
        self.tracked = TrackedLLM(llm, tracker, role)
        self.tracker = tracker
        self.role = role
        self.cache_dir = cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            ensure_ignored(cache_dir)

    def _path(self, key: str) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / key[:2] / f"{key}.json"

    def complete(self, prompt: str, *, system: str | None = None, model: str,
                 max_tokens: int, effort: str | None = None) -> LLMResponse:
        key = cache_key(model, effort, system, prompt)
        path = self._path(key)
        if path is not None and path.exists():
            try:
                data = json.loads(path.read_text())
                self.tracker.hit(self.role)
                return LLMResponse(text=data["text"], model=model, backend="cache")
            except (ValueError, KeyError):
                pass  # corrupt entry: fall through and overwrite
        try:
            resp = self.tracked.complete(prompt, system=system, model=model,
                                         max_tokens=max_tokens, effort=effort)
        except LLMError as e:
            if QUOTA_RE.search(str(e)) and self.tracker.quota_error is None:
                self.tracker.quota_error = str(e)[:300]
            raise
        if path is not None and resp.text.strip():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps({
                "text": resp.text, "model": model, "effort": effort, "role": self.role,
                "cost_usd": resp.cost_usd, "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens, "created": time.time(),
            }))
            tmp.replace(path)
        return resp
