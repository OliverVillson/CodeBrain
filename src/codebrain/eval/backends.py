"""Model backends: embedders and an LLM client.

Real runs use sentence-transformers for embeddings (in process, on the GPU) and any
OpenAI-compatible server for the LLM (vLLM, SGLang). ``--fake`` swaps both for
deterministic stand-ins so the whole pipeline can be smoke-tested on a laptop or in CI.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from codebrain.eval.lexical import split_identifiers

Vectors = npt.NDArray[np.float32]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _normalize(v: Vectors) -> Vectors:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vectors = (v / norms).astype(np.float32)
    return out


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str], *, query: bool) -> Vectors: ...


class HashEmbedder:
    """Hashed bag of split identifiers. Deterministic, no model, for smoke tests only."""

    def __init__(self, dim: int = 512) -> None:
        self.name = f"fake-hash-{dim}"
        self.dim = dim

    def embed(self, texts: Sequence[str], *, query: bool) -> Vectors:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in split_identifiers(t):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0
        return _normalize(out)


class STEmbedder:
    """A sentence-transformers model. Uses the model's ``query`` prompt when it has one."""

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 64,
        max_seq_length: int = 2048,
        trust_remote_code: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self.batch_size = batch_size
        self.model: Any = SentenceTransformer(
            model, trust_remote_code=trust_remote_code, model_kwargs={"torch_dtype": "bfloat16"}
        )
        self.model.max_seq_length = min(self.model.max_seq_length or max_seq_length, max_seq_length)
        prompts = getattr(self.model, "prompts", None) or {}
        self.query_prompt = "query" if "query" in prompts else None

    def embed(self, texts: Sequence[str], *, query: bool) -> Vectors:
        kwargs: dict[str, Any] = {"batch_size": self.batch_size, "normalize_embeddings": True}
        if query and self.query_prompt:
            kwargs["prompt_name"] = self.query_prompt
        # Longest first keeps padding low inside each batch.
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
        vecs = np.asarray(self.model.encode([texts[i] for i in order], **kwargs), np.float32)
        out = np.empty_like(vecs)
        out[order] = vecs
        return out


class CachedEmbedder:
    """Embeds each distinct text once. Keys are caller-chosen content hashes."""

    def __init__(self, inner: Embedder) -> None:
        self.inner = inner
        self.name = inner.name
        self.cache: dict[str, npt.NDArray[np.float32]] = {}
        self.seconds = 0.0
        self.count = 0

    def ensure(self, items: Sequence[tuple[str, str]]) -> None:
        """Embed ``(key, text)`` pairs that are not cached yet, in one batch."""
        todo: dict[str, str] = {}
        for key, text in items:
            if key not in self.cache:
                todo[key] = text
        if not todo:
            return
        t0 = time.perf_counter()
        vecs = self.inner.embed(list(todo.values()), query=False)
        self.seconds += time.perf_counter() - t0
        self.count += len(todo)
        for key, v in zip(todo, vecs, strict=True):
            self.cache[key] = v

    def matrix(self, keys: Sequence[str]) -> Vectors:
        out: Vectors = (
            np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, 1), np.float32)
        )
        return out

    def query(self, texts: Sequence[str]) -> Vectors:
        return self.inner.embed(texts, query=True)


class LLM(Protocol):
    name: str
    input_tokens: int
    output_tokens: int

    async def complete(self, system: str, user: str, max_tokens: int) -> str: ...

    def count_tokens(self, text: str) -> int: ...


class _TokenCounter:
    def __init__(self, tokenizer: str | None) -> None:
        self.tok: Any = None
        if tokenizer:
            try:
                from transformers import AutoTokenizer

                self.tok = AutoTokenizer.from_pretrained(tokenizer)
            except Exception as e:  # tokenizer is a nicety; fall back to a char estimate
                print(f"[warn] tokenizer {tokenizer!r} unavailable ({e}); estimating chars/4")

    def __call__(self, text: str) -> int:
        if self.tok is None:
            return max(1, len(text) // 4)
        return len(self.tok.encode(text, add_special_tokens=False))


class OpenAILLM:
    """Chat completions against an OpenAI-compatible server, with bounded concurrency."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str = "EMPTY",
        concurrency: int = 128,
        disable_thinking: bool = True,
        tokenizer: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self.name = model
        self.client: Any = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=600)
        self.sem = asyncio.Semaphore(concurrency)
        self.extra: dict[str, Any] = (
            {"chat_template_kwargs": {"enable_thinking": False}} if disable_thinking else {}
        )
        self.count_tokens = _TokenCounter(tokenizer or model)
        self.input_tokens = 0
        self.output_tokens = 0

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        async with self.sem:
            for attempt in range(4):
                try:
                    r = await self.client.chat.completions.create(
                        model=self.name,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        max_tokens=max_tokens,
                        temperature=0.0,
                        extra_body=self.extra,
                    )
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    await asyncio.sleep(2**attempt)
        if r.usage is not None:
            self.input_tokens += r.usage.prompt_tokens
            self.output_tokens += r.usage.completion_tokens
        text = r.choices[0].message.content or ""
        return _THINK.sub("", text).strip()


class FakeLLM:
    """Deterministic stand-in: summaries are the first lines, answers echo matching lines."""

    def __init__(self) -> None:
        self.name = "fake-llm"
        self.input_tokens = 0
        self.output_tokens = 0
        self.count_tokens = _TokenCounter(None)

    async def complete(self, system: str, user: str, max_tokens: int) -> str:
        self.input_tokens += self.count_tokens(system + user)
        if "Question:" in user:
            context, question = user.rsplit("Question:", 1)
            words = set(split_identifiers(question))
            best = max(
                context.splitlines() or [""],
                key=lambda line: len(words.intersection(split_identifiers(line))),
            )
            out = best
        else:
            out = " ".join(user.split()[: max_tokens // 2])
        self.output_tokens += self.count_tokens(out)
        return out
