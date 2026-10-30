import math
from types import SimpleNamespace
from typing import Any

import pytest

from codebrain.ingest.embed import HashingEmbedder, make_embedder, register_embedder
from codebrain.ingest.summarize import (
    AnthropicSummarizer,
    HeuristicSummarizer,
    LLMSummarizer,
    SummaryRequest,
    build_prompt,
    leading_doc,
    make_summarizer,
    register_summarizer,
)
from codebrain.models import UnitKind


def req(**kw: Any) -> SummaryRequest:
    base: dict[str, Any] = dict(
        kind=UnitKind.FUNCTION,
        qualified_name="retry_with_backoff",
        path="net/retry.py",
        language="python",
        text=(
            "def retry_with_backoff(fn):\n"
            '    """Retry fn with exponential backoff. Gives up after 5."""\n'
        ),
        signature="def retry_with_backoff(fn)",
    )
    base.update(kw)
    return SummaryRequest(**base)


def test_leading_doc() -> None:
    assert leading_doc('def f():\n    """Do the thing. More."""') == "Do the thing."
    assert (
        leading_doc("// Clamp a value.\n// Second line.\nfunction clamp() {}") == "Clamp a value."
    )
    assert leading_doc("#!/usr/bin/env python\nx = 1") is None


def test_heuristic_summary() -> None:
    s = HeuristicSummarizer().one(req())
    assert s == (
        "Function retry_with_backoff in net/retry.py. Retry fn with exponential backoff. "
        "Signature: def retry_with_backoff(fn)."
    )
    module = HeuristicSummarizer().one(
        req(
            kind=UnitKind.MODULE,
            qualified_name="net/retry.py",
            children=(("retry_with_backoff", "x"),),
        )
    )
    assert module.startswith("File net/retry.py.") and module.endswith(
        "Contains retry_with_backoff."
    )


def test_prompt_includes_child_summaries() -> None:
    p = build_prompt(req(kind=UnitKind.CLASS, children=(("A.f", "Does f."),)))
    assert "- A.f: Does f." in p and "```" in p


def test_llm_summarizer_isolates_failures() -> None:
    def complete(system: str, prompt: str) -> str:
        if "boom" in prompt:
            raise RuntimeError("API down")
        return "  Retries   a call.\n"

    s = LLMSummarizer(complete, name="fake", concurrency=2)
    assert s.summarize([req(), req(text="boom")]) == ["Retries a call.", None]


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return self.response


def fake_client(stop_reason: str = "end_turn") -> Any:
    response = SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Retries a call with backoff."),
        ],
    )
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(response)))


def test_anthropic_summarizer_request_shape() -> None:
    client = fake_client()
    s = AnthropicSummarizer(client=client, concurrency=1)
    assert s.name == "anthropic:claude-opus-5-5"
    assert s.summarize([req()]) == ["Retries a call with backoff."]
    [call] = client.beta.messages.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert call["fallbacks"] == "default"
    assert call["messages"][0]["role"] == "user"
    assert "retry_with_backoff" in call["messages"][0]["content"]


def test_anthropic_refusal_gives_no_summary() -> None:
    s = AnthropicSummarizer(client=fake_client("refusal"), concurrency=1)
    assert s.summarize([req()]) == [None]


def test_make_summarizer() -> None:
    assert isinstance(make_summarizer("heuristic"), HeuristicSummarizer)
    with pytest.raises(ValueError):
        make_summarizer("nope")


def test_hashing_embedder() -> None:
    e = HashingEmbedder(256)
    a, b, c = e.embed(
        [
            "def consume_tokens(bucket, n): bucket.tokens -= n",
            "def consumeTokens(bucket, n) { bucket.tokens -= n }",
            "SELECT * FROM invoices WHERE paid = false",
        ]
    )
    assert len(a) == 256 and math.isclose(sum(x * x for x in a), 1.0)
    assert e.embed(["def consume_tokens(bucket, n): bucket.tokens -= n"])[0] == a

    def cos(x: list[float], y: list[float]) -> float:
        return sum(p * q for p, q in zip(x, y, strict=True))

    assert cos(a, b) > 0.8 > cos(a, c)


def test_make_embedder() -> None:
    assert make_embedder("unset-text-model", 128).model == "hashing-v1-128"
    assert make_embedder("hashing:32").model == "hashing-v1-32"
    register_embedder("fixed", lambda arg: HashingEmbedder(int(arg)))
    assert make_embedder("fixed:16").model == "hashing-v1-16"
    with pytest.raises(ValueError):
        make_embedder("mystery:model")


def test_leading_doc_must_be_the_first_statement() -> None:
    assert leading_doc('x = 1\n\ndef g():\n    """Nested doc."""\n') is None
    assert leading_doc('def f():\n    x = """literal"""\n') is None
    sig = (
        '@cache\ndef f(\n    a: int,\n    b: dict = {},\n) -> None:  # note\n    """Real doc."""\n'
    )
    assert leading_doc(sig) == "Real doc."
    assert leading_doc('#!/usr/bin/env python\n"""Module doc."""\nimport os\n') == "Module doc."


def test_registered_summarizers_get_options() -> None:
    got: dict[str, object] = {}

    def factory(arg: str, **options: object) -> HeuristicSummarizer:
        got.update(arg=arg, **options)
        return HeuristicSummarizer()

    register_summarizer("custom", factory)
    make_summarizer("custom:big", effort="high", max_tokens=99)
    assert got == {"arg": "big", "effort": "high", "max_tokens": 99}


def test_sentence_transformers_gets_batch_size(monkeypatch: pytest.MonkeyPatch) -> None:
    import codebrain.ingest.embed as embed

    class Fake:
        def __init__(self, name: str, batch_size: int = 64) -> None:
            self.model, self.batch_size = name, batch_size

    monkeypatch.setattr(embed, "SentenceTransformerEmbedder", Fake)
    e = make_embedder("sentence-transformers:m", batch_size=8)
    assert isinstance(e, Fake) and e.batch_size == 8
