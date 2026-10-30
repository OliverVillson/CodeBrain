import pytest
from pydantic import ValidationError

from codebrain.config import Settings
from codebrain.models import (
    Edge,
    EdgeType,
    Namespace,
    Resolution,
    Span,
    Unit,
    UnitKind,
    code_unit_id,
)


def test_code_unit_id_is_stable_and_distinct() -> None:
    a = code_unit_id("repo", "pkg/bucket.py", "TokenBucket.consume")
    assert a == code_unit_id("repo", "pkg/bucket.py", "TokenBucket.consume")
    assert a != code_unit_id("repo", "pkg/bucket.py", "TokenBucket.refill")
    assert a.startswith("u_")


def test_quality_prior() -> None:
    u = Unit(kind=UnitKind.EPISODE, namespace=Namespace.PROJECT, text="x")
    assert u.quality == 0.5
    u = u.model_copy(update={"retrieved_count": 8, "used_count": 3})
    assert u.quality == pytest.approx(0.4)


def test_code_units_need_path() -> None:
    with pytest.raises(ValidationError):
        Unit(kind=UnitKind.FUNCTION, text="def f(): ...")


def test_session_units_need_session_id() -> None:
    with pytest.raises(ValidationError):
        Unit(kind=UnitKind.TURN, namespace=Namespace.SESSION, text="turns 14-22")


def test_span_order() -> None:
    with pytest.raises(ValidationError):
        Span(start_line=10, end_line=3)


def test_edge_default_weights() -> None:
    assert Edge(src_id="a", dst_id="b", type=EdgeType.CALLS).effective_weight == 1.0
    assert Edge(src_id="a", dst_id="b", type=EdgeType.IMPORTS, weight=0.2).effective_weight == 0.2


def test_resolution_value() -> None:
    assert [r.value_factor for r in Resolution] == [0.4, 0.7, 1.0]
    # Recall packs with the tuned factors, which favour code over prose.
    r = Settings().retrieval
    assert (r.value_signature, r.value_summary) == (0.15, 0.35)


def test_settings_defaults_match_sketch() -> None:
    s = Settings()
    w = s.retrieval
    assert w.w_semantic + w.w_code + w.w_lexical + w.w_graph == pytest.approx(0.75)
    assert 0 < w.quality_floor < 1 and 0 < w.freshness_floor < 1
    assert (s.chunking.tau_min, s.chunking.tau_max) == (64, 1200)
    assert s.paging.high_water == 0.70
