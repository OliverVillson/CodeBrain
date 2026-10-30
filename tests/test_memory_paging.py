from collections.abc import Iterable

import pytest

from codebrain.config import PagingParams
from codebrain.memory import (
    EpisodeRecorder,
    PinKind,
    Recalled,
    Role,
    SessionState,
    SessionTracker,
)
from codebrain.models import Namespace, UnitKind
from tests.memory_fakes import BagOfWordsEmbedder, FakeStore

W = 1000


@pytest.fixture
def store() -> FakeStore:
    return FakeStore()


def tracker(store: FakeStore, **kw: object) -> SessionTracker:
    params = kw.pop("params", PagingParams())
    return SessionTracker(
        "s1",
        W,
        store=store,
        embedder=BagOfWordsEmbedder(),
        repo="demo",
        system_prompt="You are a coding agent.",
        params=params,  # type: ignore[arg-type]
        token_counter=lambda text: len(text.split()),
        **kw,  # type: ignore[arg-type]
    )


def turn_units(store: FakeStore) -> list:  # type: ignore[type-arg]
    return [u for u in store.units.values() if u.kind is UnitKind.TURN]


def test_no_paging_below_high_water(store: FakeStore) -> None:
    t = tracker(store)
    for i in range(6):
        assert t.add_turn(Role.USER, f"message {i}", tokens=100) is None
    assert t.usage < 0.70 and store.units == {}


def test_crossing_high_water_pages_oldest_turns_down_to_low_water(store: FakeStore) -> None:
    t = tracker(store)
    t.start_task("Build a rate limiter for the public API")
    t.pin("Never touch the billing module", PinKind.CONSTRAINT)
    events = [
        t.add_turn(Role.USER if i % 2 else Role.ASSISTANT, f"turn body {i}", tokens=100)
        for i in range(7)
    ]
    *quiet, event = events
    assert all(e is None for e in quiet)
    assert event is not None

    assert event.tokens_before > 0.70 * W
    assert event.tokens_after <= 0.50 * W and not event.over_target
    assert event.evicted_turns == [1, 2, 3]
    assert [t_.index for t_ in t.state.turns] == [4, 5, 6, 7]
    # Pinned facts stay in the window.
    assert {p.kind for p in t.state.pinned} == {PinKind.TASK, PinKind.CONSTRAINT}

    units = turn_units(store)
    assert units and all(u.namespace is Namespace.SESSION and u.session_id == "s1" for u in units)
    # Nothing is summarized away: the full text of every evicted turn is stored.
    stored = "\n".join(u.text for u in units)
    for i in range(3):
        assert f"turn body {i}" in stored
    assert all((u.id, "summary", "fake-bow-64") in store.vectors for u in units)

    ctx = t.context()
    assert [m.source for m in ctx][:3] == ["system", "pinned", "paged_index"]
    assert units[0].id in ctx[2].content
    assert [m.content for m in ctx if m.source == "turn"][0] == "turn body 3"


def test_turn_units_respect_size_cap(store: FakeStore) -> None:
    t = tracker(store, params=PagingParams(turn_unit_max_tokens=150))
    for i in range(8):
        t.add_turn(Role.USER, f"t{i}", tokens=100)
    units = turn_units(store)
    assert len(units) >= 2
    assert all(u.tokens <= 150 for u in units)
    spans = [(u.labels["first_turn"], u.labels["last_turn"]) for u in units]
    assert spans == sorted(spans)


def test_oversized_turn_is_kept_whole(store: FakeStore) -> None:
    t = tracker(store, params=PagingParams(turn_unit_max_tokens=100, keep_recent_turns=1))
    t.add_turn(Role.TOOL, "huge tool output", tokens=600)
    t.add_turn(Role.USER, "next", tokens=200)
    [u] = turn_units(store)
    assert u.tokens == 600 and "huge tool output" in u.text


def test_recent_turns_and_pins_are_never_evicted(store: FakeStore) -> None:
    t = tracker(store, params=PagingParams(keep_recent_turns=2))
    t.pin("a very large pinned spec", PinKind.FACT)
    t.state.pinned[0].tokens = 650
    t.add_turn(Role.USER, "one", tokens=50)
    event = t.add_turn(Role.ASSISTANT, "two", tokens=50)
    assert event is None or not event.evicted_turns
    event = t.add_turn(Role.USER, "three", tokens=50)
    assert event is not None and event.over_target
    assert event.evicted_turns == [1]
    assert [x.index for x in t.state.turns] == [2, 3]
    assert len(t.state.pinned) == 1


def test_injected_units_are_dropped_before_turns(store: FakeStore) -> None:
    t = tracker(store)
    t.add_turn(Role.USER, "hello", tokens=300)
    t.inject("u_old_episode", "an old episode", tokens=150)
    event = t.inject("u_other", "another unit", tokens=300)
    assert event is not None
    assert event.dropped_injections == ["u_old_episode", "u_other"]
    assert event.evicted_turns == []
    assert store.units == {}


def test_prefetch_query_and_injection(store: FakeStore) -> None:
    t = tracker(store)
    t.start_task("Build a rate limiter")
    assert t.prefetch_query() == "Build a rate limiter"
    t.add_turn(Role.USER, "make it per tenant", tokens=10)
    t.set_plan_step("write TokenBucket")
    assert t.prefetch_query() == "make it per tenant\nwrite TokenBucket"

    seen: list[tuple[str, int]] = []

    def recall(query: str, budget: int) -> Iterable[Recalled]:
        seen.append((query, budget))
        return [
            Recalled(unit_id="u_a", text="token bucket code", tokens=100),
            Recalled(unit_id="u_big", text="too big", tokens=10_000),
            Recalled(unit_id="u_b", text="webhook episode"),
        ]

    free = t.free_tokens
    assert t.prefetch(recall) == ["u_a", "u_b"]
    assert seen[0] == ("make it per tenant\nwrite TokenBucket", free)
    assert t.prefetch(recall) == []  # already injected
    sources = [m.source for m in t.context()]
    assert sources.count("memory") == 2


def test_force_page(store: FakeStore) -> None:
    t = tracker(store, params=PagingParams(keep_recent_turns=1))
    for i in range(3):
        t.add_turn(Role.USER, f"m{i}", tokens=200)
    event = t.page(force=True)
    assert event is not None and event.evicted_turns == [1]


def test_state_roundtrip(store: FakeStore) -> None:
    t = tracker(store)
    t.start_task("Build a rate limiter")
    for i in range(8):
        t.add_turn(Role.USER, f"m{i}", tokens=100)
    state = SessionState.model_validate_json(t.state.model_dump_json())
    t2 = SessionTracker.from_state(
        state, store=store, embedder=BagOfWordsEmbedder(), token_counter=lambda s: len(s.split())
    )
    assert t2.tokens == t.tokens
    t2.add_turn(Role.USER, "next", tokens=10)
    assert t2.state.next_turn == 10


def test_finish_task_writes_episode_and_unpins(store: FakeStore) -> None:
    t = tracker(store)
    t.start_task("Build a rate limiter", plan="1. bucket 2. middleware")
    t.pin("Use Python 3.11", PinKind.CONSTRAINT)
    t.add_turn(Role.USER, "go", tokens=10)
    result = t.finish_task(EpisodeRecorder(store, BagOfWordsEmbedder()), tests_passed=True)

    ep = store.units[result.episode_id]
    assert ep.session_id == "s1" and ep.repo == "demo"
    [evidence] = store.episodes
    assert evidence.prompt == "Build a rate limiter" and evidence.plan == "1. bucket 2. middleware"
    assert evidence.started_at is not None
    assert [p.kind for p in t.state.pinned] == [PinKind.CONSTRAINT]
    assert t.state.task_started_at is None
