from __future__ import annotations

import threading
from types import SimpleNamespace

from mlx_moe_stream.cache import ExpertKey
from mlx_moe_stream.prefetch import AsyncExpertLoader


class _Reader:
    def __init__(self, *, block_first: bool = False) -> None:
        self.calls: list[ExpertKey] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block_first = block_first

    def read_bundle(self, bundle: SimpleNamespace) -> dict[str, bytes]:
        self.calls.append(bundle.key)
        if self.block_first and len(self.calls) == 1:
            self.started.set()
            assert self.release.wait(timeout=2)
        return {"weight": bytes([bundle.key.expert])}


def _bundle(key: ExpertKey) -> SimpleNamespace:
    return SimpleNamespace(key=key)


def test_prefetch_and_demand_coalesce_to_one_exact_read():
    key = ExpertKey(0, 1)
    reader = _Reader()
    loader = AsyncExpertLoader(reader, workers=1, max_inflight=2)
    try:
        assert loader.prefetch(key, _bundle(key))
        assert loader.demand(key, _bundle(key)) == {"weight": b"\x01"}
        stats = loader.stats()
    finally:
        loader.close()

    assert reader.calls == [key]
    assert stats.prefetch_submitted == 1
    assert stats.prefetch_hits == 1
    assert stats.coalesced_requests == 1


def test_demand_priority_runs_ahead_of_queued_prefetch():
    first = ExpertKey(0, 0)
    low_priority = ExpertKey(0, 1)
    demand_key = ExpertKey(0, 2)
    reader = _Reader(block_first=True)
    loader = AsyncExpertLoader(reader, workers=1, max_inflight=3)
    result: list[dict[str, bytes]] = []
    try:
        assert loader.prefetch(first, _bundle(first))
        assert reader.started.wait(timeout=2)
        assert loader.prefetch(low_priority, _bundle(low_priority))
        thread = threading.Thread(
            target=lambda: result.append(loader.demand(demand_key, _bundle(demand_key)))
        )
        thread.start()
        reader.release.set()
        thread.join(timeout=2)
        assert not thread.is_alive()
    finally:
        loader.close()

    assert result == [{"weight": b"\x02"}]
    assert reader.calls[:2] == [first, demand_key]


def test_predictive_prefetch_tracks_hits_and_unused_reads():
    hit = ExpertKey(0, 1)
    unused = ExpertKey(0, 2)
    reader = _Reader()
    loader = AsyncExpertLoader(reader, workers=1, max_inflight=2)
    try:
        assert loader.prefetch(hit, _bundle(hit), origin="predictive")
        assert loader.demand(hit, _bundle(hit)) == {"weight": b"\x01"}
        assert loader.prefetch(unused, _bundle(unused), origin="predictive")
    finally:
        loader.close()
    stats = loader.stats()

    assert stats.predictive_requests == 2
    assert stats.predictive_submitted == 2
    assert stats.predictive_hits == 1
    assert stats.predictive_unused == 1
