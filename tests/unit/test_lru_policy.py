from mlx_moe_stream.cache.policy import ExpertKey, LruCacheSimulator, simulate_lru_curve


def test_global_lru_evicts_least_recent_expert_by_byte_budget():
    cache = LruCacheSimulator(capacity_bytes=4)
    a, b, c = ExpertKey(0, 0), ExpertKey(0, 1), ExpertKey(1, 0)

    assert cache.access(a, nbytes=2) is False
    assert cache.access(b, nbytes=2) is False
    assert cache.access(a, nbytes=2) is True
    assert cache.access(c, nbytes=2) is False
    assert cache.access(b, nbytes=2) is False

    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 4
    assert stats.evictions == 2
    assert stats.resident_bytes == 4


def test_oversized_bundle_is_not_admitted():
    cache = LruCacheSimulator(capacity_bytes=3)
    key = ExpertKey(0, 0)
    assert cache.access(key, nbytes=4) is False
    assert cache.access(key, nbytes=4) is False
    assert cache.stats().resident_bytes == 0


def test_curve_uses_full_working_set_at_one_hundred_percent():
    keys = [ExpertKey(0, 0), ExpertKey(0, 1), ExpertKey(0, 0), ExpertKey(0, 1)]
    curve = simulate_lru_curve(keys, capacities=(0.5, 1.0))
    assert curve[0]["capacity_experts"] == 1
    assert curve[1]["capacity_experts"] == 2
    assert curve[1]["hit_rate"] == 0.5

