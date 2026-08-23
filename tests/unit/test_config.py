import pytest

from mlx_moe_stream.config import parse_bytes, parse_resident_budget


@pytest.mark.parametrize(
    ("value", "expected"),
    [("32GB", 32_000_000_000), ("32GiB", 32 * (1 << 30)), (1024, 1024)],
)
def test_parse_bytes(value, expected):
    assert parse_bytes(value) == expected


@pytest.mark.parametrize("value", ["auto", "0GB", "-1GB", "1PB"])
def test_parse_bytes_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        parse_bytes(value)


def test_parse_resident_budget_distinguishes_auto_from_no_cache():
    assert parse_resident_budget(None) == (None, False)
    assert parse_resident_budget("auto") == (None, True)
    assert parse_resident_budget("AUTO") == (None, True)
    assert parse_resident_budget("2GB") == (2_000_000_000, False)
