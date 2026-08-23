import mlx_moe_stream.routing as routing


def test_loader_keeps_remote_code_disabled_when_supported():
    recorded = {}

    def fake_load(_path, *, trust_remote_code):
        recorded["trust_remote_code"] = trust_remote_code
        return "model", "tokenizer"

    assert routing._load_model_without_remote_code(fake_load, "model") == ("model", "tokenizer")
    assert recorded["trust_remote_code"] is False


def test_loader_supports_mlx_lm_without_the_newer_keyword():
    def fake_load(_path):
        return "model", "tokenizer"

    assert routing._load_model_without_remote_code(fake_load, "model") == ("model", "tokenizer")
