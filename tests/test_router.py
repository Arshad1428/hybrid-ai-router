"""Router tests use fakes only: zero Ollama, zero AWS, zero cost."""
import pytest

from hybrid_router.config import Settings
from hybrid_router.router import HybridRouter


class Fake:
    def __init__(self, text="ok", fail=False):
        self.text, self.fail, self.calls = text, fail, 0

    def generate(self, prompt):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return self.text


def make(enabled=True, default="local", limit=5, cloud_fail=False):
    s = Settings(default_provider=default, aws_cloud_enabled=enabled, max_cloud_calls_per_session=limit)
    local, cloud = Fake("local-answer"), Fake("cloud-answer", fail=cloud_fail)
    return HybridRouter(s, local, cloud), local, cloud


def test_local_default_never_calls_cloud():
    r, local, cloud = make()
    res = r.generate("hi")
    assert res.provider == "local" and not res.fallback_used and cloud.calls == 0


def test_cloud_route_uses_bedrock():
    r, local, cloud = make()
    res = r.generate("hi", "cloud")
    assert res.provider == "bedrock" and res.response == "cloud-answer" and local.calls == 0


def test_cloud_disabled_blocks_bedrock():
    r, local, cloud = make(enabled=False)
    res = r.generate("hi", "cloud")
    assert res.provider == "local" and res.fallback_reason == "cloud_disabled" and cloud.calls == 0


def test_fallback_on_bedrock_failure():
    r, local, cloud = make(cloud_fail=True)
    res = r.generate("hi", "cloud")
    assert res.provider == "local" and res.fallback_used and res.fallback_reason.startswith("bedrock_error")


def test_cloud_call_guard():
    r, local, cloud = make(limit=2)
    assert r.generate("a", "cloud").provider == "bedrock"
    assert r.generate("b", "cloud").provider == "bedrock"
    res = r.generate("c", "cloud")
    assert res.provider == "local" and res.fallback_reason == "cloud_call_limit_reached"
    assert cloud.calls == 2


def test_failed_attempts_count_toward_guard():
    r, _, cloud = make(limit=1, cloud_fail=True)
    r.generate("a", "cloud")
    res = r.generate("b", "cloud")
    assert res.fallback_reason == "cloud_call_limit_reached" and cloud.calls == 1


def test_invalid_route():
    r, _, _ = make()
    with pytest.raises(ValueError):
        r.generate("x", "bogus")
