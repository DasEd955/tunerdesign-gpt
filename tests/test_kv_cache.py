"""test_kv_cache.py - Unit tests for model/kv_cache.py.

Tests the KVCache accumulator (initialization, incremental update, clear)
and CachedAttention (output shape, cache state propagation, rounding).
"""

import torch
import pytest
from model.kv_cache import KVCache, CachedAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cache():
    """Return a fresh KVCache instance."""
    return KVCache()


@pytest.fixture
def ca():
    """Return a CachedAttention module with model_dim=8."""
    return CachedAttention(model_dim=8)


@pytest.fixture
def step_input():
    """A (1, 3, 8) input tensor representing a 3-token step."""
    torch.manual_seed(11)
    return torch.randn(1, 3, 8)


# ---------------------------------------------------------------------------
# KVCache
# ---------------------------------------------------------------------------

class TestKVCache:
    """Tests for KVCache (model/kv_cache.py)."""

    def test_initial_cache_is_none(self, cache):
        """A new KVCache has no stored keys or values."""
        assert cache.cache_k is None
        assert cache.cache_v is None

    def test_first_update_initializes_cache(self, cache):
        """The first update call sets cache_k and cache_v to the given tensors."""
        k = torch.ones(1, 3, 8)
        v = torch.ones(1, 3, 8)
        cache.update(k, v)
        assert cache.cache_k is not None
        assert cache.cache_v is not None

    def test_first_update_stores_exact_tensors(self, cache):
        """After the first update, the cached tensors equal the inputs."""
        k = torch.randn(1, 3, 8)
        v = torch.randn(1, 3, 8)
        ck, cv = cache.update(k, v)
        assert torch.equal(ck, k)
        assert torch.equal(cv, v)

    def test_second_update_concatenates(self, cache):
        """Subsequent updates concatenate along the sequence dimension (dim=1)."""
        k1 = torch.ones(1, 2, 8)
        v1 = torch.ones(1, 2, 8)
        cache.update(k1, v1)

        k2 = torch.zeros(1, 3, 8)
        v2 = torch.zeros(1, 3, 8)
        ck, cv = cache.update(k2, v2)
        assert ck.shape == (1, 5, 8)
        assert cv.shape == (1, 5, 8)

    def test_accumulated_content_is_correct(self, cache):
        """The accumulated cache contains the first batch followed by the second."""
        k1 = torch.ones(1, 2, 8) * 1.0
        k2 = torch.ones(1, 3, 8) * 2.0
        ck, _ = cache.update(k1, torch.zeros_like(k1))
        ck, _ = cache.update(k2, torch.zeros_like(k2))
        assert torch.all(ck[:, :2, :] == 1.0)
        assert torch.all(ck[:, 2:, :] == 2.0)

    def test_clear_resets_cache(self, cache):
        """clear() sets both cache tensors back to None."""
        k = torch.randn(1, 3, 8)
        cache.update(k, k)
        cache.clear()
        assert cache.cache_k is None
        assert cache.cache_v is None

    def test_clear_then_update_reinitializes(self, cache):
        """After clear(), the next update starts fresh."""
        k1 = torch.ones(1, 4, 8)
        cache.update(k1, k1)
        cache.clear()
        k2 = torch.zeros(1, 2, 8)
        ck, _ = cache.update(k2, k2)
        assert ck.shape == (1, 2, 8)

    def test_multiple_steps_accumulate_correctly(self, cache):
        """Three sequential updates produce the correct accumulated sequence length."""
        for step in range(3):
            k = torch.randn(1, 1, 8)
            ck, _ = cache.update(k, k)
        assert ck.shape == (1, 3, 8)

    def test_returns_tuple(self, cache):
        """update() returns a 2-tuple of tensors."""
        result = cache.update(torch.zeros(1, 1, 8), torch.zeros(1, 1, 8))
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# CachedAttention
# ---------------------------------------------------------------------------

class TestCachedAttention:
    """Tests for CachedAttention.forward (model/kv_cache.py)."""

    def test_output_shape(self, ca, step_input):
        """Output tensor has the same shape as the input."""
        out, _ = ca(step_input)
        assert out.shape == step_input.shape

    def test_returns_tensor_and_cache(self, ca, step_input):
        """forward returns a (Tensor, KVCache) 2-tuple."""
        result = ca(step_input)
        assert isinstance(result, tuple)
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], KVCache)

    def test_creates_new_cache_when_none_provided(self, ca, step_input):
        """Passing kv_cache=None creates a new KVCache internally."""
        _, cache = ca(step_input, kv_cache=None)
        assert cache.cache_k is not None

    def test_cache_grows_across_steps(self, ca):
        """Passing the cache from step N to step N+1 extends the stored history."""
        x1 = torch.randn(1, 2, 8)
        _, cache = ca(x1, kv_cache=None)
        x2 = torch.randn(1, 1, 8)
        _, cache = ca(x2, kv_cache=cache)
        assert cache.cache_k.shape[1] == 3

    def test_rounded_to_four_decimals(self, ca, step_input):
        """All output values are rounded to 4 decimal places."""
        out, _ = ca(step_input)
        assert torch.all(out == torch.round(out, decimals=4))

    def test_reusing_cache_changes_output(self, ca):
        """Using a populated cache gives a different output than starting fresh."""
        torch.manual_seed(0)
        x = torch.randn(1, 2, 8)
        _, cache = ca(x, kv_cache=None)
        x_next = torch.randn(1, 1, 8)
        out_cached, _ = ca(x_next, kv_cache=cache)

        fresh_cache = KVCache()
        out_fresh, _ = ca(x_next, kv_cache=fresh_cache)
        assert not torch.allclose(out_cached, out_fresh, atol=1e-3)
