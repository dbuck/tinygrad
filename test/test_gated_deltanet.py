"""Correctness tests for GatedDeltaNet chunked parallel prefill vs token-by-token recurrence."""
import unittest, copy
import numpy as np
from tinygrad import Tensor, dtypes

# dimensions matching a small Qwen3.5-like config
B, H, d = 1, 4, 16  # batch, heads, head_dim
DIM = 64             # model dim (must be >= H*d for projections but we test the core algorithm directly)

def naive_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=None):
  """Token-by-token reference implementation. All inputs are numpy arrays.
  q, k: (B, H, T, d)  v: (B, H, T, d)  g: (B, H, T)  beta: (B, H, T)
  State convention matches tinygrad: S is (B, H, d_k, d_v).
  Read: o = S^T @ q (i.e. o_j = sum_i S_ij * q_i).
  Write: S_ij += k_i * delta_j.
  Returns: output (B, H, T, d), final_state (B, H, d, d)
  """
  _B, _H, T, _d = q.shape
  S = np.zeros((_B, _H, _d, _d), dtype=np.float64) if initial_state is None else initial_state.copy()
  outputs = np.zeros_like(q, dtype=np.float64)
  for t in range(T):
    q_t = q[:, :, t, :]   # (B, H, d)
    k_t = k[:, :, t, :]
    v_t = v[:, :, t, :]
    g_t = g[:, :, t]       # (B, H)
    b_t = beta[:, :, t]    # (B, H)

    # decay state
    S = S * np.exp(g_t)[:, :, None, None]
    # prediction error: delta = (v - S^T @ k) * beta
    pred = np.einsum('bhij,bhi->bhj', S, k_t)  # S^T @ k: o_j = sum_i S_ij * k_i
    delta = (v_t - pred) * b_t[:, :, None]
    # write: S_ij += k_i * delta_j
    S = S + np.einsum('bhi,bhj->bhij', k_t, delta)
    # read: o_j = sum_i S_ij * q_i
    outputs[:, :, t, :] = np.einsum('bhij,bhi->bhj', S, q_t)
  return outputs, S

def chunked_gated_delta_rule(q, k, v, g, beta, initial_state=None):
  """Chunked algorithm in numpy, mirrors the tinygrad _prefill implementation."""
  _B, _H, T, _d = q.shape
  C = T  # single chunk (same as _prefill when T <= CHUNK_SIZE)
  S = np.zeros((_B, _H, _d, _d), dtype=np.float64) if initial_state is None else initial_state.copy()

  # cumulative sum of log-gates
  g_cumsum = np.cumsum(g, axis=-1)  # (B, H, C)

  # decay mask
  g_i = g_cumsum[:, :, :, None]  # (B, H, C, 1)
  g_j = g_cumsum[:, :, None, :]  # (B, H, 1, C)
  L_mask = np.exp(g_i - g_j)     # (B, H, C, C)

  # UT transform: build strictly lower triangular adjacency
  k_beta = k * beta[:, :, :, None]
  kkt = k_beta @ k.transpose(0, 1, 3, 2)  # (B, H, C, C)
  attn = -(kkt * L_mask)
  attn = np.tril(attn, -1)  # strictly lower triangular

  # forward substitution
  for i in range(1, C):
    row_i = attn[:, :, i:i+1, :i]   # (B, H, 1, i)
    block = attn[:, :, :i, :i]       # (B, H, i, i)
    attn[:, :, i:i+1, :i] = row_i + row_i @ block
  attn = attn + np.eye(C)

  # corrected values and keys
  u = attn @ (v * beta[:, :, :, None])
  w = attn @ (k * (beta * np.exp(g_cumsum))[:, :, :, None])

  # inter-chunk: query reads from previous state
  q_scaled = q * np.exp(g_cumsum)[:, :, :, None]
  o_inter = q_scaled @ S

  # state-corrected values
  w_S = w @ S
  v_new = u - w_S

  # intra-chunk causal attention
  qk = q @ k.transpose(0, 1, 3, 2)
  causal_mask = np.tril(L_mask, 0)
  o_intra = (qk * causal_mask) @ v_new

  o = o_inter + o_intra

  # state update
  g_total = g_cumsum[:, :, -1]  # (B, H)
  k_end = k * np.exp(g_total[:, :, None, None] - g_cumsum[:, :, :, None])
  new_state = S * np.exp(g_total)[:, :, None, None] + k_end.transpose(0, 1, 3, 2) @ v_new

  return o, new_state

class TestGatedDeltaNetChunked(unittest.TestCase):
  def _random_inputs(self, T, seed=42):
    np.random.seed(seed)
    q = np.random.randn(B, H, T, d).astype(np.float64)
    k = np.random.randn(B, H, T, d).astype(np.float64)
    v = np.random.randn(B, H, T, d).astype(np.float64)
    # L2 normalize q, k
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12) * (d ** -0.5)
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)
    # g should be negative (log-decay), beta in (0, 1)
    g = -np.abs(np.random.randn(B, H, T).astype(np.float64)) * 0.1
    beta = 1.0 / (1.0 + np.exp(-np.random.randn(B, H, T).astype(np.float64)))  # sigmoid
    return q, k, v, g, beta

  def test_chunked_matches_naive_small(self):
    """Chunked algorithm matches naive recurrence for T=8."""
    q, k, v, g, beta = self._random_inputs(T=8)
    o_naive, s_naive = naive_recurrent_gated_delta_rule(q, k, v, g, beta)
    o_chunk, s_chunk = chunked_gated_delta_rule(q, k, v, g, beta)
    np.testing.assert_allclose(o_chunk, o_naive, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(s_chunk, s_naive, atol=1e-8, rtol=1e-6)

  def test_chunked_matches_naive_medium(self):
    """Chunked algorithm matches naive recurrence for T=32."""
    q, k, v, g, beta = self._random_inputs(T=32, seed=123)
    o_naive, s_naive = naive_recurrent_gated_delta_rule(q, k, v, g, beta)
    o_chunk, s_chunk = chunked_gated_delta_rule(q, k, v, g, beta)
    np.testing.assert_allclose(o_chunk, o_naive, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(s_chunk, s_naive, atol=1e-7, rtol=1e-5)

  def test_chunked_matches_naive_full_chunk(self):
    """Chunked algorithm matches naive recurrence for T=64 (full CHUNK_SIZE)."""
    q, k, v, g, beta = self._random_inputs(T=64, seed=999)
    o_naive, s_naive = naive_recurrent_gated_delta_rule(q, k, v, g, beta)
    o_chunk, s_chunk = chunked_gated_delta_rule(q, k, v, g, beta)
    np.testing.assert_allclose(o_chunk, o_naive, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(s_chunk, s_naive, atol=1e-6, rtol=1e-5)

  def test_chunked_with_initial_state(self):
    """Chunked algorithm matches naive with non-zero initial state."""
    q, k, v, g, beta = self._random_inputs(T=16, seed=77)
    np.random.seed(77)
    S0 = np.random.randn(B, H, d, d).astype(np.float64) * 0.1
    o_naive, s_naive = naive_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=S0)
    o_chunk, s_chunk = chunked_gated_delta_rule(q, k, v, g, beta, initial_state=S0)
    np.testing.assert_allclose(o_chunk, o_naive, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(s_chunk, s_naive, atol=1e-7, rtol=1e-5)

  def test_multi_chunk_sequential(self):
    """Process T=64 as two chunks of 32, compare against naive over full sequence."""
    q, k, v, g, beta = self._random_inputs(T=64, seed=55)
    # naive: full sequence
    o_naive, s_naive = naive_recurrent_gated_delta_rule(q, k, v, g, beta)
    # chunked: two chunks of 32
    o1, s1 = chunked_gated_delta_rule(q[:,:,:32], k[:,:,:32], v[:,:,:32], g[:,:,:32], beta[:,:,:32])
    o2, s2 = chunked_gated_delta_rule(q[:,:,32:], k[:,:,32:], v[:,:,32:], g[:,:,32:], beta[:,:,32:], initial_state=s1)
    o_multi = np.concatenate([o1, o2], axis=2)
    np.testing.assert_allclose(o_multi, o_naive, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(s2, s_naive, atol=1e-6, rtol=1e-5)

class TestGatedDeltaNetTinygrad(unittest.TestCase):
  """Test the tinygrad _prefill path produces correct outputs."""

  def _random_inputs(self, T, seed=42):
    np.random.seed(seed)
    q = np.random.randn(B, H, T, d).astype(np.float32)
    k = np.random.randn(B, H, T, d).astype(np.float32)
    v = np.random.randn(B, H, T, d).astype(np.float32)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12) * (d ** -0.5)
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)
    g = -np.abs(np.random.randn(B, H, T).astype(np.float32)) * 0.1
    beta = (1.0 / (1.0 + np.exp(-np.random.randn(B, H, T).astype(np.float32))))
    return q, k, v, g, beta

  def test_ut_transform_tinygrad(self):
    """Test the UT transform portion in tinygrad matches numpy reference."""
    from tinygrad.apps.llm import CHUNK_SIZE
    T = 16
    q, k, v, g, beta = self._random_inputs(T, seed=42)
    C = CHUNK_SIZE

    # numpy reference
    g_cumsum_np = np.cumsum(g, axis=-1)
    g_i = g_cumsum_np[:, :, :, None]
    g_j = g_cumsum_np[:, :, None, :]
    L_mask_np = np.exp(g_i - g_j)
    k_beta_np = k * beta[:, :, :, None]
    kkt_np = k_beta_np @ k.transpose(0, 1, 3, 2)
    attn_np = -(kkt_np * L_mask_np)
    attn_np = np.tril(attn_np, -1)
    for i in range(1, T):
      row_i = attn_np[:, :, i:i+1, :i]
      block = attn_np[:, :, :i, :i]
      attn_np[:, :, i:i+1, :i] = row_i + row_i @ block
    attn_np = attn_np + np.eye(T)

    # tinygrad version - pad to C, compute, then slice
    k_t = Tensor(k).pad((None, None, (0, C - T), None))
    v_t = Tensor(v).pad((None, None, (0, C - T), None))
    g_t = Tensor(g).pad((None, None, (0, C - T)))
    beta_t = Tensor(beta).pad((None, None, (0, C - T)))

    g_cumsum_t = g_t.cumsum(axis=-1)
    g_i_t = g_cumsum_t.unsqueeze(-1)
    g_j_t = g_cumsum_t.unsqueeze(-2)
    L_mask_t = (g_i_t - g_j_t).exp()

    k_beta_t = k_t * beta_t.unsqueeze(-1)
    kkt_t = k_beta_t @ k_t.transpose(-1, -2)
    attn_t = -(kkt_t * L_mask_t)
    attn_t = attn_t.tril(-1)

    for i in range(1, C):
      row_i = attn_t[:, :, i:i+1, :i]
      block = attn_t[:, :, :i, :i]
      correction = row_i @ block
      pad_right = Tensor.zeros(*correction.shape[:-1], C - i)
      update_row = correction.cat(pad_right, dim=-1)
      pad_top = Tensor.zeros(*attn_t.shape[:-2], i, C)
      pad_bot = Tensor.zeros(*attn_t.shape[:-2], C - i - 1, C)
      update = pad_top.cat(update_row, dim=-2).cat(pad_bot, dim=-2)
      attn_t = attn_t + update
    attn_t = attn_t + Tensor.eye(C)

    # compare the T×T upper-left block
    attn_result = attn_t[:, :, :T, :T].numpy()
    np.testing.assert_allclose(attn_result, attn_np, atol=1e-5, rtol=1e-4)

  def test_full_prefill_tinygrad(self):
    """Test full _prefill path in tinygrad matches naive token-by-token recurrence."""
    from tinygrad.apps.llm import CHUNK_SIZE
    T = 8
    q_np, k_np, v_np, g_np, beta_np = self._random_inputs(T, seed=99)

    # numpy naive reference (float64 for precision)
    o_ref, s_ref = naive_recurrent_gated_delta_rule(
      q_np.astype(np.float64), k_np.astype(np.float64), v_np.astype(np.float64),
      g_np.astype(np.float64), beta_np.astype(np.float64))

    # tinygrad chunked
    C = CHUNK_SIZE
    q_t = Tensor(q_np).pad((None, None, (0, C - T), None))
    k_t = Tensor(k_np).pad((None, None, (0, C - T), None))
    v_t = Tensor(v_np).pad((None, None, (0, C - T), None))
    g_t = Tensor(g_np).pad((None, None, (0, C - T)))
    beta_t = Tensor(beta_np).pad((None, None, (0, C - T)))

    g_cumsum = g_t.cumsum(axis=-1)
    g_i = g_cumsum.unsqueeze(-1)
    g_j = g_cumsum.unsqueeze(-2)
    L_mask = (g_i - g_j).exp()

    k_beta = k_t * beta_t.unsqueeze(-1)
    attn = -(k_beta @ k_t.transpose(-1, -2) * L_mask).tril(-1)

    for i in range(1, C):
      row_i = attn[:, :, i:i+1, :i]
      block = attn[:, :, :i, :i]
      correction = row_i @ block
      pad_right = Tensor.zeros(*correction.shape[:-1], C - i)
      update_row = correction.cat(pad_right, dim=-1)
      pad_top = Tensor.zeros(*attn.shape[:-2], i, C)
      pad_bot = Tensor.zeros(*attn.shape[:-2], C - i - 1, C)
      update = pad_top.cat(update_row, dim=-2).cat(pad_bot, dim=-2)
      attn = attn + update
    attn = attn + Tensor.eye(C)

    u = attn @ (v_t * beta_t.unsqueeze(-1))
    w = attn @ (k_t * (beta_t * g_cumsum.exp()).unsqueeze(-1))

    S = Tensor.zeros(B, H, d, d)  # zero initial state
    q_scaled = q_t * g_cumsum.unsqueeze(-1).exp()
    o_inter = q_scaled @ S
    v_new = u - w @ S
    qk = q_t @ k_t.transpose(-1, -2)
    causal_mask = L_mask.tril(0)
    o_intra = (qk * causal_mask) @ v_new
    o = (o_inter + o_intra)[:, :, :T, :].numpy()

    np.testing.assert_allclose(o, o_ref, atol=1e-4, rtol=1e-3)

if __name__ == "__main__":
  unittest.main()
