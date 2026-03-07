"""Test GatedDeltaNet _prefill against naive recurrent (ground truth) and FLA chunk reference."""
import numpy as np
np.random.seed(42)

# ---- Ground truth: naive recurrent (from FLA, ported to numpy) ----
def naive_recurrent(q, k, v, beta, g, initial_state=None):
    """Naive recurrent gated delta rule. All inputs: (B,H,T,d). beta/g: (B,H,T). Returns (o, final_state)."""
    B, H, T, d = q.shape
    V = v.shape[-1]
    o = np.zeros((B, H, T, V), dtype=np.float64)
    S = np.zeros((B, H, d, V), dtype=np.float64) if initial_state is None else initial_state.copy()

    for i in range(T):
        b_q = q[:, :, i]          # (B,H,d)
        b_k = k[:, :, i]          # (B,H,d)
        b_v = v[:, :, i].copy()   # (B,H,V)
        # decay
        S = S * np.exp(g[:, :, i])[:, :, None, None]
        # delta: prediction error
        b_v = b_v - np.einsum('bhkv,bhk->bhv', S, b_k)
        b_v = b_v * beta[:, :, i, None]
        # write
        S = S + b_k[:, :, :, None] * b_v[:, :, None, :]
        # read
        o[:, :, i] = np.einsum('bhd,bhdv->bhv', b_q, S)

    return o, S


# ---- FLA chunk reference (ported to numpy) ----
def fla_chunk(q, k, v, beta, g, chunk_size=64, initial_state=None):
    """FLA's naive_chunk_gated_delta_rule ported to numpy. All (B,H,T,d). Returns (o, final_state)."""
    B, H, T, d = q.shape
    V = v.shape[-1]
    C = chunk_size
    pad_len = (C - (T % C)) % C
    if pad_len > 0:
        q = np.pad(q, ((0,0),(0,0),(0,pad_len),(0,0)))
        k = np.pad(k, ((0,0),(0,0),(0,pad_len),(0,0)))
        v = np.pad(v, ((0,0),(0,0),(0,pad_len),(0,0)))
        beta = np.pad(beta, ((0,0),(0,0),(0,pad_len)))
        g = np.pad(g, ((0,0),(0,0),(0,pad_len)))

    L = q.shape[2]
    N = L // C
    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]

    # reshape to (B,H,N,C,d)
    q_c = q.reshape(B, H, N, C, d)
    k_c = k.reshape(B, H, N, C, d)
    v_c = v_beta.reshape(B, H, N, C, V)
    k_beta_c = k_beta.reshape(B, H, N, C, d)
    g_c = g.reshape(B, H, N, C)

    decay = np.cumsum(g_c, axis=-1)  # (B,H,N,C)

    # L_mask: (B,H,N,C,C)
    L_mask = np.exp(decay[..., :, None] - decay[..., None, :])
    L_mask = np.tril(L_mask)

    # UT transform: attn = -(k_beta @ k^T) * L_mask, masked to strict lower tri
    attn = np.zeros((B, H, N, C, C), dtype=np.float64)
    for n in range(N):
        A = -(k_beta_c[:, :, n] @ k_c[:, :, n].transpose(0, 1, 3, 2)) * L_mask[:, :, n]
        # zero out upper triangle + diagonal
        A = np.tril(A, k=-1)
        # forward substitution
        for i in range(1, C):
            A[:, :, i, :i] = A[:, :, i, :i] + np.einsum('bhj,bhji->bhi', A[:, :, i:i+1, :i].squeeze(-2), A[:, :, :i, :i])
        A = A + np.eye(C)
        attn[:, :, n] = A

    # corrected values
    u = np.zeros_like(v_c)
    w = np.zeros_like(k_beta_c)
    for n in range(N):
        u[:, :, n] = attn[:, :, n] @ v_c[:, :, n]
        w[:, :, n] = attn[:, :, n] @ (k_beta_c[:, :, n] * np.exp(decay[:, :, n, :, None]))

    # inter-chunk + intra-chunk
    S = np.zeros((B, H, d, V), dtype=np.float64) if initial_state is None else initial_state.copy()
    o = np.zeros((B, H, N, C, V), dtype=np.float64)
    for n in range(N):
        q_n = q_c[:, :, n]  # (B,H,C,d)
        k_n = k_c[:, :, n]
        v_n = u[:, :, n]    # corrected v
        w_n = w[:, :, n]    # corrected k

        # causal QK attention within chunk
        qk = q_n @ k_n.transpose(0, 1, 3, 2)  # (B,H,C,C)
        causal_qk = qk * np.tril(L_mask[:, :, n])

        # v_new = u - w @ S
        v_prime = w_n @ S
        v_new = v_n - v_prime

        # o = q*exp(cumsum_g) @ S + causal_qk @ v_new
        o_inter = (q_n * np.exp(decay[:, :, n, :, None])) @ S
        o[:, :, n] = o_inter + causal_qk @ v_new

        # state update
        g_total = decay[:, :, n, -1]  # (B,H)
        k_end = k_n * np.exp(g_total[:, :, None, None] - decay[:, :, n, :, None])
        S = S * np.exp(g_total[:, :, None, None]) + k_end.transpose(0, 1, 3, 2) @ v_new

    o = o.reshape(B, H, L, V)[:, :, :T]
    return o, S


# ---- Test ----
def test_prefill_matches_naive():
    B, T, H, d = 1, 7, 4, 8  # small dims for testing
    scale = 1.0 / (d ** 0.5)

    # random inputs
    q = np.random.randn(B, H, T, d).astype(np.float64) * scale
    k = np.random.randn(B, H, T, d).astype(np.float64)
    v = np.random.randn(B, H, T, d).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-np.random.randn(B, H, T)))  # sigmoid
    g = -np.abs(np.random.randn(B, H, T)) * 0.1  # small negative (log-decay)

    # normalize q, k like our code does
    q_norm = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12) * scale
    k_norm = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)

    S0 = np.random.randn(B, H, d, d).astype(np.float64) * 0.01

    # ground truth: naive recurrent
    o_naive, S_naive = naive_recurrent(q_norm, k_norm, v, beta, g, initial_state=S0)

    # FLA chunk reference
    o_chunk, S_chunk = fla_chunk(q_norm, k_norm, v, beta, g, chunk_size=64, initial_state=S0)

    # compare
    o_err = np.max(np.abs(o_naive - o_chunk))
    S_err = np.max(np.abs(S_naive - S_chunk))
    print(f"naive vs FLA chunk:")
    print(f"  output max error: {o_err:.2e}")
    print(f"  state  max error: {S_err:.2e}")
    assert o_err < 1e-10, f"output mismatch: {o_err}"
    assert S_err < 1e-10, f"state mismatch: {S_err}"
    print("  ✅ PASS")


def test_our_prefill_core():
    """Test just the core math of our _prefill (UT transform + intra/inter chunk), without tinygrad."""
    B, T, H, d = 1, 11, 4, 8
    C = 64  # CHUNK_SIZE
    scale = 1.0 / (d ** 0.5)

    q = np.random.randn(B, H, T, d).astype(np.float64)
    k = np.random.randn(B, H, T, d).astype(np.float64)
    v = np.random.randn(B, H, T, d).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-np.random.randn(B, H, T)))
    g = -np.abs(np.random.randn(B, H, T)) * 0.1

    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12) * scale
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)

    S0 = np.random.randn(B, H, d, d).astype(np.float64) * 0.01

    # ground truth
    o_naive, S_naive = naive_recurrent(q, k, v, beta, g, initial_state=S0)

    # our prefill logic (numpy port of _prefill)
    # pad to C
    q_p = np.pad(q, ((0,0),(0,0),(0,C-T),(0,0)))
    k_p = np.pad(k, ((0,0),(0,0),(0,C-T),(0,0)))
    v_p = np.pad(v, ((0,0),(0,0),(0,C-T),(0,0)))
    beta_p = np.pad(beta, ((0,0),(0,0),(0,C-T)))
    g_p = np.pad(g, ((0,0),(0,0),(0,C-T)))

    g_cumsum = np.cumsum(g_p, axis=-1)
    L_mask = np.exp(g_cumsum[:, :, :, None] - g_cumsum[:, :, None, :])

    # UT transform
    k_beta = k_p * beta_p[..., None]
    attn = -(k_beta @ k_p.transpose(0, 1, 3, 2)) * L_mask
    attn = np.tril(attn, k=-1)

    for i in range(1, C):
        correction = attn[:, :, i:i+1, :i] @ attn[:, :, :i, :i]
        attn[:, :, i, :i] += correction[:, :, 0, :]

    attn = attn + np.eye(C)

    u = attn @ (v_p * beta_p[..., None])
    w = attn @ (k_beta * np.exp(g_cumsum)[..., None])

    S = S0.copy()
    q_scaled = q_p * np.exp(g_cumsum)[..., None]
    o_inter = q_scaled @ S

    w_S = w @ S
    v_new = u - w_S

    qk = q_p @ k_p.transpose(0, 1, 3, 2)
    causal_mask = np.tril(L_mask)
    o_intra = (qk * causal_mask) @ v_new

    o_ours = (o_inter + o_intra)[:, :, :T]

    # state update
    g_total = g_cumsum[:, :, -1]
    k_end = k_p * np.exp(g_total[:, :, None, None] - g_cumsum[:, :, :, None])
    S_ours = S * np.exp(g_total[:, :, None, None]) + k_end.transpose(0, 1, 3, 2) @ v_new

    o_err = np.max(np.abs(o_naive - o_ours))
    S_err = np.max(np.abs(S_naive - S_ours))
    print(f"\nnaive vs our prefill core:")
    print(f"  output max error: {o_err:.2e}")
    print(f"  state  max error: {S_err:.2e}")
    assert o_err < 1e-10, f"output mismatch: {o_err}"
    assert S_err < 1e-10, f"state mismatch: {S_err}"
    print("  ✅ PASS")


def neumann_ut_transform(A_raw, C):
    """Neumann series doubling: (I-A)^{-1} = (I+A)(I+A^2)(I+A^4)... for nilpotent A."""
    A = np.tril(A_raw, k=-1)
    attn = np.eye(C) + A
    An = A
    n = 1
    while n < C:
        An = An @ An
        attn = attn @ (np.eye(C) + An)
        n *= 2
    return attn

def forward_sub_ut_transform(A_raw, C):
    """Original forward substitution: (I-A)^{-1} via sequential row updates."""
    A = np.tril(A_raw, k=-1)
    for i in range(1, C):
        A[:, :, i, :i] += (A[:, :, i:i+1, :i] @ A[:, :, :i, :i])[:, :, 0, :]
    return A + np.eye(C)

def test_neumann_matches_forward_sub():
    """Verify Neumann doubling produces identical results to forward substitution."""
    for C in [8, 16, 32, 64]:
        B, H = 2, 4
        A_raw = np.random.randn(B, H, C, C).astype(np.float64) * 0.1

        attn_neumann = neumann_ut_transform(A_raw, C)
        attn_fwdsub = forward_sub_ut_transform(A_raw, C)

        err = np.max(np.abs(attn_neumann - attn_fwdsub))
        print(f"  Neumann vs forward_sub (C={C}): max error = {err:.2e}")
        assert err < 1e-10, f"mismatch at C={C}: {err}"
    print("  ✅ PASS")


def test_neumann_prefill_core():
    """Test _prefill with Neumann doubling (C=16) matches naive recurrent."""
    B, T, H, d = 1, 11, 4, 8
    C = 16  # new CHUNK_SIZE
    scale = 1.0 / (d ** 0.5)

    q = np.random.randn(B, H, T, d).astype(np.float64)
    k = np.random.randn(B, H, T, d).astype(np.float64)
    v = np.random.randn(B, H, T, d).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-np.random.randn(B, H, T)))
    g = -np.abs(np.random.randn(B, H, T)) * 0.1

    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12) * scale
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)

    S0 = np.random.randn(B, H, d, d).astype(np.float64) * 0.01

    # ground truth
    o_naive, S_naive = naive_recurrent(q, k, v, beta, g, initial_state=S0)

    # prefill with Neumann doubling, C=16
    q_p = np.pad(q, ((0,0),(0,0),(0,C-T),(0,0)))
    k_p = np.pad(k, ((0,0),(0,0),(0,C-T),(0,0)))
    v_p = np.pad(v, ((0,0),(0,0),(0,C-T),(0,0)))
    beta_p = np.pad(beta, ((0,0),(0,0),(0,C-T)))
    g_p = np.pad(g, ((0,0),(0,0),(0,C-T)))

    g_cumsum = np.cumsum(g_p, axis=-1)
    L_mask = np.exp(g_cumsum[:, :, :, None] - g_cumsum[:, :, None, :])

    k_beta = k_p * beta_p[..., None]
    A_raw = -(k_beta @ k_p.transpose(0, 1, 3, 2)) * L_mask
    attn = neumann_ut_transform(A_raw, C)

    u = attn @ (v_p * beta_p[..., None])
    w = attn @ (k_beta * np.exp(g_cumsum)[..., None])

    S = S0.copy()
    q_scaled = q_p * np.exp(g_cumsum)[..., None]
    o_inter = q_scaled @ S
    v_new = u - w @ S
    causal_mask = np.tril(L_mask)
    o_intra = (q_p @ k_p.transpose(0, 1, 3, 2) * causal_mask) @ v_new
    o_ours = (o_inter + o_intra)[:, :, :T]

    g_total = g_cumsum[:, :, -1]
    k_end = k_p * np.exp(g_total[:, :, None, None] - g_cumsum[:, :, :, None])
    S_ours = S * np.exp(g_total[:, :, None, None]) + k_end.transpose(0, 1, 3, 2) @ v_new

    o_err = np.max(np.abs(o_naive - o_ours))
    S_err = np.max(np.abs(S_naive - S_ours))
    print(f"\nnaive vs Neumann prefill (C={C}, T={T}):")
    print(f"  output max error: {o_err:.2e}")
    print(f"  state  max error: {S_err:.2e}")
    assert o_err < 1e-10, f"output mismatch: {o_err}"
    assert S_err < 1e-10, f"state mismatch: {S_err}"
    print("  ✅ PASS")


def test_multichunk_prefill():
    """Test multi-chunk prefill (multiple C=16 chunks) matches naive recurrent."""
    B, H, d = 1, 4, 8
    C = 16
    T = 37  # spans 3 chunks: 16+16+5
    scale = 1.0 / (d ** 0.5)

    q = np.random.randn(B, H, T, d).astype(np.float64)
    k = np.random.randn(B, H, T, d).astype(np.float64)
    v = np.random.randn(B, H, T, d).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-np.random.randn(B, H, T)))
    g = -np.abs(np.random.randn(B, H, T)) * 0.1

    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12) * scale
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)

    S0 = np.random.randn(B, H, d, d).astype(np.float64) * 0.01

    # ground truth
    o_naive, S_naive = naive_recurrent(q, k, v, beta, g, initial_state=S0)

    # multi-chunk prefill
    S = S0.copy()
    o_chunks = []
    pos = 0
    while pos < T:
        chunk_len = min(C, T - pos)
        q_c = q[:, :, pos:pos+chunk_len]
        k_c = k[:, :, pos:pos+chunk_len]
        v_c = v[:, :, pos:pos+chunk_len]
        beta_c = beta[:, :, pos:pos+chunk_len]
        g_c = g[:, :, pos:pos+chunk_len]

        # pad to C
        pad = C - chunk_len
        q_p = np.pad(q_c, ((0,0),(0,0),(0,pad),(0,0)))
        k_p = np.pad(k_c, ((0,0),(0,0),(0,pad),(0,0)))
        v_p = np.pad(v_c, ((0,0),(0,0),(0,pad),(0,0)))
        beta_p = np.pad(beta_c, ((0,0),(0,0),(0,pad)))
        g_p = np.pad(g_c, ((0,0),(0,0),(0,pad)))

        g_cumsum = np.cumsum(g_p, axis=-1)
        L_mask = np.exp(g_cumsum[:, :, :, None] - g_cumsum[:, :, None, :])

        k_beta = k_p * beta_p[..., None]
        A_raw = -(k_beta @ k_p.transpose(0, 1, 3, 2)) * L_mask
        attn = neumann_ut_transform(A_raw, C)

        u = attn @ (v_p * beta_p[..., None])
        w = attn @ (k_beta * np.exp(g_cumsum)[..., None])

        q_scaled = q_p * np.exp(g_cumsum)[..., None]
        o_inter = q_scaled @ S
        v_new = u - w @ S
        causal_mask = np.tril(L_mask)
        o_intra = (q_p @ k_p.transpose(0, 1, 3, 2) * causal_mask) @ v_new
        o_chunk = (o_inter + o_intra)[:, :, :chunk_len]
        o_chunks.append(o_chunk)

        # state update
        g_total = g_cumsum[:, :, -1]
        k_end = k_p * np.exp(g_total[:, :, None, None] - g_cumsum[:, :, :, None])
        S = S * np.exp(g_total[:, :, None, None]) + k_end.transpose(0, 1, 3, 2) @ v_new
        pos += chunk_len

    o_multi = np.concatenate(o_chunks, axis=2)

    o_err = np.max(np.abs(o_naive - o_multi))
    S_err = np.max(np.abs(S_naive - S))
    print(f"\nnaive vs multi-chunk prefill (C={C}, T={T}):")
    print(f"  output max error: {o_err:.2e}")
    print(f"  state  max error: {S_err:.2e}")
    assert o_err < 1e-10, f"output mismatch: {o_err}"
    assert S_err < 1e-10, f"state mismatch: {S_err}"
    print("  ✅ PASS")


def test_rollout_matches_naive():
    """Verify our _rollout logic (token-by-token) matches naive recurrent."""
    B, T, H, d = 1, 13, 4, 8
    scale = 1.0 / (d ** 0.5)

    q = np.random.randn(B, H, T, d).astype(np.float64)
    k = np.random.randn(B, H, T, d).astype(np.float64)
    v = np.random.randn(B, H, T, d).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-np.random.randn(B, H, T)))
    g = -np.abs(np.random.randn(B, H, T)) * 0.1

    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12) * scale
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)

    S0 = np.random.randn(B, H, d, d).astype(np.float64) * 0.01

    # ground truth
    o_naive, S_naive = naive_recurrent(q, k, v, beta, g, initial_state=S0)

    # simulate _rollout: process one token at a time
    S = S0.copy()
    o_rollout = np.zeros_like(o_naive)
    for t in range(T):
        b_q = q[:, :, t]
        b_k = k[:, :, t]
        b_v = v[:, :, t]
        b_g = g[:, :, t]
        b_beta = beta[:, :, t]

        S = S * np.exp(b_g)[:, :, None, None]
        delta = (b_v - np.einsum('bhkv,bhk->bhv', S, b_k)) * b_beta[:, :, None]
        S = S + b_k[:, :, :, None] * delta[:, :, None, :]
        o_rollout[:, :, t] = np.einsum('bhd,bhdv->bhv', b_q, S)

    o_err = np.max(np.abs(o_naive - o_rollout))
    S_err = np.max(np.abs(S_naive - S))
    print(f"\nnaive vs rollout (token-by-token):")
    print(f"  output max error: {o_err:.2e}")
    print(f"  state  max error: {S_err:.2e}")
    assert o_err < 1e-12, f"output mismatch: {o_err}"
    assert S_err < 1e-12, f"state mismatch: {S_err}"
    print("  ✅ PASS")


def test_prefill_equals_rollout():
    """The key test: does processing T tokens via _prefill give the same result as T sequential _rollout calls?"""
    B, T, H, d = 1, 17, 4, 16  # slightly larger
    C = 16  # matches new CHUNK_SIZE
    scale = 1.0 / (d ** 0.5)

    q = np.random.randn(B, H, T, d).astype(np.float64)
    k = np.random.randn(B, H, T, d).astype(np.float64)
    v = np.random.randn(B, H, T, d).astype(np.float64)
    beta = 1.0 / (1.0 + np.exp(-np.random.randn(B, H, T)))
    g = -np.abs(np.random.randn(B, H, T)) * 0.1

    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12) * scale
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-12)

    S0 = np.random.randn(B, H, d, d).astype(np.float64) * 0.01

    # rollout (ground truth)
    o_naive, S_naive = naive_recurrent(q, k, v, beta, g, initial_state=S0)

    # multi-chunk prefill with Neumann doubling (matches new implementation)
    S = S0.copy()
    o_chunks = []
    pos = 0
    while pos < T:
        chunk_len = min(C, T - pos)
        q_c, k_c, v_c = q[:,:,pos:pos+chunk_len], k[:,:,pos:pos+chunk_len], v[:,:,pos:pos+chunk_len]
        beta_c, g_c = beta[:,:,pos:pos+chunk_len], g[:,:,pos:pos+chunk_len]

        pad = C - chunk_len
        q_p = np.pad(q_c, ((0,0),(0,0),(0,pad),(0,0)))
        k_p = np.pad(k_c, ((0,0),(0,0),(0,pad),(0,0)))
        v_p = np.pad(v_c, ((0,0),(0,0),(0,pad),(0,0)))
        beta_p = np.pad(beta_c, ((0,0),(0,0),(0,pad)))
        g_p = np.pad(g_c, ((0,0),(0,0),(0,pad)))

        g_cumsum = np.cumsum(g_p, axis=-1)
        L_mask = np.exp(g_cumsum[:, :, :, None] - g_cumsum[:, :, None, :])

        k_beta = k_p * beta_p[..., None]
        A_raw = -(k_beta @ k_p.transpose(0, 1, 3, 2)) * L_mask
        attn = neumann_ut_transform(A_raw, C)

        u = attn @ (v_p * beta_p[..., None])
        w = attn @ (k_beta * np.exp(g_cumsum)[..., None])

        q_scaled = q_p * np.exp(g_cumsum)[..., None]
        o_inter = q_scaled @ S
        v_new = u - w @ S
        causal_mask = np.tril(L_mask)
        o_intra = (q_p @ k_p.transpose(0, 1, 3, 2) * causal_mask) @ v_new
        o_chunks.append((o_inter + o_intra)[:, :, :chunk_len])

        g_total = g_cumsum[:, :, -1]
        k_end = k_p * np.exp(g_total[:, :, None, None] - g_cumsum[:, :, :, None])
        S = S * np.exp(g_total[:, :, None, None]) + k_end.transpose(0, 1, 3, 2) @ v_new
        pos += chunk_len

    o_prefill = np.concatenate(o_chunks, axis=2)
    S_prefill = S

    o_err = np.max(np.abs(o_naive - o_prefill))
    S_err = np.max(np.abs(S_naive - S_prefill))
    print(f"\nprefill vs rollout (T={T}, H={H}, d={d}, C={C}):")
    print(f"  output max error: {o_err:.2e}")
    print(f"  state  max error: {S_err:.2e}")
    assert o_err < 1e-9, f"output mismatch: {o_err}"
    assert S_err < 1e-9, f"state mismatch: {S_err}"
    print("  ✅ PASS")


if __name__ == "__main__":
    test_prefill_matches_naive()
    test_our_prefill_core()
    test_neumann_matches_forward_sub()
    test_neumann_prefill_core()
    test_multichunk_prefill()
    test_rollout_matches_naive()
    test_prefill_equals_rollout()
    print("\n🎉 All tests passed!")
