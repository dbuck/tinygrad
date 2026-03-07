from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools, itertools
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function
from tinygrad.uop.ops import resolve
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3"):
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","qwen35","olmo"): raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"])

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset in ('qwen2', 'qwen35'):
      if role == 'developer': role = 'system'  # Qwen3.5 doesn't support developer role natively
      return self.encode("<|im_start|>" + role + "\n")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self, eos_id:int):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset in ('qwen2', 'qwen35'): return [eos_id] + self.encode("\n")
    return [eos_id]

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).contiguous()

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor, rope_dim:int=0) -> Tensor:
  x_rot, x_pass = (x[..., :rope_dim], x[..., rope_dim:]) if rope_dim and rope_dim < x.shape[-1] else (x, None)
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x_rot.chunk(2, dim=-1)
  rot = (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)
  return rot.cat(x_pass, dim=-1) if x_pass is not None else rot

def write_after(dst:Tensor, val:Tensor, result:Tensor) -> Tensor:
  """Return result, but ensure dst.assign(val) happens first."""
  return Tensor((x:=result.uop.after(dst.uop.assign(val.uop))), device=x.device)

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, rope_dim:int=0, attn_gate:bool=False):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = head_dim
    self.rope_theta   = rope_theta
    self.max_context  = max_context
    self.qk_norm      = qk_norm
    self.rope_dim     = rope_dim
    self.attn_gate    = attn_gate

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = self.head_dim * n_heads * (2 if attn_gate else 1)
    kv_proj_out      = self.head_dim * n_kv_heads
    self.attn_q      = nn.Linear(dim, q_proj_out,  bias=False)
    self.attn_k      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_v      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_output = nn.Linear(self.head_dim * n_heads, dim,  bias=False)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm    = nn.RMSNorm(dim, norm_eps)
    if qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(qk_norm, norm_eps), nn.RMSNorm(qk_norm, norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if num_experts > 0:
      self.num_experts_per_tok = num_experts_per_tok
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)  # router
      self.ffn_gate_exps = ExpertWeights(num_experts, dim, hidden_dim)
      self.ffn_up_exps = ExpertWeights(num_experts, dim, hidden_dim)
      self.ffn_down_exps = ExpertWeights(num_experts, hidden_dim, dim)
    else:
      self.ffn_gate    = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(dim, hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(hidden_dim, dim, bias=False)

  @function(precompile=bool(getenv("PRECOMPILE", 0)))
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)
    if self.qk_norm and self.qk_norm != self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.attn_gate:
      q = q.reshape(B, T, self.n_heads, self.head_dim * 2)
      q, gate = q[..., :self.head_dim].transpose(1, 2), q[..., self.head_dim:].transpose(1, 2)  # (B,H,T,Hd) each
    else:
      q = q.reshape(B, T, self.n_heads,    self.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.qk_norm == self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    rope_d = self.rope_dim or self.head_dim
    freqs_cis = precompute_freqs_cis(rope_d, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    q = apply_rope(q, freqs_cis, self.rope_dim)
    k = apply_rope(k, freqs_cis, self.rope_dim)

    # TODO: fix assign to behave like this
    cache = write_after(self.cache_kv[:, :, :, start_pos:start_pos+T, :], Tensor.stack(k, v).contiguous(), self.cache_kv)
    k = cache[0, :, :, 0:start_pos+T, :]
    v = cache[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(start_pos+1) if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    if self.attn_gate: attn = attn * gate.transpose(1, 2).reshape(B, T, -1).sigmoid()
    attn = self.attn_output(attn)
    return x + attn

  @function(precompile=bool(getenv("PRECOMPILE", 0)))
  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = self.ffn_norm(h)
    if hasattr(self, 'ffn_gate_exps'):
      x = h_norm.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)  # (B, T, k) each
      x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))  # (B, T, k, D)
      return h + (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
    # TODO: remove the need for this contiguous
    gated  = self.ffn_gate(h_norm).silu().contiguous() * self.ffn_up(h_norm)
    return h + self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      # NOTE: clone is used to promise the creation of a specific buffer
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.n_kv_heads, self.max_context, self.head_dim, device=x.device).clone()
    return self._feed_forward(self._attention(x, start_pos)).contiguous()

CHUNK_SIZE = 16  # fixed chunk size for GatedDeltaNet parallel prefill (small = fewer graph ops for BEAM)

class GatedDeltaNetBlock:
  def __init__(self, dim:int, hidden_dim:int, norm_eps:float, n_k_heads:int, n_v_heads:int, head_dim:int, conv_kernel:int):
    self.n_k_heads, self.n_v_heads, self.head_dim, self.conv_kernel = n_k_heads, n_v_heads, head_dim, conv_kernel
    self.kv_repeat = n_v_heads // n_k_heads
    key_dim, value_dim = n_k_heads * head_dim, n_v_heads * head_dim
    conv_dim = key_dim * 2 + value_dim

    self.attn_norm  = nn.RMSNorm(dim, norm_eps)
    self.ffn_norm   = nn.RMSNorm(dim, norm_eps)  # loaded from post_attention_norm
    self.ssm_norm   = nn.RMSNorm(head_dim, norm_eps)

    self.attn_qkv   = nn.Linear(dim, conv_dim, bias=False)
    self.attn_gate  = nn.Linear(dim, value_dim, bias=False)
    self.ssm_alpha  = nn.Linear(dim, n_v_heads, bias=False)
    self.ssm_beta   = nn.Linear(dim, n_v_heads, bias=False)
    self.ssm_out    = nn.Linear(value_dim, dim, bias=False)
    self.ssm_a      = Tensor.zeros(n_v_heads)
    self.ssm_dt     = Tensor.zeros(n_v_heads)
    self.ssm_conv1d = Tensor.zeros(conv_dim, conv_kernel)

    self.ffn_gate   = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_up     = nn.Linear(dim, hidden_dim, bias=False)
    self.ffn_down   = nn.Linear(hidden_dim, dim, bias=False)

  def _ensure_states(self, x:Tensor):
    B = x.shape[0]
    key_dim, value_dim = self.n_k_heads * self.head_dim, self.n_v_heads * self.head_dim
    conv_dim = key_dim * 2 + value_dim
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(B, conv_dim, self.conv_kernel - 1, device=x.device).clone()
    if not hasattr(self, "ssm_state"):
      self.ssm_state = Tensor.zeros(B, self.n_v_heads, self.head_dim, self.head_dim, device=x.device).clone()

  def _split_qkv(self, qkv_conv:Tensor, B:int):
    """Split conv output into Q, K, V with correct head counts, normalize, and expand Q/K via GQA repeat."""
    key_dim = self.n_k_heads * self.head_dim
    q = qkv_conv[..., :key_dim].reshape(*qkv_conv.shape[:-1], self.n_k_heads, self.head_dim)
    k = qkv_conv[..., key_dim:key_dim*2].reshape(*qkv_conv.shape[:-1], self.n_k_heads, self.head_dim)
    v = qkv_conv[..., key_dim*2:].reshape(*qkv_conv.shape[:-1], self.n_v_heads, self.head_dim)
    q, k = q.normalize(dim=-1) * (self.head_dim ** -0.5), k.normalize(dim=-1)
    if self.kv_repeat > 1:
      q = q.repeat_interleave(self.kv_repeat, dim=-2)
      k = k.repeat_interleave(self.kv_repeat, dim=-2)
    return q, k, v

  def _rollout(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    """Single-token recurrence (T=1)."""
    B = x.shape[0]
    Hv, d = self.n_v_heads, self.head_dim
    x_s = x[:, 0]                                                                      # (B,D)
    x_norm = self.attn_norm(x_s)

    # input projections
    qkv = self.attn_qkv(x_norm)                                                       # (B, conv_dim)
    z = self.attn_gate(x_norm).reshape(B, Hv, d)                                      # (B,Hv,d)
    beta = self.ssm_beta(x_norm).sigmoid()                                             # (B,Hv)
    g = (-self.ssm_a.exp() * (self.ssm_alpha(x_norm) + self.ssm_dt).softplus())        # (B,Hv)

    # causal depthwise conv1d (single token)
    conv_window = self.conv_state.cat(qkv.unsqueeze(-1), dim=-1)                       # (B,C,K)
    qkv_conv = (conv_window * self.ssm_conv1d.reshape(1, -1, self.conv_kernel)).sum(-1).silu().contiguous()
    qkv_conv = write_after(self.conv_state, conv_window[:, :, 1:].contiguous(), qkv_conv)

    # split Q, K, V with GQA expand
    q, k, v = self._split_qkv(qkv_conv, B)                                            # all (B,Hv,d)

    # gated delta rule recurrence
    state = self.ssm_state * g.exp().unsqueeze(-1).unsqueeze(-1)                       # (B,Hv,d,d)
    delta = (v - (state * k.unsqueeze(-1)).sum(-2)) * beta.unsqueeze(-1)               # (B,Hv,d)
    state = (state + k.unsqueeze(-1) * delta.unsqueeze(-2)).contiguous()                # (B,Hv,d,d)

    o = (state * q.unsqueeze(-1)).sum(-2).contiguous()                                 # (B,Hv,d)
    o = write_after(self.ssm_state, state, o)

    # gated output + residual + FFN
    h = x_s + self.ssm_out((self.ssm_norm(o) * z.silu()).reshape(B, -1))
    h_norm = self.ffn_norm(h)
    return (h + self.ffn_down(self.ffn_gate(h_norm).silu().contiguous() * self.ffn_up(h_norm))).reshape(B, 1, -1).contiguous()

  def _prefill(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    """Chunked parallel prefill (T > 1). Pads T to CHUNK_SIZE, processes in parallel via WY/UT transform."""
    C = CHUNK_SIZE
    B, T = x.shape[0], x.shape[1]
    Hv, d = self.n_v_heads, self.head_dim

    x_norm = self.attn_norm(x)                                                         # (B,T,D)

    # input projections — all (B,T,...)
    qkv = self.attn_qkv(x_norm)                                                       # (B,T,conv_dim)
    z = self.attn_gate(x_norm).reshape(B, T, Hv, d)                                   # (B,T,Hv,d)
    beta = self.ssm_beta(x_norm).sigmoid()                                             # (B,T,Hv)
    g = (-self.ssm_a.exp() * (self.ssm_alpha(x_norm) + self.ssm_dt).softplus())       # (B,T,Hv)

    # causal depthwise conv1d over T tokens
    K = self.conv_kernel
    qkv_t = qkv.permute(0, 2, 1)                                                      # (B,conv_dim,T)
    padded = self.conv_state.cat(qkv_t, dim=-1)                                        # (B,conv_dim,T+K-1)
    # build (B,conv_dim,T,K) windows via shifted slices
    windows = Tensor.stack(*[padded[:, :, i:i+T] for i in range(K)], dim=-1)
    qkv_conv = (windows * self.ssm_conv1d.reshape(1, -1, 1, K)).sum(-1).silu()         # (B,conv_dim,T)
    qkv_conv = qkv_conv.contiguous()
    new_conv_state = padded[:, :, T:].contiguous()                                     # (B,conv_dim,K-1)
    qkv_conv = write_after(self.conv_state, new_conv_state, qkv_conv)
    qkv_conv = qkv_conv.permute(0, 2, 1)                                              # (B,T,conv_dim)

    # split Q, K, V with GQA expand — all (B,T,Hv,d)
    q, k, v = self._split_qkv(qkv_conv, B)

    # pad T → C along dim 1, with zeros (beta=0, g=0 → padded positions are no-ops)
    q    =    q.pad((None, (0, C - T), None, None))                                    # (B,C,Hv,d)
    k    =    k.pad((None, (0, C - T), None, None))                                    # (B,C,Hv,d)
    v    =    v.pad((None, (0, C - T), None, None))                                    # (B,C,Hv,d)
    g    =    g.pad((None, (0, C - T), None))                                          # (B,C,Hv)
    beta = beta.pad((None, (0, C - T), None))                                          # (B,C,Hv)

    # transpose to (B,Hv,C,d) for matmuls
    q = q.permute(0, 2, 1, 3)                                                          # (B,Hv,C,d)
    k = k.permute(0, 2, 1, 3)                                                          # (B,Hv,C,d)
    v = v.permute(0, 2, 1, 3)                                                          # (B,Hv,C,d)
    g = g.permute(0, 2, 1)                                                             # (B,Hv,C)
    beta = beta.permute(0, 2, 1)                                                       # (B,Hv,C)

    # cumulative sum of log-gates within the chunk
    g_cumsum = g.cumsum(axis=-1)                                                       # (B,Hv,C)

    # decay mask: Gamma[i,j] = exp(g_cumsum[i] - g_cumsum[j]) for i >= j
    g_cumsum_i = g_cumsum.unsqueeze(-1)                                                # (B,Hv,C,1)
    g_cumsum_j = g_cumsum.unsqueeze(-2)                                                # (B,Hv,1,C)
    L_mask = (g_cumsum_i - g_cumsum_j).exp()                                           # (B,Hv,C,C)

    # === UT Transform (Neumann series doubling) ===
    k_beta = k * beta.unsqueeze(-1)                                                    # (B,Hv,C,d)
    A = (-(k_beta @ k.transpose(-1, -2)) * L_mask).tril(-1)                           # (B,Hv,C,C) strict lower tri

    # (I-A)^{-1} = (I+A)(I+A^2)(I+A^4)...(I+A^{C/2}) since A is nilpotent (A^C = 0)
    eye_C = Tensor.eye(C, device=x.device)
    attn = (eye_C + A).contiguous()
    An = A
    n = 1
    while n < C:
      An = (An @ An).contiguous()                                                      # A^{2^k}
      attn = (attn @ (eye_C + An)).contiguous()                                        # accumulate
      n *= 2

    # corrected values and keys
    u = attn @ (v * beta.unsqueeze(-1))                                                # (B,Hv,C,d)
    w = attn @ (k * (beta * g_cumsum.exp()).unsqueeze(-1))                              # (B,Hv,C,d)

    # === Intra-chunk output computation ===
    S = self.ssm_state                                                                 # (B,Hv,d,d)

    q_scaled = q * g_cumsum.unsqueeze(-1).exp()                                        # (B,Hv,C,d)
    o_inter = q_scaled @ S                                                             # (B,Hv,C,d)

    w_S = w @ S                                                                        # (B,Hv,C,d)
    v_new = u - w_S                                                                    # (B,Hv,C,d)

    qk = q @ k.transpose(-1, -2)                                                      # (B,Hv,C,C)
    causal_mask = L_mask.tril(0)
    o_intra = (qk * causal_mask) @ v_new                                               # (B,Hv,C,d)

    o = o_inter + o_intra                                                              # (B,Hv,C,d)

    # === State update ===
    g_total = g_cumsum[:, :, -1]                                                       # (B,Hv)
    k_end = k * (g_total.unsqueeze(-1).unsqueeze(-1) - g_cumsum.unsqueeze(-1)).exp()   # (B,Hv,C,d)
    new_state = (S * g_total.unsqueeze(-1).unsqueeze(-1).exp() +
                 k_end.transpose(-1, -2) @ v_new).contiguous()                        # (B,Hv,d,d)

    # slice output back to T positions, transpose to (B,T,Hv,d)
    o = o[:, :, :T, :].permute(0, 2, 1, 3).contiguous()                               # (B,T,Hv,d)
    o = write_after(self.ssm_state, new_state, o)

    # gated output: RMSNorm(o) * SiLU(z), then project + residual + FFN
    h = x + self.ssm_out((self.ssm_norm(o) * z.silu()).reshape(B, T, -1))              # (B,T,D)
    h_norm = self.ffn_norm(h)
    return (h + self.ffn_down(self.ffn_gate(h_norm).silu().contiguous() * self.ffn_up(h_norm))).contiguous()

  def __call__(self, x:Tensor, start_pos:int|UOp):
    self._ensure_states(x)
    return self._rollout(x, start_pos)

  def call_prefill(self, x:Tensor, start_pos:int|UOp):
    self._ensure_states(x)
    return self._prefill(x, start_pos)

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, blk:list|None=None):
    self.blk = blk if blk is not None else [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context,
                                                              qk_norm, num_experts, num_experts_per_tok) for _ in range(num_blocks)]
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)
    self.max_context = max_context
    self._cached_tokens: list[int] = []
    self._generated_ids: set[int] = set()
    self._is_hybrid = blk is not None and any(isinstance(b, GatedDeltaNetBlock) for b in (blk or []))
    # sampling parameters (defaults: greedy; set via configure_sampling)
    self.temperature, self.top_k, self.top_p, self.presence_penalty = 0, 0, 1.0, 0
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward_prefill if self._is_hybrid else self.forward)
    self.rollout_jit = TinyJit(self.forward if self._is_hybrid else self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # TODO: add temperature
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def _sample(self, logits:Tensor) -> Tensor:
    if self.temperature == 0: return logits.softmax(-1, dtype="float").argmax(-1, keepdim=True)
    # scaled sampling with top-k + top-p (Qwen3.5 recommended: temp=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)
    logits = logits / self.temperature
    if self.presence_penalty != 0:
      for tid in self._generated_ids: logits[:, tid] -= self.presence_penalty
    if self.top_k > 0:
      val = logits.topk(self.top_k, dim=-1)[0]
      logits = logits.where(logits >= val[:, -1:], -float('inf'))
    probs = logits.softmax(-1, dtype="float")
    if self.top_p < 1.0:
      sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
      cumsum = sorted_probs.cumsum(axis=-1)
      sorted_probs = sorted_probs.where(cumsum - sorted_probs <= self.top_p, 0.0)
      probs = sorted_probs.gather(-1, sorted_idx.argsort(-1))
      probs = probs / probs.sum(axis=-1, keepdim=True)
    return probs.multinomial(1)

  def forward_prefill(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk:
      x = block.call_prefill(x, start_pos) if isinstance(block, GatedDeltaNetBlock) else block(x, start_pos)
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return (self.prefill_jit if resolve(tokens.shape[1] != 1) else self.rollout_jit)(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    kv, state_dict = nn.state.gguf_load(gguf.to(None).realize())

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    # Permute Q/K weights from interleaved to half-split RoPE layout (llama-style models only)
    if arch == 'llama':
      for name in state_dict:
        if 'attn_q.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_heads, two=2)
        if 'attn_k.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_kv_heads, two=2)

    dim, hidden_dim = kv[f'{arch}.embedding_length'], kv.get(f'{arch}.expert_feed_forward_length', kv[f'{arch}.feed_forward_length'])
    norm_eps = kv[f'{arch}.attention.layer_norm_rms_epsilon']
    head_dim = kv.get(f'{arch}.attention.key_length', dim // n_heads)
    rope_theta = kv[f'{arch}.rope.freq_base']
    qk_norm = next((int(v.shape[0]) for k, v in state_dict.items() if k.endswith('.attn_q_norm.weight')), 0)

    if arch == 'qwen35':
      full_attn_interval, num_blocks = kv[f'{arch}.full_attention_interval'], kv[f'{arch}.block_count']
      ssm_n_k_heads, ssm_head_dim, conv_kernel = kv[f'{arch}.ssm.group_count'], kv[f'{arch}.ssm.state_size'], kv[f'{arch}.ssm.conv_kernel']
      ssm_inner = kv[f'{arch}.ssm.inner_size']
      ssm_key_dim = ssm_n_k_heads * ssm_head_dim
      ssm_n_v_heads = (ssm_inner - ssm_key_dim * 2) // ssm_head_dim   # conv_dim = key_dim*2 + value_dim → value_dim = inner - key_dim*2... no
      # GGUF: inner_size = value_dim (out_proj input), group_count = n_k_heads, state_size = head_dim
      # value_dim = inner_size = 4096, key_dim = group_count * state_size = 16*128 = 2048
      # n_v_heads = value_dim / state_size = 4096 / 128 = 32
      ssm_n_v_heads = ssm_inner // ssm_head_dim
      rope_dim = kv[f'{arch}.rope.dimension_count']
      blk: list = []
      for i in range(num_blocks):
        if (i + 1) % full_attn_interval == 0:
          blk.append(TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context,
                                      qk_norm=qk_norm, rope_dim=rope_dim, attn_gate=True))
        else:
          blk.append(GatedDeltaNetBlock(dim, hidden_dim, norm_eps, ssm_n_k_heads, ssm_n_v_heads, ssm_head_dim, conv_kernel))
      state_dict = {k.replace('post_attention_norm', 'ffn_norm').replace('ssm_dt.bias', 'ssm_dt').replace('ssm_conv1d.weight', 'ssm_conv1d'):
                    v for k, v in state_dict.items()}
    else:
      blk = None

    model = Transformer(num_blocks=kv[f'{arch}.block_count'], dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                        norm_eps=norm_eps, vocab_size=len(kv['tokenizer.ggml.tokens']), head_dim=head_dim, rope_theta=rope_theta,
                        max_context=max_context, qk_norm=qk_norm,
                        num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0), blk=blk)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]):
    return sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))

  def generate(self, tokens:list[int], chunk_size:int=32):
    if self._is_hybrid: chunk_size = min(chunk_size, CHUNK_SIZE)
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size) if chunk_size > 1 else None
    # assign all input tokens once, then slice from start_pos for the model call
    t_list = tokens + [0] * (self.max_context - len(tokens))
    t = Tensor(t_list, dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the kv cache
    start_pos = self.get_start_pos(tokens)
    out = None
    while len(tokens) < self.max_context:
      sp = v_start_pos.bind(start_pos)
      if self._is_hybrid:
        # hybrid: always process 1 token via rollout_jit (prefill exists but graph is too large, see PREFILL_OPTIMIZATION.md)
        if v_toks is not None:
          out = self.rollout_jit(t[:, sp:sp+v_toks.bind(1)], sp).realize()
        else:
          out = self.rollout_jit(t[:, sp:sp+1], sp).realize()
        start_pos += 1
      elif v_toks is not None:
        nt = v_toks.bind(min(chunk_size, len(tokens) - start_pos))
        out = self(t[:, sp:sp+nt] if out is None else out, sp).realize()
        start_pos += nt.val
      else:
        out = self(t[:, sp:sp+1], sp).realize()
        start_pos += 1
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tok_id = int(out.item())
      tokens.append(tok_id)
      self._generated_ids.add(tok_id)
      self._cached_tokens = tokens[:]
      yield tok_id

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_0.gguf",
  "qwen3.5:27b": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
}

# *** simple OpenAI compatible server on 11434 to match ollama ***
# OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama uvx --from gpt-command-line gpt

CHAT_HTML = b'''<!DOCTYPE html><html><head><title>tinygrad chat</title><style>
  * { margin: 0 }
  body { background: #212121; color: #e3e3e3; font-family: system-ui;
         height: 100vh; display: flex; flex-direction: column }
  #chat { flex: 1; overflow-y: auto; padding: 20px }
  .msg { padding: 10px 16px; margin: 8px 0; white-space: pre-wrap; border-radius: 18px }
  .user { background: #2f2f2f; margin-left: auto; width: fit-content; max-width: 70% }
  #input { max-width: 768px; width: 100%; margin: 20px auto; padding: 14px 20px;
           background: #2f2f2f; color: inherit; font: inherit;
           border: none; outline: none; resize: none; border-radius: 24px; field-sizing: content }
</style></head><body><div id="chat"></div>
<textarea id="input" rows="1" placeholder="Ask anything" autofocus></textarea>
<script>
  input.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); send() } }
  const msgs = [];
  async function send() {
    if (!input.value.trim()) return;
    msgs.push({role: 'user', content: input.value.trim()});
    chat.innerHTML += '<div class="msg user">' + input.value.trim().replace(/</g, '&lt;') + '</div>';
    input.value = '';
    const d = document.createElement('div'); d.className = 'msg'; chat.appendChild(d);
    const r = await fetch('/v1/chat/completions', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model: 'llama', messages: msgs, stream: true})});
    for (const rd = r.body.getReader(), dec = new TextDecoder();;) {
      const {done, value} = await rd.read();
      if (done) break;
      for (const ln of dec.decode(value).split('\\n'))
        if (ln.startsWith('data: ') && !ln.includes('[DONE]'))
          try { d.textContent += JSON.parse(ln.slice(6)).choices[0]?.delta?.content || '' } catch {}
      chat.scrollTop = chat.scrollHeight;
    }
    msgs.push({role: 'assistant', content: d.textContent});
  }
</script></body></html>'''

class Handler(HTTPRequestHandler):
  def log_request(self, code='-', size='-'): pass
  def do_GET(self): self.send_data(CHAT_HTML, content_type="text/html")
  def run_model(self, ids:list[int], model_name:str, include_usage=False):
    cache_start_pos = model.get_start_pos(ids)
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{len(ids)-cache_start_pos:5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    st = time.perf_counter()
    for next_id in model.generate(ids):
      if len(out) == 0: stderr_log(f"prefill:{(len(ids)-cache_start_pos)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id == eos_id: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"stop"}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    stderr_log(f"gen:{len(out)/(time.perf_counter()-pt):4.0f} tok/s  {colored('--', 'BLACK')}  out:{len(out):5d}\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
      # apply sampling params from request
      model.temperature = body.get("temperature", model.temperature)
      model.top_p = body.get("top_p", model.top_p)
      model.top_k = body.get("top_k", model.top_k)
      model.presence_penalty = body.get("presence_penalty", model.presence_penalty)
      model._generated_ids.clear()

      # extract tokens
      ids: list[int] = [bos_id] if bos_id is not None else []
      for msg in body["messages"]:
        ids += tok.role(msg["role"])
        # content can be a str or a list
        content = msg["content"]
        if isinstance(content, str): ids += tok.encode(content)
        elif isinstance(content, list):
          for c in content:
            if c["type"] == "text": ids += tok.encode(c["text"])
            else: raise RuntimeError(f"unhandled type: {c['type']}")
        else: raise RuntimeError(f"unknown content type: {type(content)}")
        ids += tok.end_turn(eos_id)
      ids += tok.role("assistant")

      # reply
      chunks = self.run_model(ids, body["model"], not body.get("stream") or body.get("stream_options",{}).get("include_usage", False))
      if body.get("stream"): self.stream_json(chunks)
      else:
        out = []
        for c in chunks: out.append(c["choices"][0]["delta"].get("content", "") if c["choices"] else "")
        self.send_data(json.dumps({**c, "object":"chat.completion",
          "choices":[{"index":0, "message":{"role":"assistant","content":"".join(out)}, "finish_reason":"stop"}]}).encode())
    else:
      raise RuntimeError(f"unhandled path {self.path}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", choices=list(models.keys()), default=list(models.keys())[0], help="Model choice")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature (0=greedy, Qwen3.5 recommends 1.0)")
  parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0=disabled, Qwen3.5 recommends 20)")
  parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling (1.0=disabled, Qwen3.5 recommends 0.95)")
  parser.add_argument("--presence_penalty", type=float, default=0, help="Presence penalty (Qwen3.5 recommends 1.5 for thinking mode)")
  parser.add_argument("--think", action="store_true", help="Enable thinking mode (inject <think> after assistant header)")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI compatible API (optional port, default 11434)")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  raw_model = Tensor.from_url(models[args.model])
  model, kv = Transformer.from_gguf(raw_model, args.max_context)
  if DEBUG >= 1 or args.benchmark:
    print(f"using model {args.model} with {raw_model.nbytes():,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")
  del raw_model

  # TODO: why this is required to free the RAM of the GGUF copy?
  import gc
  gc.collect()

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  # apply sampling parameters
  model.temperature, model.top_k, model.top_p = args.temperature, args.top_k, args.top_p
  model.presence_penalty = args.presence_penalty

  # do benchmark
  if args.benchmark:
    gen = model.generate(toks:=[bos_id or 0])
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")): next(gen)
    exit(0)

  # start server
  if args.serve:
    # warmup: run 2 tokens through the model twice to capture the JIT before serving
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))
    TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  # interactive chat
  ids: list[int] = [bos_id] if bos_id is not None else []
  think_prefix = tok.encode("<think>\n") if args.think else []
  while 1:
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.role("assistant") + think_prefix
    except EOFError:
      break
    model._generated_ids.clear()
    for next_id in model.generate(ids):
      sys.stdout.write(tok.decode([next_id]) if next_id != eos_id else "\n\n")
      sys.stdout.flush()
      if next_id == eos_id: break
