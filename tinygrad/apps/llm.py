from __future__ import annotations
import sys, argparse, typing, re, unicodedata, json, uuid, time, functools, math
from tinygrad import Tensor, nn, UOp, TinyJit, getenv, function, dtypes
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored
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
    if self.preset in ('qwen2', 'qwen35'): return self.encode("<|im_start|>" + role + "\n")
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

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def l2norm(x:Tensor, dim:int=-1, eps:float=1e-6) -> Tensor:
  return x * (x * x).sum(dim, keepdim=True).add(eps).rsqrt()

class GatedDeltaNetBlock:
  """Gated DeltaNet block: linear-time recurrence with delta rule, replacing attention in hybrid models."""
  def __init__(self, dim:int, hidden_dim:int, norm_eps:float, num_v_heads:int, num_k_heads:int,
               head_k_dim:int, head_v_dim:int, d_conv:int, num_experts:int=0, num_experts_per_tok:int=0,
               shared_hidden_dim:int=0, linear=nn.Linear, expert_weights_cls=None):
    self.num_v_heads = num_v_heads
    self.num_k_heads = num_k_heads
    self.head_k_dim = head_k_dim
    self.head_v_dim = head_v_dim
    self.d_conv = d_conv
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim
    self.key_dim, self.value_dim, self.conv_dim = key_dim, value_dim, conv_dim
    self.gqa_factor = num_v_heads // num_k_heads

    # Input projections
    self.attn_qkv = linear(dim, conv_dim, bias=False)        # Q+K+V (goes through conv1d)
    self.attn_gate = linear(dim, value_dim, bias=False)      # z gate (does NOT go through conv1d)
    self.ssm_beta = linear(dim, num_v_heads, bias=False)     # update rate -> sigmoid
    self.ssm_alpha = linear(dim, num_v_heads, bias=False)    # decay input -> softplus

    # SSM parameters
    self.ssm_a = Tensor.zeros(num_v_heads)                   # -exp(A_log), stored post-negation in GGUF
    self.ssm_dt = Tensor.zeros(num_v_heads)                  # dt_bias (stored as ssm_dt.bias in GGUF)
    self.ssm_conv1d = Tensor.zeros(conv_dim, d_conv)         # depthwise conv kernel
    self.ssm_norm = nn.RMSNorm(head_v_dim, norm_eps)         # gated output norm
    self.ssm_out = linear(value_dim, dim, bias=False)        # output projection

    # Norms: attn_norm = pre-SSM, post_attention_norm = pre-FFN (no separate ffn_norm in Qwen3.5)
    self.attn_norm = nn.RMSNorm(dim, norm_eps)
    self.post_attention_norm = nn.RMSNorm(dim, norm_eps)

    # FFN: MoE or dense (shared between SSM and attention blocks)
    if num_experts == 0:
      self.ffn_gate = linear(dim, hidden_dim, bias=False)
      self.ffn_up = linear(dim, hidden_dim, bias=False)
      self.ffn_down = linear(hidden_dim, dim, bias=False)
    elif num_experts > 0:
      _ew = expert_weights_cls if expert_weights_cls is not None else ExpertWeights
      self.num_experts_per_tok = num_experts_per_tok
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)
      self.ffn_gate_exps = _ew(num_experts, dim, hidden_dim)
      self.ffn_up_exps = _ew(num_experts, dim, hidden_dim)
      self.ffn_down_exps = _ew(num_experts, hidden_dim, dim)
      if shared_hidden_dim > 0:
        self.ffn_gate_shexp = linear(dim, shared_hidden_dim, bias=False)
        self.ffn_up_shexp = linear(dim, shared_hidden_dim, bias=False)
        self.ffn_down_shexp = linear(shared_hidden_dim, dim, bias=False)
        self.ffn_gate_inp_shexp = Tensor.zeros(1)  # scalar sigmoid gate

  def _deltanet_recurrent(self, q:Tensor, k:Tensor, v:Tensor, g:Tensor, beta:Tensor) -> Tensor:
    """Single-token autoregressive delta rule recurrence. Uses uop.assign pattern (no realize).
    q,k: (B, H, 1, d_k), v: (B, H, 1, d_v), g: (B, H, 1), beta: (B, H, 1)
    """
    B = q.shape[0]
    q_t = q[:, :, 0] * (self.head_k_dim ** -0.5)
    k_t = k[:, :, 0]
    v_t = v[:, :, 0]
    g_t = g[:, :, 0].exp().reshape(B, self.num_v_heads, 1, 1)
    beta_t = beta[:, :, 0].reshape(B, self.num_v_heads, 1)

    # Decay, delta update, and assign in one step
    S_decayed = self.ssm_state * g_t
    kv_mem = (S_decayed * k_t.unsqueeze(-1)).sum(-2)
    delta = (v_t - kv_mem) * beta_t
    S_new = S_decayed + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

    assigned = self.ssm_state.uop.after(self.ssm_state.uop.assign(S_new.contiguous().uop))
    S_final = Tensor(assigned, device=assigned.device)
    y = (S_final * q_t.unsqueeze(-1)).sum(-2)
    return y.unsqueeze(2)

  def _conv1d_step(self, xBC:Tensor) -> Tensor:
    """Single-token conv1d update using rolling buffer. Uses uop.assign pattern."""
    new_state = self.conv_state[:, :, 1:].cat(xBC.unsqueeze(-1), dim=-1)
    assigned = self.conv_state.uop.after(self.conv_state.uop.assign(new_state.contiguous().uop))
    state = Tensor(assigned, device=assigned.device)
    x = (state * self.ssm_conv1d.unsqueeze(0)).sum(-1)
    return x.silu()

  def _conv1d_prefill(self, xBC:Tensor) -> Tensor:
    """Prefill conv1d. xBC: (B, T, conv_dim). Uses sliding window over padded input."""
    B, T, C = xBC.shape
    x = xBC.permute(0, 2, 1)  # (B, C, T)
    x_padded = Tensor.zeros(B, C, self.d_conv - 1, device=x.device).cat(x, dim=-1)
    # Depthwise conv via sliding window
    out_cols = []
    for t in range(T):
      window = x_padded[:, :, t:t+self.d_conv]
      out_cols.append((window * self.ssm_conv1d.unsqueeze(0)).sum(-1))
    out = Tensor.stack(*out_cols, dim=-1)  # (B, C, T)
    return out.permute(0, 2, 1).silu()

  def _deltanet_prefill(self, x:Tensor) -> Tensor:
    """Prefill path (sequential scan, NOT inside @function). Called from __call__."""
    x_norm = self.attn_norm(x)
    B, T, D = x.shape
    qkv_raw = self.attn_qkv(x_norm)  # pre-conv QKV (kept for conv state)
    z = self.attn_gate(x_norm)
    b = self.ssm_beta(x_norm)
    a = self.ssm_alpha(x_norm)

    qkv = self._conv1d_prefill(qkv_raw)
    q, k, v = qkv[:, :, :self.key_dim], qkv[:, :, self.key_dim:self.key_dim*2], qkv[:, :, self.key_dim*2:]
    q = q.reshape(B, T, self.num_k_heads, self.head_k_dim).transpose(1, 2)
    k = k.reshape(B, T, self.num_k_heads, self.head_k_dim).transpose(1, 2)
    v = v.reshape(B, T, self.num_v_heads, self.head_v_dim).transpose(1, 2)

    beta = b.sigmoid().transpose(1, 2)
    g = (self.ssm_a * (a + self.ssm_dt).softplus()).transpose(1, 2)

    if self.gqa_factor > 1:
      q = q.repeat_interleave(self.gqa_factor, dim=1)
      k = k.repeat_interleave(self.gqa_factor, dim=1)
    q, k = l2norm(q, dim=-1), l2norm(k, dim=-1)

    # Sequential scan (not inside @function so realize works)
    scale = self.head_k_dim ** -0.5
    q = q * scale
    S = Tensor.zeros(B, self.num_v_heads, self.head_k_dim, self.head_v_dim, device=q.device)
    outputs = []
    for t in range(T):
      q_t, k_t, v_t = q[:, :, t], k[:, :, t], v[:, :, t]
      g_t = g[:, :, t].exp().reshape(B, self.num_v_heads, 1, 1)
      beta_t = beta[:, :, t].reshape(B, self.num_v_heads, 1)
      S = S * g_t
      kv_mem = (S * k_t.unsqueeze(-1)).sum(-2)
      delta = (v_t - kv_mem) * beta_t
      S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
      outputs.append((S * q_t.unsqueeze(-1)).sum(-2))

    # Save final state and conv state for subsequent autoregressive
    self.ssm_state.assign(S.contiguous()).realize()
    conv_input = qkv_raw.permute(0, 2, 1)  # (B, C, T)
    if T < self.d_conv:
      conv_input = Tensor.zeros(B, self.conv_dim, self.d_conv - T, device=x.device).cat(conv_input, dim=-1)
    else:
      conv_input = conv_input[:, :, -self.d_conv:]
    self.conv_state.assign(conv_input.contiguous()).realize()

    y = Tensor.stack(*outputs, dim=2).transpose(1, 2).reshape(B, T, self.value_dim)
    z_r = z.reshape(B * T, self.num_v_heads, self.head_v_dim)
    y_r = y.reshape(B * T, self.num_v_heads, self.head_v_dim)
    y_normed = (self.ssm_norm(y_r) * z_r.silu()).reshape(B, T, self.value_dim)
    return x + self.ssm_out(y_normed)

  @function
  def _deltanet_step(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    """Single-token autoregressive path (inside @function for JIT)."""
    x_norm = self.attn_norm(x)
    B = x.shape[0]
    qkv = self.attn_qkv(x_norm).reshape(B, -1)
    z = self.attn_gate(x_norm)
    b = self.ssm_beta(x_norm)
    a = self.ssm_alpha(x_norm)

    # Conv1d step
    qkv = self._conv1d_step(qkv).unsqueeze(1)  # (B, 1, conv_dim)
    q, k, v = qkv[:, :, :self.key_dim], qkv[:, :, self.key_dim:self.key_dim*2], qkv[:, :, self.key_dim*2:]

    q = q.reshape(B, 1, self.num_k_heads, self.head_k_dim).transpose(1, 2)
    k = k.reshape(B, 1, self.num_k_heads, self.head_k_dim).transpose(1, 2)
    v = v.reshape(B, 1, self.num_v_heads, self.head_v_dim).transpose(1, 2)

    beta = b.sigmoid().transpose(1, 2)
    g = (self.ssm_a * (a + self.ssm_dt).softplus()).transpose(1, 2)

    if self.gqa_factor > 1:
      q = q.repeat_interleave(self.gqa_factor, dim=1)
      k = k.repeat_interleave(self.gqa_factor, dim=1)
    q, k = l2norm(q, dim=-1), l2norm(k, dim=-1)

    y = self._deltanet_recurrent(q, k, v, g, beta)
    y = y.transpose(1, 2).reshape(B, 1, self.value_dim)

    z_r = z.reshape(B, self.num_v_heads, self.head_v_dim)
    y_r = y.reshape(B, self.num_v_heads, self.head_v_dim)
    y_normed = (self.ssm_norm(y_r) * z_r.silu()).reshape(B, 1, self.value_dim)
    return x + self.ssm_out(y_normed)

  @function
  def _feed_forward(self, h:Tensor) -> Tensor:
    h_norm = self.post_attention_norm(h)
    if hasattr(self, 'ffn_gate_exps'):
      x = h_norm.unsqueeze(2)
      probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)
      x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))
      moe_out = (x_down * probs.unsqueeze(-1)).sum(axis=2)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp_out = self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
        if hasattr(self, 'ffn_gate_inp_shexp'):
          shexp_out = shexp_out * self.ffn_gate_inp_shexp.sigmoid()
        moe_out = moe_out + shexp_out
      return h + moe_out
    if hasattr(self, 'ffn_gate'):
      return h + self.ffn_down(self.ffn_gate(h_norm).silu() * self.ffn_up(h_norm))
    raise NotImplementedError("GatedDeltaNetBlock requires MoE or dense FFN")

  def __call__(self, x:Tensor, start_pos:int|UOp):
    if not hasattr(self, '_state_init'):
      B = x.shape[0]
      self.conv_state = Tensor.zeros(B, self.conv_dim, self.d_conv, device=x.device).contiguous().realize()
      self.ssm_state = Tensor.zeros(B, self.num_v_heads, self.head_k_dim, self.head_v_dim, device=x.device).contiguous().realize()
      self._state_init = True
    T = x.shape[1]
    if T == 1:
      h = self._deltanet_step(x, start_pos)
    else:
      h = self._deltanet_prefill(x)
    return self._feed_forward(h).contiguous()

class TransformerBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, n_kv_heads:int, norm_eps:float, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0,
               shared_hidden_dim:int=0, linear=nn.Linear, expert_weights_cls=None, gated_attn:bool=False, rope_dim:int=0):
    self.n_heads      = n_heads
    self.n_kv_heads   = n_kv_heads
    self.head_dim     = head_dim
    self.rope_dim     = rope_dim if rope_dim > 0 else head_dim  # partial RoPE: only rotate first rope_dim dims
    self.rope_theta   = rope_theta
    self.max_context  = max_context
    self.qk_norm      = qk_norm
    self.gated_attn   = gated_attn

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = self.head_dim * n_heads
    kv_proj_out      = self.head_dim * n_kv_heads
    self.attn_q      = linear(dim, q_proj_out * (2 if gated_attn else 1),  bias=False)
    self.attn_k      = nn.Linear(dim, kv_proj_out, bias=False)  # K/V are small, always dense (often Q8_0 not Q4_0)
    self.attn_v      = nn.Linear(dim, kv_proj_out, bias=False)
    self.attn_output = linear(q_proj_out, dim,  bias=False)

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(dim, norm_eps)
    # Qwen3.5 GGUF uses post_attention_norm, llama uses ffn_norm
    if gated_attn: self.post_attention_norm = nn.RMSNorm(dim, norm_eps)
    else: self.ffn_norm = nn.RMSNorm(dim, norm_eps)
    if qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(qk_norm, norm_eps), nn.RMSNorm(qk_norm, norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if num_experts > 0:
      _ew = expert_weights_cls if expert_weights_cls is not None else ExpertWeights
      self.num_experts_per_tok = num_experts_per_tok
      self.ffn_gate_inp = nn.Linear(dim, num_experts, bias=False)  # router always dense
      self.ffn_gate_exps = _ew(num_experts, dim, hidden_dim)
      self.ffn_up_exps = _ew(num_experts, dim, hidden_dim)
      self.ffn_down_exps = _ew(num_experts, hidden_dim, dim)
      if shared_hidden_dim > 0:
        self.ffn_gate_shexp = linear(dim, shared_hidden_dim, bias=False)
        self.ffn_up_shexp = linear(dim, shared_hidden_dim, bias=False)
        self.ffn_down_shexp = linear(shared_hidden_dim, dim, bias=False)
        self.ffn_gate_inp_shexp = Tensor.zeros(1)  # scalar sigmoid gate
    else:
      self.ffn_gate    = linear(dim, hidden_dim, bias=False)
      self.ffn_up      = linear(dim, hidden_dim, bias=False)
      self.ffn_down    = linear(hidden_dim, dim, bias=False)

  @function
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    x_norm = self.attn_norm(x)                       # (B,T,D)
    q_out, k, v = self.attn_q(x_norm), self.attn_k(x_norm), self.attn_v(x_norm)

    B, T, _ = x.shape
    if self.gated_attn:
      # Gated attention: attn_q outputs Q and gate concatenated (2× width)
      q_out = q_out.reshape(B, T, self.n_heads, self.head_dim * 2)
      q, gate = q_out[:, :, :, :self.head_dim], q_out[:, :, :, self.head_dim:]  # each (B,T,H,Hd)
      gate = gate.reshape(B, T, -1)  # (B,T,H*Hd) for later application
      q = q.transpose(1, 2)  # (B,H,T,Hd)
    else:
      q = q_out.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Hd)

    if self.qk_norm and self.qk_norm != self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)
    k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.qk_norm == self.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    freqs_cis = precompute_freqs_cis(self.rope_dim, self.max_context, self.rope_theta)[start_pos:start_pos+T]
    if self.rope_dim < self.head_dim:
      # Partial RoPE: only rotate first rope_dim dimensions, pass through the rest
      q_rot, q_pass = q[..., :self.rope_dim], q[..., self.rope_dim:]
      k_rot, k_pass = k[..., :self.rope_dim], k[..., self.rope_dim:]
      q = apply_rope(q_rot, freqs_cis).cat(q_pass, dim=-1)
      k = apply_rope(k_rot, freqs_cis).cat(k_pass, dim=-1)
    else:
      q = apply_rope(q, freqs_cis)
      k = apply_rope(k, freqs_cis)

    # TODO: fix assign to behave like this
    assigned_kv = self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.assign(Tensor.stack(k, v).contiguous().uop))
    tensor_assigned_kv = Tensor(assigned_kv, device=assigned_kv.device)
    k = tensor_assigned_kv[0, :, :, 0:start_pos+T, :]
    v = tensor_assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, device=x.device).triu(int(start_pos)+1) if T > 1 else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    if self.gated_attn: attn = attn * gate.sigmoid()
    attn = self.attn_output(attn)
    return x + attn

  @function
  def _feed_forward(self, h: Tensor) -> Tensor:
    h_norm = (self.post_attention_norm if self.gated_attn else self.ffn_norm)(h)
    if hasattr(self, 'ffn_gate_exps'):
      x = h_norm.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      probs, sel = self.ffn_gate_inp(h_norm).softmax(-1).topk(self.num_experts_per_tok)  # (B, T, k) each
      x_down = self.ffn_down_exps(sel, self.ffn_gate_exps(sel, x).silu() * self.ffn_up_exps(sel, x))  # (B, T, k, D)
      moe_out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      # Add shared expert if present
      if hasattr(self, 'ffn_gate_shexp'):
        shexp_out = self.ffn_down_shexp(self.ffn_gate_shexp(h_norm).silu() * self.ffn_up_shexp(h_norm))
        if hasattr(self, 'ffn_gate_inp_shexp'):
          shexp_out = shexp_out * self.ffn_gate_inp_shexp.sigmoid()
        moe_out = moe_out + shexp_out
      return h + moe_out
    # TODO: remove the need for this contiguous
    gated  = self.ffn_gate(h_norm).silu().contiguous() * self.ffn_up(h_norm)
    return h + self.ffn_down(gated)

  def __call__(self, x: Tensor, start_pos: int|UOp):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      self.cache_kv = Tensor.zeros(2, x.shape[0], self.n_kv_heads, self.max_context, self.head_dim, device=x.device).contiguous().realize()
    return self._feed_forward(self._attention(x, start_pos)).contiguous()

class Transformer:
  def __init__(self, *, num_blocks, dim, hidden_dim, n_heads, n_kv_heads, norm_eps, vocab_size, head_dim:int, rope_theta:float,
               max_context:int=0, qk_norm:int=0, num_experts:int=0, num_experts_per_tok:int=0, shared_hidden_dim:int=0,
               linear=nn.Linear, expert_weights_cls=None, blocks:list|None=None):
    if blocks is not None:
      self.blk = blocks  # pre-built blocks (for hybrid architectures)
    else:
      self.blk = [TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta, max_context, qk_norm,
                                   num_experts, num_experts_per_tok, shared_hidden_dim, linear, expert_weights_cls) for _ in range(num_blocks)]
    self.token_embd  = nn.Embedding(vocab_size, dim)
    self.output_norm = nn.RMSNorm(dim, norm_eps)
    self.output = nn.Linear(dim, vocab_size, bias=False)  # lm_head is often Q6_K, not Q4_0
    self.max_context = max_context
    # JIT is used if T=1 and start_pos is a UOp. TODO: make this not needed by including T in the JIT and making start_pos always a UOp
    self.forward_jit = TinyJit(self.forward)

  def forward(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens)                           # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    # TODO: add temperature
    return self.output(self.output_norm(x))[:, -1, :].softmax(-1, dtype="float").argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp=0) -> Tensor:
    return (self.forward_jit if getenv("JIT", 1) and tokens.shape[1] == 1 and isinstance(start_pos, UOp) else self.forward)(tokens, start_pos)

  @staticmethod
  def from_gguf(gguf:Tensor, max_context:int|None=None, realize=bool(getenv("REALIZE", 1))) -> tuple[Transformer, dict]:
    # Try Q4_0-aware loader for memory-efficient loading of quantized models
    try:
      from q4_0_linear import Q4_0Linear, Q4_0ExpertWeights, gguf_load_q4_0, dequant_q4_0_blocks, tensor_to_q4_0_blocks
      kv, state_dict, tensor_info = gguf_load_q4_0(gguf)
      has_q4_0 = any(t[0] == 2 for t in tensor_info.values())
    except ImportError:
      kv, state_dict = nn.state.gguf_load(gguf)
      has_q4_0, tensor_info = False, {}

    if has_q4_0:
      linear, expert_weights_cls = Q4_0Linear, Q4_0ExpertWeights  # expert weights stored as Q4_0 raw blocks, dequant only selected experts
    else:
      linear, expert_weights_cls = nn.Linear, None

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']
    vocab_size = len(kv['tokenizer.ggml.tokens'])
    dim = kv[f'{arch}.embedding_length']
    num_blocks = kv[f'{arch}.block_count']
    hidden_dim = kv.get(f'{arch}.expert_feed_forward_length') or kv[f'{arch}.feed_forward_length']
    norm_eps = kv[f'{arch}.attention.layer_norm_rms_epsilon']
    num_experts = kv.get(f'{arch}.expert_count', 0)
    num_experts_per_tok = kv.get(f'{arch}.expert_used_count', 0)
    # Shared expert hidden dim: feed_forward_length when expert_feed_forward_length is also present
    shared_hidden_dim = kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.expert_feed_forward_length') else 0
    head_dim = kv.get(f'{arch}.attention.key_length', dim // n_heads)
    rope_theta = kv[f'{arch}.rope.freq_base']
    rope_dim = kv.get(f'{arch}.rope.dimension_count', 0)  # partial RoPE: 0 = use full head_dim
    # Detect QK norm — for hybrid models, check the first attention block (not block 0 which may be DeltaNet)
    _fai = kv.get(f'{arch}.full_attention_interval', 0)
    first_attn_blk = _fai - 1 if _fai > 0 else 0
    qk_norm_key = f'blk.{first_attn_blk}.attn_q_norm.weight'
    qk_norm = int(state_dict[qk_norm_key].shape[0]) if qk_norm_key in state_dict else 0

    # Handle tied output weight — both embedding and output are dense (nn.Linear)
    if 'output.weight' not in state_dict:
      if has_q4_0 and tensor_info.get('token_embd.weight', (None,))[0] == 2:
        dequanted = dequant_q4_0_blocks(state_dict['token_embd.weight'], vocab_size, dim)
        state_dict['token_embd.weight'] = dequanted
        state_dict['output.weight'] = dequanted
      else:
        state_dict['output.weight'] = state_dict['token_embd.weight']

    # Ensure token_embd is always dense (embedding needs random access per token, not Q4_0 matmul)
    if has_q4_0 and tensor_info.get('token_embd.weight', (None,))[0] == 2 and state_dict['token_embd.weight'].dtype == dtypes.uint8:
      state_dict['token_embd.weight'] = dequant_q4_0_blocks(state_dict['token_embd.weight'], vocab_size, dim)

    # Detect hybrid architecture (DeltaNet + attention)
    is_hybrid = kv.get(f'{arch}.full_attention_interval', 0) > 0
    full_attn_interval = kv.get(f'{arch}.full_attention_interval', 0)

    # Dequant Q4_0 blocks for layers that must stay dense
    _q4_0_layer_keys = ('attn_q.', 'attn_output.', 'attn_qkv.', 'attn_gate.', 'ffn_gate_exps.', 'ffn_up_exps.', 'ffn_down_exps.',
                        'ssm_alpha.', 'ssm_beta.', 'ssm_out.',
                        'ffn_gate.', 'ffn_up.', 'ffn_down.',
                        'ffn_gate_shexp.', 'ffn_up_shexp.', 'ffn_down_shexp.')
    if has_q4_0:
      for name in list(state_dict.keys()):
        if name == 'token_embd.weight': continue
        info = tensor_info.get(name)
        if info and info[0] == 2:
          is_quantized_layer = any(k in name for k in _q4_0_layer_keys)
          if not is_quantized_layer:
            # Skip if already dequanted by numpy path (dtype won't be uint8)
            if state_dict[name].dtype == dtypes.uint8:
              _, dims = info
              state_dict[name] = dequant_q4_0_blocks(state_dict[name], *reversed(dims))

    # Requant non-Q4_0 tensors that the model expects as Q4_0Linear (e.g. Q8_0 ssm_alpha/beta, Q5_K ssm_out)
    if has_q4_0:
      for name in list(state_dict.keys()):
        if name == 'token_embd.weight': continue
        info = tensor_info.get(name)
        if info and info[0] != 2 and state_dict[name].dtype != dtypes.uint8:
          is_quantized_layer = any(k in name for k in _q4_0_layer_keys)
          if is_quantized_layer:
            state_dict[name] = tensor_to_q4_0_blocks(state_dict[name])

    # Cast non-Q4_0 tensors to float16
    # Native GGUF types (F32/F16) are disk-backed bitcasts — keep as f32. These are small tensors
    # (norms, ssm_a, ssm_dt, routing weights) so the memory overhead is negligible.
    # Dequanted tensors (Q4_0/Q4_1 → f32) also reference disk. The scheduler fuses
    # DISK→dequant→CAST(f32→f16)→COPY into one op that no renderer can handle.
    # Fix: realize f32 on device first (breaks the disk chain), then cast to f16.
    # Process one tensor at a time so peak memory is only one extra f32 buffer.
    if getenv("HALF", 1):
      for k in list(state_dict.keys()):
        v = state_dict[k]
        if v.dtype in (dtypes.uint8, dtypes.float16): continue
        state_dict[k] = v.half()

    # Permute Q/K weights from interleaved to half-split RoPE layout (llama-style models only)
    if arch == 'llama':
      for name in list(state_dict.keys()):
        if has_q4_0 and state_dict[name].dtype == dtypes.uint8: continue
        if 'attn_q.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_heads, two=2)
        if 'attn_k.weight' in name: state_dict[name] = state_dict[name].rearrange("(n h two) d -> (n two h) d", n=n_kv_heads, two=2)

    # Build blocks - either hybrid (DeltaNet + attention) or uniform
    if is_hybrid:
      ssm_cfg = {
        'num_v_heads': kv.get(f'{arch}.ssm.time_step_rank', 32),
        'num_k_heads': kv.get(f'{arch}.ssm.group_count', 16),
        'head_k_dim': 128,  # inferred from ssm.state_size
        'head_v_dim': kv.get(f'{arch}.ssm.state_size', 128),
        'd_conv': kv.get(f'{arch}.ssm.conv_kernel', 4),
      }
      blocks = []
      for i in range(num_blocks):
        if (i + 1) % full_attn_interval == 0:
          # Full attention block (every Nth layer) — Qwen3.5 uses gated attention
          blocks.append(TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads, norm_eps, head_dim, rope_theta,
                                         max_context, qk_norm, num_experts, num_experts_per_tok, shared_hidden_dim,
                                         linear, expert_weights_cls, gated_attn=True, rope_dim=rope_dim))
        else:
          # DeltaNet SSM block
          blocks.append(GatedDeltaNetBlock(dim, hidden_dim, norm_eps, num_experts=num_experts,
                                           num_experts_per_tok=num_experts_per_tok, shared_hidden_dim=shared_hidden_dim,
                                           linear=linear, expert_weights_cls=expert_weights_cls, **ssm_cfg))
      model = Transformer(num_blocks=num_blocks, dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                          norm_eps=norm_eps, vocab_size=vocab_size, head_dim=head_dim, rope_theta=rope_theta,
                          max_context=max_context, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
                          linear=linear, expert_weights_cls=expert_weights_cls, blocks=blocks)
    else:
      model = Transformer(num_blocks=num_blocks, dim=dim, hidden_dim=hidden_dim, n_heads=n_heads, n_kv_heads=n_kv_heads,
                          norm_eps=norm_eps, vocab_size=vocab_size, head_dim=head_dim, rope_theta=rope_theta,
                          max_context=max_context, qk_norm=qk_norm, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok,
                          shared_hidden_dim=shared_hidden_dim, linear=linear, expert_weights_cls=expert_weights_cls)
    # Remap GGUF tensor names for hybrid models
    if is_hybrid:
      remap = {}
      for name in list(state_dict.keys()):
        new_name = name
        # ssm_dt.bias -> ssm_dt (bare tensor, not a module)
        if 'ssm_dt.bias' in name: new_name = name.replace('ssm_dt.bias', 'ssm_dt')
        # ssm_conv1d.weight -> ssm_conv1d (bare tensor)
        if 'ssm_conv1d.weight' in name: new_name = name.replace('ssm_conv1d.weight', 'ssm_conv1d')
        # ssm_a (no suffix) -> maps directly to self.ssm_a
        if new_name != name:
          remap[name] = new_name
      for old, new in remap.items():
        state_dict[new] = state_dict.pop(old)

    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def generate(self, tokens:list[int], start_pos=0):
    v_start_pos = UOp.variable("start_pos", 1, self.max_context-1)
    t = Tensor([tokens[start_pos:]], dtype="int32")
    while len(tokens) < self.max_context:
      t = self(t, v_start_pos.bind(start_pos) if getenv("SYM", 1) and start_pos != 0 and t.shape[-1] == 1 else start_pos)
      next_id = int(t.item())
      tokens.append(next_id)
      start_pos = len(tokens) - 1
      yield next_id

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
<textarea id="input" rows="1" placeholder="Ask anything"></textarea>
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
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  in:{len(ids):5d}  {colored('--', 'BLACK')}  ")
    tmpl = {"id":f"chatcmpl-{uuid.uuid4().hex[:24]}", "object":"chat.completion.chunk", "created":int(time.time()), "model":model_name}
    yield {"choices": [{"index":0, "delta":{"role":"assistant","content":""}, "finish_reason":None}], **tmpl}
    out: list[int] = []
    st = time.perf_counter()
    for next_id in model.generate(ids):
      if len(out) == 0: stderr_log(f"prefill:{len(ids)/((pt:=time.perf_counter())-st):4.0f} tok/s  {colored('--', 'BLACK')}  ")
      if next_id == eos_id: break
      out.append(next_id)
      yield {"choices": [{"index":0, "delta":{"content":tok.decode([next_id])}, "finish_reason":None}], **tmpl}
    yield {"choices": [{"index":0, "delta":{},"finish_reason":"stop"}], **tmpl}
    if include_usage:
      yield {"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}, **tmpl}
    stderr_log(f"out:{len(out):5d}  {colored('--', 'BLACK')}  gen: {len(out)/(time.perf_counter()-pt):4.0f} tok/s\n")

  def do_POST(self):
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8"))
    if DEBUG >= 1: print(json.dumps(body, indent=2))
    if self.path == "/v1/chat/completions":
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
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help="Model name or path to GGUF file")
  parser.add_argument("--max_context", type=int, default=4096, help="Max Context Length")
  parser.add_argument("--serve", nargs='?', type=int, const=11434, metavar="PORT", help="Run OpenAI compatible API (optional port, default 11434)")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  args = parser.parse_args()

  # load the model
  import pathlib, os
  if args.model in models:
    raw_model = Tensor.from_url(models[args.model])
  elif os.path.exists(args.model):
    raw_model = Tensor(pathlib.Path(args.model))
  else:
    parser.error(f"Unknown model '{args.model}'. Available: {', '.join(models.keys())}, or pass a path to a GGUF file.")
  model, kv = Transformer.from_gguf(raw_model, args.max_context)
  if DEBUG >= 1 or args.benchmark:
    print(f"using model {args.model} with {raw_model.nbytes():,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")
  del raw_model

  # TODO: why this is required to free the RAM of the GGUF copy?
  import gc
  gc.collect()

  # do benchmark
  if args.benchmark:
    gen = model.generate([0], 0)
    for _ in range(args.benchmark):
      GlobalCounters.reset()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB"): next(gen)
    exit(0)

  # extract some metadata
  tok = SimpleTokenizer.from_gguf_kv(kv)
  bos_id: int|None = kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None
  eos_id: int = kv['tokenizer.ggml.eos_token_id']

  # start server
  if args.serve: TCPServerWithReuse(('', args.serve), Handler).serve_forever()

  ids: list[int] = [bos_id] if bos_id is not None else []
  while 1:
    start_pos = max(len(ids) - 1, 0)
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn(eos_id) + tok.role("assistant")
    except EOFError:
      break
    for next_id in model.generate(ids, start_pos):
      sys.stdout.write(tok.decode([next_id]) if next_id != eos_id else "\n\n")
      sys.stdout.flush()
      if next_id == eos_id: break
