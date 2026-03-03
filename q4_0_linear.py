"""Q4_0Linear: drop-in nn.Linear replacement storing weights as Q4_0 blocks.

Weights are stored as raw Q4_0 blocks: (n_blocks, 18) uint8 tensor (0.5625 bytes/param).
Dequantization happens lazily in __call__ using tensor ops, so the compiler can fuse it into the matmul kernel.
FP32 intermediates exist only in registers/cache, never materialized in VRAM.

Q4_0 block layout (18 bytes -> 32 elements):
  bytes [0:2]  — fp16 scale (d)
  bytes [2:18] — 16 packed bytes, 2 nibbles each -> 32 x 4-bit unsigned quants
  dequant: float = d * (uint4_value - 8)

Nibble packing order (matches ggml):
  low  nibble of byte[j] = element[j]      (j = 0..15)
  high nibble of byte[j] = element[j + 16]  (j = 0..15)
"""
from __future__ import annotations
import struct, functools, io
from typing import Any, Callable
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import prod, round_up
from tinygrad.nn.state import TensorIO, ggml_data_to_tensor

class Q4_0Linear:
  """Drop-in replacement for nn.Linear that stores weights in Q4_0 format."""

  def __init__(self, in_features: int, out_features: int, bias=False):
    assert bias == False
    assert (in_features * out_features) % 32 == 0, "total weight elements must be divisible by 32"
    n_blocks = (in_features * out_features) // 32
    self.weight = Tensor.zeros(n_blocks, 18, dtype=dtypes.uint8)
    self.in_features = in_features
    self.out_features = out_features

  def __call__(self, x: Tensor) -> Tensor:
    return x.dot(self._dequant_weight().T)

  def _dequant_weight(self) -> Tensor:
    """Lazy dequant: build a tensor graph from Q4_0 blocks to fp32 weight matrix."""
    blocks = self.weight  # (n_blocks, 18)

    # Extract fp16 scales from first 2 bytes of each block -> fp32
    scales = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32)  # (n_blocks, 1)

    # Unpack 4-bit nibbles from the 16 quant bytes
    qbytes = blocks[:, 2:]                  # (n_blocks, 16)
    low  = qbytes.bitwise_and(0x0F)         # (n_blocks, 16) — elements 0..15
    high = qbytes.rshift(4)                 # (n_blocks, 16) — elements 16..31
    nibbles = low.cat(high, dim=-1)         # (n_blocks, 32)

    # Dequantize: float = d * (q - 8)
    w = (nibbles.cast(dtypes.float32) - 8.0) * scales  # (n_blocks, 32)
    return w.reshape(self.out_features, self.in_features)

  def dequantize(self) -> Tensor:
    """Debug: fully dequantize weights to fp32 tensor of shape (out_features, in_features)."""
    return self._dequant_weight()

  @staticmethod
  def from_gguf_raw(raw_data: Tensor, n_elements: int, out_features: int, in_features: int) -> Q4_0Linear:
    """Zero-copy path: takes raw uint8 slice from GGUF, reshapes to (n_blocks, 18).

    Args:
      raw_data: 1-D uint8 tensor containing the raw Q4_0 block bytes.
      n_elements: total number of float elements encoded (must equal out_features * in_features).
      out_features: number of output features (rows of weight matrix).
      in_features: number of input features (cols of weight matrix).
    """
    assert n_elements == out_features * in_features
    assert n_elements % 32 == 0
    n_blocks = n_elements // 32
    layer = object.__new__(Q4_0Linear)
    layer.in_features = in_features
    layer.out_features = out_features
    layer.weight = raw_data[:n_blocks * 18].reshape(n_blocks, 18)
    return layer

  @staticmethod
  def quantize(tensors: dict[str, Tensor], device) -> dict[str, Tensor]:
    """Convert FP32 state_dict entries to Q4_0 blocks. Same pattern as Int8Linear.quantize().

    Quantizes feed_forward and attention weight tensors to Q4_0 format.
    Non-weight tensors are passed through unchanged.
    """
    new_tensors: dict[str, Tensor] = {}
    for name, v in tensors.items():
      if "feed_forward" in name or "attention.w" in name:
        assert "weight" in name, name
        flat = v.cast(dtypes.float32).flatten()
        n = flat.shape[0]
        assert n % 32 == 0, f"weight {name} has {n} elements, not divisible by 32"
        blocks = flat.reshape(-1, 32)  # (n_blocks, 32)

        # Per-block scale: d = amax / 8 (symmetric quantization)
        amax = blocks.abs().max(axis=-1, keepdim=True)  # (n_blocks, 1)
        d = amax / 8.0  # (n_blocks, 1)

        # Quantize: q = clamp(round(v / d + 8), 0, 15)
        safe_d = d + (d == 0.0).cast(dtypes.float32) * 1e-10  # avoid division by zero
        q = (blocks / safe_d + 8.0).round().clip(0, 15).cast(dtypes.uint8)  # (n_blocks, 32)

        # Pack nibbles: low[j] = q[j], high[j] = q[j+16]
        lo = q[:, :16]                        # (n_blocks, 16)
        hi = q[:, 16:]                        # (n_blocks, 16)
        packed = lo.bitwise_or(hi.lshift(4))   # (n_blocks, 16)

        # Scale to fp16 bytes
        d_bytes = d.cast(dtypes.float16).bitcast(dtypes.uint8)  # (n_blocks, 2)

        # Assemble Q4_0 block: [scale(2), packed(16)] = 18 bytes
        q4_block = d_bytes.cat(packed, dim=-1)  # (n_blocks, 18)

        new_tensors[name] = q4_block
        if isinstance(device, tuple):
          new_tensors[name].shard_(device, axis=0)
      else:
        new_tensors[name] = v
    return new_tensors


def dequant_q4_0_blocks(blocks: Tensor, *shape) -> Tensor:
  """Dequant Q4_0 blocks (n_blocks, 18) uint8 to float32 tensor with given shape."""
  scales = blocks[:, :2].bitcast(dtypes.float16).cast(dtypes.float32)
  qbytes = blocks[:, 2:]
  low = qbytes.bitwise_and(0x0F)
  high = qbytes.rshift(4)
  nibbles = low.cat(high, dim=-1)
  w = (nibbles.cast(dtypes.float32) - 8.0) * scales
  return w.reshape(*shape)

def tensor_to_q4_0_blocks(t: Tensor) -> Tensor:
  """Convert a float tensor to Q4_0 blocks (n_blocks, 18) uint8."""
  flat = t.cast(dtypes.float32).flatten()
  assert flat.shape[0] % 32 == 0
  blocks = flat.reshape(-1, 32)
  amax = blocks.abs().max(axis=-1, keepdim=True)
  d = amax / 8.0
  safe_d = d + (d == 0.0).cast(dtypes.float32) * 1e-10
  q = (blocks / safe_d + 8.0).round().clip(0, 15).cast(dtypes.uint8)
  lo = q[:, :16]
  hi = q[:, 16:]
  packed = lo.bitwise_or(hi.lshift(4))
  d_bytes = d.cast(dtypes.float16).bitcast(dtypes.uint8)
  return d_bytes.cat(packed, dim=-1)

class Q4_0ExpertWeights:
  """Drop-in replacement for ExpertWeights storing weights in Q4_0 format.

  Weight shape: (total_blocks, 18) uint8 where total_blocks = (num_experts * out_features * in_features) / 32.
  Dequantization happens lazily after gathering selected experts, so only active experts are dequanted.
  """
  def __init__(self, num_experts: int, in_features: int, out_features: int):
    assert (in_features * out_features) % 32 == 0
    n_blocks = (num_experts * in_features * out_features) // 32
    self.weight = Tensor.zeros(n_blocks, 18, dtype=dtypes.uint8)
    self.num_experts = num_experts
    self.in_features = in_features
    self.out_features = out_features

  def __call__(self, sel: Tensor, x: Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    blocks_per_expert = (self.in_features * self.out_features) // 32
    w3d = self.weight.reshape(self.num_experts, blocks_per_expert, 18)
    expert_blocks = w3d[sel]  # (B, T, k, blocks_per_expert, 18)

    # Dequant selected experts only
    scales = expert_blocks[..., :2].bitcast(dtypes.float16).cast(dtypes.float32)
    qbytes = expert_blocks[..., 2:]
    low = qbytes.bitwise_and(0x0F)
    high = qbytes.rshift(4)
    nibbles = low.cat(high, dim=-1)
    w = (nibbles.cast(dtypes.float32) - 8.0) * scales  # (B, T, k, blocks_per_expert, 32)
    w = w.reshape(*sel.shape, self.out_features, self.in_features)

    return (x.unsqueeze(-2) @ w.transpose(-1, -2)).squeeze(-2)


def _numpy_ggml_dequant(mm, offset: int, n_elements: int, ggml_type: int):
  """Dequant GGML tensor data in pure numpy. Returns float32 ndarray."""
  import numpy as np

  # Native types
  if ggml_type == 0:  # F32
    return np.frombuffer(mm, dtype=np.float32, count=n_elements, offset=offset).copy()
  if ggml_type == 1:  # F16
    return np.frombuffer(mm, dtype=np.float16, count=n_elements, offset=offset).astype(np.float32)

  # Quantized types
  if ggml_type == 2:  # Q4_0: 32 elements per 18-byte block
    n_blocks = n_elements // 32
    raw = np.frombuffer(mm, dtype=np.uint8, count=n_blocks * 18, offset=offset).reshape(n_blocks, 18)
    d = raw[:, :2].view(np.float16).astype(np.float32)  # (n_blocks, 1)
    qbytes = raw[:, 2:]  # (n_blocks, 16)
    lo = (qbytes & 0x0F).astype(np.int8)
    hi = (qbytes >> 4).astype(np.int8)
    quants = np.concatenate([lo, hi], axis=-1).astype(np.float32) - 8.0  # (n_blocks, 32)
    return (quants * d).flatten()

  if ggml_type == 3:  # Q4_1: 32 elements per 20-byte block
    n_blocks = n_elements // 32
    raw = np.frombuffer(mm, dtype=np.uint8, count=n_blocks * 20, offset=offset).reshape(n_blocks, 20)
    d = raw[:, :2].view(np.float16).astype(np.float32)  # (n_blocks, 1)
    m = raw[:, 2:4].view(np.float16).astype(np.float32)  # (n_blocks, 1)
    qbytes = raw[:, 4:]  # (n_blocks, 16)
    lo = (qbytes & 0x0F).astype(np.float32)
    hi = (qbytes >> 4).astype(np.float32)
    quants = np.concatenate([lo, hi], axis=-1)  # (n_blocks, 32)
    return (quants * d + m).flatten()

  if ggml_type == 8:  # Q8_0: 32 elements per 34-byte block
    n_blocks = n_elements // 32
    raw = np.frombuffer(mm, dtype=np.uint8, count=n_blocks * 34, offset=offset).reshape(n_blocks, 34)
    d = raw[:, :2].view(np.float16).astype(np.float32)  # (n_blocks, 1)
    qs = raw[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)
    return (d * qs).flatten()

  if ggml_type == 14:  # Q6_K: 256 elements per 210-byte block
    n_blocks = n_elements // 256
    raw = np.frombuffer(mm, dtype=np.uint8, count=n_blocks * 210, offset=offset).reshape(n_blocks, 210)
    ql = raw[:, :128].reshape(n_blocks, 2, 64)  # low 4 bits
    qh = raw[:, 128:192].reshape(n_blocks, 2, 32)  # high 2 bits
    scales = raw[:, 192:208].view(np.int8).astype(np.float32)  # (n_blocks, 16)
    d = raw[:, 208:210].view(np.float16).astype(np.float32)  # (n_blocks, 1)
    # Unpack: 4-bit low nibbles
    xl_lo = (ql & 0x0F).astype(np.uint8)
    xl_hi = (ql >> 4).astype(np.uint8)
    xl = np.stack([xl_lo, xl_hi], axis=2).reshape(n_blocks, 4, 64).reshape(n_blocks, 256)
    # Unpack: 2-bit high parts
    xh_0 = (qh & 0x03).astype(np.uint8)
    xh_1 = ((qh >> 2) & 0x03).astype(np.uint8)
    xh_2 = ((qh >> 4) & 0x03).astype(np.uint8)
    xh_3 = ((qh >> 6) & 0x03).astype(np.uint8)
    xh = np.stack([xh_0, xh_1, xh_2, xh_3], axis=2).reshape(n_blocks, 256)
    # Combine: 6-bit value = low4 | (high2 << 4), then subtract 32
    q = (xl | (xh << 4)).astype(np.int8).astype(np.float32) - 32.0
    # Scale: each group of 16 gets its own scale
    scales_expanded = np.repeat(scales, 16, axis=-1)  # (n_blocks, 256)
    return (d * q * scales_expanded).flatten()

  if ggml_type == 13:  # Q5_K: 256 elements per 176-byte block
    n_blocks = n_elements // 256
    raw = np.frombuffer(mm, dtype=np.uint8, count=n_blocks * 176, offset=offset).reshape(n_blocks, 176)
    d = raw[:, :2].view(np.float16).astype(np.float32)  # (n_blocks, 1)
    dmin = raw[:, 2:4].view(np.float16).astype(np.float32)  # (n_blocks, 1)
    scales = raw[:, 4:16]  # (n_blocks, 12)
    qh = raw[:, 16:48]  # (n_blocks, 32)
    ql = raw[:, 48:]  # (n_blocks, 128)
    # Unpack scales (Q4_K get_scale_min pattern)
    scales = scales.reshape((n_blocks, 3, 4))
    sd, sm, m_d = np.split(scales, 3, axis=-2)
    sc = np.concatenate([sd & 0x3F, (m_d & 0x0F) | ((sd >> 2) & 0x30)], axis=-1).reshape((n_blocks, 8))
    m = np.concatenate([sm & 0x3F, (m_d >> 4) | ((sm >> 2) & 0x30)], axis=-1).reshape((n_blocks, 8))
    # Unpack 4-bit low nibbles
    ql = ql.reshape((n_blocks, -1, 1, 32))
    ql_unpacked = np.concatenate([(ql & 0x0F), (ql >> 4)], axis=2).reshape((n_blocks, -1, 32))
    # Unpack 1-bit high parts
    qh = qh.reshape((n_blocks, -1, 1, 32))
    shifts = np.array(range(8), dtype=np.uint8).reshape((1, 1, 8, 1))
    qh_unpacked = ((qh >> shifts) & 0x01).reshape((n_blocks, -1, 32))
    # Combine: 5-bit = low4 | (high1 << 4)
    q = (ql_unpacked | (qh_unpacked << 4)).astype(np.float32)
    d_scaled = (d * sc.astype(np.float32)).reshape((n_blocks, 8, 1))
    dm = (dmin * m.astype(np.float32)).reshape((n_blocks, 8, 1))
    return (d_scaled * q - dm).reshape((n_blocks, 256)).flatten()

  raise ValueError(f"GGML type '{ggml_type}' is not supported in numpy dequant!")


def _numpy_f32_to_q4_0(f32):
  """Requantize f32 ndarray to Q4_0 blocks: (n_blocks, 18) uint8."""
  import numpy as np
  f32 = f32.flatten()
  assert len(f32) % 32 == 0
  blocks = f32.reshape(-1, 32)
  n_blocks = blocks.shape[0]

  # Per-block scale: d = amax / 8
  amax = np.abs(blocks).max(axis=-1, keepdims=True)
  d = amax / 8.0
  safe_d = np.where(d == 0, 1e-10, d)

  # Quantize: q = clamp(round(v / d + 8), 0, 15)
  q = np.clip(np.round(blocks / safe_d + 8.0), 0, 15).astype(np.uint8)

  # Pack nibbles: byte[j] = q[j] | (q[j+16] << 4)
  lo = q[:, :16]
  hi = q[:, 16:]
  packed = (lo | (hi << 4)).astype(np.uint8)  # (n_blocks, 16)

  # Assemble: [scale_fp16(2 bytes), packed(16 bytes)] = 18 bytes
  d_bytes = d.astype(np.float16).view(np.uint8)  # (n_blocks, 2)
  result = np.concatenate([d_bytes, packed], axis=-1)  # (n_blocks, 18)
  return result


def gguf_load_q4_0(tensor: Tensor) -> tuple[dict, dict[str, Tensor], dict[str, tuple]]:
  """GGUF loader that keeps Q4_0 tensors as raw (n_blocks, 18) uint8 blocks.

  Other tensor types are dequanted normally via ggml_data_to_tensor.
  Returns (kv_data, state_dict, tensor_info) where tensor_info maps name -> (ggml_type, dims).
  """
  reader, kv_data, state_dict, tensor_info = io.BufferedReader(TensorIO(tensor), 1_000_000), {}, {}, {}
  def read_unpack(fmt: str, n: int): return struct.unpack(fmt, reader.read(n))[0]
  def read_str(): return str(reader.read(read_uint64()), "utf-8")
  def read_arr():
    reader, n = readers[read_int32()], read_uint64()
    return [reader() for _ in range(n)]

  readers: dict[int, Callable[[], Any]] = { 8: read_str, 9: read_arr, **{ t: functools.partial(read_unpack, "<"+f, nb) for t,f,nb in
    [(0,"c",1), (1,"b",1), (2,"H",2), (3,"h",2), (4,"I",4), (5,"i",4), (6,"f",4), (7,"?",1), (10,"Q",8), (11,"q",8), (12,"d",8)] } }
  read_uint32, read_int32, read_uint64, read_int64 = readers[4], readers[5], readers[10], readers[11]

  magic, version, n_tensors, n_kv = reader.read(4), read_int32(), read_int64(), read_int64()
  if magic != b"GGUF" or version not in [2, 3]: raise ValueError("Invalid GGUF format!")
  for _ in range(n_kv):
    k, typ = read_str(), read_int32()
    kv_data[k] = readers[typ]()

  t_infos = [(read_str(), tuple(read_uint64() for _ in range(read_uint32())), read_int32(), read_uint64()) for _ in range(n_tensors)]
  alignment, pos = kv_data.get("general.alignment", 32), reader.tell()
  data_start = round_up(pos, alignment)

  # Layer name patterns where Q4_0 raw blocks should be kept (projection weights used in Q4_0Linear)
  _q4_0_keep = {'attn_q.', 'attn_qkv.', 'attn_gate.', 'attn_output.', 'ssm_alpha.', 'ssm_beta.', 'ssm_out.',
                  'ffn_gate.', 'ffn_up.', 'ffn_down.', 'ffn_gate_shexp.', 'ffn_up_shexp.', 'ffn_down_shexp.'}

  # Memory-map the GGUF file for direct numpy dequant (avoids tinygrad's lazy graph scheduler issues)
  import numpy as np, mmap as _mmap
  disk_path = None
  for s in tensor.uop.toposort():
    if s.op.name == 'DEVICE' and str(s.arg).startswith('DISK:'):
      disk_path = str(s.arg)[5:]
      break
  assert disk_path is not None, "Could not find disk path from tensor UOp"
  f = open(disk_path, 'rb')
  mm = _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ)

  # Expert weight names that should stay as Q4_0 raw blocks for Q4_0ExpertWeights
  _expert_keys = {'ffn_gate_exps.', 'ffn_up_exps.', 'ffn_down_exps.'}

  for name, dims, typ, off in t_infos:
    n_elements = prod(dims)
    tensor_info[name] = (typ, dims)
    is_q4_proj = any(k in name for k in _q4_0_keep)
    is_expert = any(k in name for k in _expert_keys)

    if is_q4_proj:
      # Keep/requant projection weights as Q4_0 raw blocks for Q4_0Linear
      n_blocks = n_elements // 32
      if typ == 2:  # already Q4_0: zero-copy from disk
        state_dict[name] = tensor[data_start + off : data_start + off + n_blocks * 18].reshape(n_blocks, 18)
      else:  # other quant types: dequant to f32 via numpy, then requant to Q4_0
        f32 = _numpy_ggml_dequant(mm, data_start + off, n_elements, typ)
        state_dict[name] = Tensor(_numpy_f32_to_q4_0(f32))
        del f32
    elif is_expert:
      # Expert weights: keep as Q4_0 raw blocks for Q4_0ExpertWeights (dequant only selected experts at runtime)
      # Q4_0 experts: read raw bytes directly
      # Q4_1/other experts: dequant to f32 in numpy, then requant to Q4_0
      n_blocks = n_elements // 32
      if typ == 2:  # already Q4_0
        raw = np.frombuffer(mm, dtype=np.uint8, count=n_blocks * 18, offset=data_start + off).copy()
        state_dict[name] = Tensor(raw.reshape(n_blocks, 18))
      else:
        # Dequant to f32, then requant to Q4_0 blocks
        f32 = _numpy_ggml_dequant(mm, data_start + off, n_elements, typ)
        state_dict[name] = Tensor(_numpy_f32_to_q4_0(f32))
        del f32
    else:
      # Everything else: dequant in numpy, cast to fp16
      arr = _numpy_ggml_dequant(mm, data_start + off, n_elements, typ).astype('float16')
      state_dict[name] = Tensor(arr).reshape(*reversed(dims))
      del arr

  mm.close()
  f.close()
  return kv_data, state_dict, tensor_info
