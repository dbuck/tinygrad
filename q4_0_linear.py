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

  for name, dims, typ, off in t_infos:
    n_elements = prod(dims)
    tensor_info[name] = (typ, dims)
    if typ == 2:  # Q4_0: keep as raw blocks instead of dequanting
      n_blocks = n_elements // 32
      state_dict[name] = tensor[data_start + off : data_start + off + n_blocks * 18].reshape(n_blocks, 18)
    else:
      state_dict[name] = ggml_data_to_tensor(tensor[data_start + off:], n_elements, typ).reshape(*reversed(dims))

  return kv_data, state_dict, tensor_info
