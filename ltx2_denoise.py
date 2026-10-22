"""Split LTX-2 (2.3 and 2.5) denoising across two GPUs (sequence parallelism over the video tokens).

Principle
---------
LTX-2 is a dual-stream DiT: each of its 48 blocks updates a video stream (thousands of tokens, 32 heads x 128) and an
audio stream (a few hundred tokens, 32 heads x 64). Per block, the video tokens go through
  1. self-attention (every video token attends to every video token),
  2. cross-attention to the text prompt,
  3. audio-to-video cross-attention (video queries, audio keys/values),
  4. the feed-forward network,
and the audio tokens through the same steps, with a video-to-audio cross-attention (audio queries, video keys/values).

The plugin cuts the video sequence in two: the first rows stay on the primary GPU (where WanGP runs), the last rows go
to the secondary GPU. Everything that works token by token (projections, norms, AdaLN modulation, feed-forward, text and
audio-to-video cross-attention) runs on each GPU for its own rows. The audio stream is small: each GPU computes all of
it. Only two attentions need keys/values from all the video rows: the video self-attention and the video-to-audio
cross-attention. For those, the two GPUs swap their keys and values through pinned RAM, chunk by chunk while the previous
chunk is being attended (GeForce cards have no direct GPU-to-GPU link), and the keys are put back in their original order,
so that every query sees exactly the keys it sees on one GPU. Nothing is approximated or skipped: same weights, same
LoRAs, any number of steps.

Rotary embeddings (a ``RopeCache`` made of grid "layouts") and per-frame timesteps are cut exactly at the split point:
each GPU gets the same values for its rows as with one GPU.

Weights
-------
The primary GPU keeps using WanGP's memory manager (MMGP) untouched: the plugin replaces only the loop over the blocks
(``LTXModel._process_transformer_blocks``) and still calls the real blocks, so MMGP loads them as usual. The secondary
GPU runs hook-free copies of the blocks whose weights are streamed by a loader thread from the same RAM that MMGP uses
(no second copy of the model in RAM), two blocks ahead of the compute.

Attention (WanGP v18.17+)
-------------------------
LTX-2's Attention.forward reaches the attention in two ways: through shared.attention_kit (qkv_attention for the
self-attentions, cross_qkv_attention for long cross-attentions with WanGP's "Attention Head Split"), or by calling
pay_attention itself (other cross-attentions, an attention backend forced by the model). Both are intercepted: inside a
split pass, when the calling Attention module is a block's ``attn1`` or ``video_to_audio_attn`` (marked by the plugin),
attention_kit gets the plugin's distributed attention as its ``attention_fn`` (with all heads at once, so that both GPUs
exchange the same chunks), and pay_attention is replaced by it.

Limits
------
A pass runs on one GPU when it uses an option this version does not split (torch.compile, CUDA graphs, a self-attention
or audio/video cross-attention mask, sequences shorter than MIN_TOKENS, inputs of different shapes in one CFG batch).
Any failure of the multi-GPU path reruns the pass on one GPU and disables the feature until it is switched off and on
again.

Ported from the "Multi-GPU LTX-2" plugin for WanGP 11.22 (v0.2.1) to WanGP v18.17+ (mmgp 4.x).
"""
import contextlib
import dataclasses
import math
import sys
import threading
import time
import traceback
import weakref

import torch
import torch.nn as nn

from . import kernel_compat, settings
from .clones import (CloneError, clone_module, install as _install, mmgp_index, module_is_compiled, own_modules,
                     registry)
from .common import (clear_stale_cuda_error, cuda_graphs_on, deep_map, executor, iter_tensors, tensor_bytes,
                     warn_once)

GB = 1024 ** 3
MIN_TOKENS = 2048          # shorter video sequences stay on one GPU: too little work to share
ROW_ALIGN = 128            # split point aligned to attention/quantization tiles when the timesteps allow any split
CHUNK_HEADS = 4            # attention heads exchanged and attended per chunk
SLOT_BYTES = 64 << 20      # cap on the keys/values of one chunk: long sequences use fewer heads per chunk
SEND_AHEAD = 3             # chunks of keys/values sent ahead of the chunk being attended
SLOTS = SEND_AHEAD + 3     # pinned RAM slots per GPU for the exchange
PREFETCH_BLOCKS = 2        # blocks of weights loaded on the secondary GPU ahead of the block being computed
WAIT_POLL = 0.05
BIG_HOST_TENSOR = 64 << 20 # larger tensors handed over through pageable RAM (pinning hundreds of MB costs more)
ROLE = "_wangp_mgpu_ltx2_role"
ROLES = (("attn1", "self"), ("video_to_audio_attn", "v2a"))   # attentions whose keys/values span all the video rows

_tls = threading.local()
_state = {
    "installed": False,
    "orig_process": None,
    "orig_process_joint": None,
    "orig_pay_attention": None,
    "orig_kit_qkv": None,
    "orig_kit_cross": None,
    "watching": False,
    "announced": set(),
    "broken": None,
    "last_report": "No multi-GPU LTX-2 denoising pass yet.",
    "test_secondary": None,   # tests only: a device standing for the secondary GPU
    "slot_pool": None,
    "clones": None,           # (weakref to the model, _Clones) kept between the passes of a generation
    "last_tokens": None,      # video tokens of the previous pass: a new count is a new phase
    "last_secondary": None,
}


class SPFailure(RuntimeError):
    """The multi-GPU path failed during this pass: it is run again on one GPU and the feature is disabled."""


class _Unsupported(Exception):
    """This pass uses something the split does not handle: it runs on one GPU, the feature stays on."""


class _NoRoom(Exception):
    """Not enough VRAM on the secondary GPU for this pass: it runs on one GPU, the feature stays on."""


class _Aborted(Exception):
    pass


class _Interrupted(Exception):
    pass


def last_report():
    if _state["broken"]:
        return f"Multi-GPU denoising disabled after an error: {_state['broken']} (switch it off and on to retry)."
    return _state["last_report"]


def reset_broken():
    _state["broken"] = None


def drop_clones():
    """Forget the block copies kept between the passes of a generation (they reference MMGP's CPU tensors of the
    model: kept across a model switch, they would keep that model's pinned RAM from being freed). Called when a
    generation ends, when the VAE decode starts, after an interruption or a failure, and by release()."""
    had = _state.get("clones") is not None
    _state["clones"] = None
    _state["last_tokens"] = None
    return had


def _empty_secondary_cache():
    device = _state.get("last_secondary")
    if device is None or not _is_cuda(device):
        return False
    try:
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        return True
    except Exception as exc:
        print(f"[Multi-GPU] Could not empty the VRAM cache of {device}: {exc}")
        return False


def end_of_denoising():
    """The denoising of a generation is over (the VAE decode starts): drop the block copies and give the VRAM that
    the secondary GPU kept cached between passes back, for the VAE copy."""
    drop_clones()
    _empty_secondary_cache()


def release():
    """Drop everything the denoising split keeps between passes."""
    freed = drop_clones()
    _state["slot_pool"] = None
    return _empty_secondary_cache() and freed


# ----------------------------------------------------------------------------------------------------------------------
# Small device helpers (CUDA, or CPU for the tests)
# ----------------------------------------------------------------------------------------------------------------------

def _is_cuda(device):
    return torch.device(device).type == "cuda"


def _new_stream(device):
    return torch.cuda.Stream(device=device) if _is_cuda(device) else None


def _current_stream(device):
    return torch.cuda.current_stream(device) if _is_cuda(device) else None


def _on(stream):
    return torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()


def _event(device, timing=False):
    return torch.cuda.Event(enable_timing=timing) if _is_cuda(device) else None


def _record(event, stream):
    if event is not None:
        event.record(stream)
    return event


def _wait_event(stream, event):
    if stream is not None and event is not None:
        stream.wait_event(event)


def _device_index(device):
    device = torch.device(device)
    if device.type != "cuda":
        return -1
    return device.index if device.index is not None else torch.cuda.current_device()


def _host_slots(nbytes, count, cuda):
    """The exchange's pinned slots, kept between passes and reallocated only when larger ones are needed."""
    pool = _state.get("slot_pool")
    if pool is not None and pool[0] >= nbytes and pool[1] == count and pool[2] == cuda:
        return pool[3]
    _state["slot_pool"] = None
    pool = None
    if cuda:
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
        if empty_host_cache is not None:
            empty_host_cache()
    slots = [[torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=cuda) for _ in range(count)] for _ in range(2)]
    _state["slot_pool"] = (nbytes, count, cuda, slots)
    return slots


def _to_host(tensor, pinned=None):
    """Synchronous copy to host memory: pinned for small tensors, pageable for large ones (and when pinned=False)."""
    if not tensor.is_cuda:
        return tensor.detach().clone()
    if pinned is None:
        pinned = tensor_bytes(tensor) < BIG_HOST_TENSOR
    if not pinned or type(tensor) is not torch.Tensor:
        return tensor.to("cpu")
    host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
    host.copy_(tensor, non_blocking=True)
    torch.cuda.current_stream(tensor.device).synchronize()
    return host


# ----------------------------------------------------------------------------------------------------------------------
# Cutting the arguments of the blocks at the split point
# ----------------------------------------------------------------------------------------------------------------------

def _boxes(start, stop, sizes):
    """A contiguous range [start, stop) of a row-major grid of shape `sizes`, as a list of boxes (one (first, last)
    range per axis), each one contiguous in the flat order and listed in that order."""
    if stop <= start:
        return []
    if len(sizes) == 1:
        return [[(start, stop)]]
    inner = math.prod(sizes[1:])
    i0, r0 = divmod(start, inner)
    i1, r1 = divmod(stop, inner)
    if i0 == i1:
        return [[(i0, i0 + 1)] + box for box in _boxes(r0, r1, sizes[1:])]
    out = []
    if r0:
        out += [[(i0, i0 + 1)] + box for box in _boxes(r0, inner, sizes[1:])]
        i0 += 1
    if i1 > i0:
        out.append([(i0, i1)] + [(0, n) for n in sizes[1:]])
    if r1:
        out += [[(i1, i1 + 1)] + box for box in _boxes(0, r1, sizes[1:])]
    return out


def _slice_rope_cache(cache, start, stop):
    """Rows [start, stop) of an LTX-2 RopeCache: the layouts that cover these rows, cut along their grid axes. Each token
    gets exactly the cos/sin values it gets in the full cache."""
    from models.ltx2.ltx_core.model.transformer.rope import RopeLayoutCache, _build_layout_axis_token_indices
    axes = tuple(cache.rope_axes or ())
    layouts = []
    for layout in cache.layouts:
        first, last = max(layout.token_start, start), min(layout.token_stop, stop)
        if last <= first:
            continue
        sizes = tuple(int(s) for s in layout.grid_sizes)
        if math.prod(sizes) != layout.token_stop - layout.token_start:
            raise _Unsupported("unexpected RoPE layout")
        position = first - start
        for box in _boxes(first - layout.token_start, last - layout.token_start, sizes):
            box_sizes = tuple(b - a for a, b in box)
            count = math.prod(box_sizes)
            axis_cos, axis_sin, token_indices = [], [], []
            for i, axis in enumerate(axes):
                a, b = box[axis]
                axis_cos.append(layout.axis_cos[i][a:b].contiguous())
                axis_sin.append(layout.axis_sin[i][a:b].contiguous())
                token_indices.append(_build_layout_axis_token_indices(box_sizes, axis, layout.axis_cos[i].device))
            layouts.append(RopeLayoutCache(token_start=position, token_stop=position + count, grid_sizes=box_sizes,
                                           axis_cos=tuple(axis_cos), axis_sin=tuple(axis_sin),
                                           axis_token_indices=tuple(token_indices)))
            position += count
    return dataclasses.replace(cache, layouts=tuple(layouts), token_count=stop - start)


def _slice_rope(pe, start, stop, tokens):
    if pe is None:
        return None
    from models.ltx2.ltx_core.model.transformer.rope import RopeCache
    if isinstance(pe, RopeCache):
        if pe.token_count != tokens:
            raise _Unsupported(f"RoPE cache for {pe.token_count} tokens, video has {tokens}")
        return _slice_rope_cache(pe, start, stop)
    if isinstance(pe, (tuple, list)) and len(pe) == 2 and all(torch.is_tensor(t) for t in pe):
        def cut(t):
            if t.dim() == 4 and t.shape[2] == tokens:
                return t[:, :, start:stop]
            if t.dim() >= 2 and t.shape[1] == tokens:
                return t[:, start:stop]
            raise _Unsupported(f"unexpected RoPE shape {tuple(t.shape)}")
        return type(pe)(cut(t) for t in pe)
    raise _Unsupported(f"unexpected RoPE type {type(pe).__name__}")


def _quantum(videos, tokens):
    """Rows per timestep group: the split point must fall on a group boundary when the timesteps are given per group of
    tokens (per latent frame) rather than per token. 1 when any split works."""
    quantum = 1
    for video in videos:
        for t in (video.timesteps, video.cross_scale_shift_timestep, video.cross_gate_timestep):
            if t is None or not torch.is_tensor(t) or t.dim() < 2:
                continue
            n = t.shape[1]
            if n in (1, tokens):
                continue
            if tokens % n:
                raise _Unsupported(f"timesteps for {n} groups do not divide {tokens} video tokens")
            q = tokens // n
            if quantum not in (1, q):
                raise _Unsupported("timesteps with different groupings")
            quantum = q
    return quantum


def _slice_tokens(t, start, stop, tokens, quantum):
    if t is None or not torch.is_tensor(t) or t.dim() < 2:
        return t
    n = t.shape[1]
    if n == tokens:
        return t[:, start:stop]
    if n == 1:
        return t
    if quantum > 1 and n * quantum == tokens and start % quantum == 0 and stop % quantum == 0:
        return t[:, start // quantum:stop // quantum]
    raise _Unsupported(f"cannot cut timesteps of shape {tuple(t.shape)}")


def _video_part(video, start, stop, tokens, quantum, own_x):
    x = video.x[:, start:stop]
    return dataclasses.replace(
        video,
        x=x.clone() if own_x else x,
        timesteps=_slice_tokens(video.timesteps, start, stop, tokens, quantum),
        embedded_timestep=None,   # only used after the blocks, on the full sequence
        positional_embeddings=_slice_rope(video.positional_embeddings, start, stop, tokens),
        cross_positional_embeddings=_slice_rope(video.cross_positional_embeddings, start, stop, tokens),
        cross_scale_shift_timestep=_slice_tokens(video.cross_scale_shift_timestep, start, stop, tokens, quantum),
        cross_gate_timestep=_slice_tokens(video.cross_gate_timestep, start, stop, tokens, quantum),
    )


def _audio_copy(audio):
    """The audio stream with its own hidden states (the blocks update them in place; the original stays intact for a
    rerun on one GPU)."""
    if audio is None:
        return None
    return dataclasses.replace(audio, x=audio.x.clone())


# ----------------------------------------------------------------------------------------------------------------------
# Copies of the blocks for the secondary GPU
# ----------------------------------------------------------------------------------------------------------------------

def _mark_roles(block):
    for name, role in ROLES:
        module = block._modules.get(name)
        if module is not None and module.__dict__.get(ROLE) != role:
            module.__dict__[ROLE] = role


class _Clones:
    """Hook-free copies of the blocks of one LTX model and, per block, where each of their tensors comes from."""

    def __init__(self, model):
        blocks = list(model.transformer_blocks)
        for block in blocks:
            _mark_roles(block)
        counts = {}
        for block in blocks:
            for module in own_modules(block):
                counts[id(module)] = counts.get(id(module), 0) + 1
        shared = {key for key, count in counts.items() if count > 1}
        for block in blocks:
            for module in own_modules(block):
                if id(module) in shared and (len(module._parameters) or len(module._buffers)):
                    raise _Unsupported(f"module {type(module).__name__} with weights is shared between blocks")
        index = mmgp_index([m for block in blocks for m in own_modules(block)])
        self.blocks, self.entries, self.loras, self.block_bytes = [], [], [], []
        for block in blocks:
            memo = {}
            try:
                clone = clone_module(block, memo, shared)
            except CloneError as exc:
                raise _Unsupported(str(exc)) from exc
            entries = registry(block, memo, index)
            self.blocks.append(clone)
            self.entries.append(entries)
            self.loras.append([(memo[id(m)], m) for m in own_modules(block)
                               if id(m) in memo and "_mm_lora_old_forward" in m.__dict__])
            self.block_bytes.append(sum(tensor_bytes(entry[2]) for entry in entries))
        self.model_ref = weakref.ref(model)


def _clones_for(model):
    """Block copies for the passes of one generation (kept between its steps, rebuilt for the next generation: see
    drop_clones)."""
    cached = _state.get("clones")
    if cached is not None and cached[0]() is model and len(cached[1].blocks) == len(model.transformer_blocks):
        return cached[1]
    started = time.perf_counter()
    clones = _Clones(model)
    _state["clones"] = (weakref.ref(model), clones)
    if id(model) not in _state["announced"]:
        _state["announced"].add(id(model))
        print(f"[Multi-GPU] Prepared {len(clones.blocks)} LTX-2 block copies for the secondary GPU "
              f"({max(clones.block_bytes) / GB:.2f} GB for the largest block) in {time.perf_counter() - started:.1f} s.")
    return clones


def _resolve_lora(value, adapter, shortcuts):
    while isinstance(value, str):
        target, slot = value.rsplit("#", 1)
        value = shortcuts[target][adapter][int(slot)]
    return value


def _load_block(clones, index, device):
    """Installs the weights (and active LoRAs) of block `index` on its copy, on `device`. Returns the tensors to keep."""
    moved = {}

    def move(tensor):
        if tensor is None or not torch.is_tensor(tensor):
            return tensor
        key = id(tensor)
        if key not in moved:
            source = tensor.data if isinstance(tensor, nn.Parameter) else tensor
            moved[key] = source.to(device, non_blocking=True)
        return moved[key]

    for module, name, tensor, is_buffer, tied in clones.entries[index]:
        copy = move(tensor)
        _install(module, name, copy, is_buffer)
        if tied is not None:
            _install(tied[0], tied[1], copy, tied[1] in tied[0]._buffers)
    for clone, original in clones.loras[index]:
        data = original.__dict__.get("_mm_lora_data") or {}
        cpu = {key: value for key, value in data.items() if not key.endswith("_GPU")}
        model = getattr(original, "_mm_lora_model", None)   # registered as a child module by mmgp 3.7.6
        shortcuts = getattr(model, "_loras_model_shortcuts", None)
        gpu = {}
        for adapter in (getattr(model, "_loras_active_adapters", None) or ()):
            entry = cpu.get(adapter)
            if entry is None:
                continue
            entry = list(entry)
            for i in range(min(4, len(entry))):
                entry[i] = move(_resolve_lora(entry[i], adapter, shortcuts))
            gpu[adapter + "_GPU"] = entry
        clone.__dict__["_mm_lora_data"] = {**cpu, **gpu}
    return moved


def _unload_block(clones, index):
    for module, name, tensor, is_buffer, tied in clones.entries[index]:
        _install(module, name, tensor, is_buffer)
        if tied is not None:
            _install(tied[0], tied[1], tensor, tied[1] in tied[0]._buffers)
    for clone, original in clones.loras[index]:
        data = original.__dict__.get("_mm_lora_data") or {}
        clone.__dict__["_mm_lora_data"] = {key: value for key, value in data.items() if not key.endswith("_GPU")}


# ----------------------------------------------------------------------------------------------------------------------
# Key/value exchange between the two GPUs, through pinned RAM
# ----------------------------------------------------------------------------------------------------------------------

def _partial_attention(query, key, value, score_bytes=256 << 20):
    """Attention of all the queries over some of the keys, with the log-sum-exp of each query's scores, in float32.
    Two partial results over disjoint keys merge exactly into the attention over all the keys (see _merge_partial).
    Used for the video-to-audio cross-attention: a few hundred audio queries, so computing the scores explicitly is
    cheap, and exchanging these results (a few MB) replaces exchanging the video keys/values (hundreds of MB)."""
    batch, queries, heads, head_dim = query.shape
    keys = key.shape[1]
    scale = head_dim ** -0.5
    out = torch.empty((batch, queries, heads, head_dim), dtype=torch.float32, device=query.device)
    lse = torch.empty((batch, heads, queries), dtype=torch.float32, device=query.device)
    step = max(1, min(heads, score_bytes // max(1, batch * queries * keys * 4)))
    for start in range(0, heads, step):
        stop = min(start + step, heads)
        q = query[:, :, start:stop].transpose(1, 2).float()
        k = key[:, :, start:stop].transpose(1, 2).float()
        v = value[:, :, start:stop].transpose(1, 2).float()
        scores = torch.matmul(q, k.transpose(-1, -2)).mul_(scale)
        del q, k
        peak = scores.amax(dim=-1, keepdim=True)
        scores.sub_(peak).exp_()
        total = scores.sum(dim=-1, keepdim=True)
        part = torch.matmul(scores, v).div_(total)
        del scores, v
        out[:, :, start:stop] = part.transpose(1, 2)
        lse[:, start:stop] = (peak + total.log()).squeeze(-1)
        del part, peak, total
    return out, lse


def _merge_partial(first, second):
    """Exact attention over the keys of both parts, from (output, log-sum-exp) of each part (first = rows of cuda:0)."""
    (o0, l0), (o1, l1) = first, second
    peak = torch.maximum(l0, l1)
    w0 = (l0 - peak).exp_()
    w1 = (l1 - peak).exp_()
    total = w0 + w1
    w0 = w0.div_(total).transpose(1, 2).unsqueeze(-1)
    w1 = w1.div_(total).transpose(1, 2).unsqueeze(-1)
    return o0 * w0 + o1 * w1


class _Exchange:
    """Each side writes its keys/values, chunk by chunk, into its own ring of pinned slots; the other side reads them.
    GPU-side ordering uses CUDA events (a stream of one GPU may wait on an event of the other); host-side handshakes make
    sure an event has been recorded before the other side waits on it."""

    def __init__(self, session, slot_bytes, slot_count, cuda):
        self.session = session
        self.slot_bytes = slot_bytes
        self.slots = _host_slots(slot_bytes, slot_count, cuda)
        self.next = [0, 0]
        self.pending = [set(), set()]
        self.published, self.consumed = {}, {}

    def acquire(self, side, stream):
        index = self.next[side]
        self.next[side] = (index + 1) % len(self.slots[side])
        if index in self.pending[side]:
            event = self.session.wait_for(lambda: (side, index) in self.consumed,
                                          lambda: self.consumed.pop((side, index)))
            self.pending[side].discard(index)
            _wait_event(stream, event)
        return index, self.slots[side][index]

    def publish(self, side, call, chunk, index, event, shape):
        with self.session.cond:
            self.published[(side, call, chunk)] = (index, event, shape)
            self.pending[side].add(index)
            self.session.cond.notify_all()

    def take(self, side, call, chunk):
        return self.session.wait_for(lambda: (side, call, chunk) in self.published,
                                     lambda: self.published.pop((side, call, chunk)))

    def release(self, side, index, event):
        with self.session.cond:
            self.consumed[(side, index)] = event
            self.session.cond.notify_all()


class _Side:
    """What one GPU needs during the pass: its streams and the distributed attention."""

    def __init__(self, session, side, device):
        self.session, self.side, self.device = session, side, device
        self.call = 0
        self.send_stream = self.recv_stream = None
        self.stalls = []          # (event before, event after) around waits for the other GPU
        self.weight_stalls = []   # secondary GPU: around waits for streamed weights

    def streams(self):
        if self.send_stream is None and _is_cuda(self.device):
            self.send_stream, self.recv_stream = _new_stream(self.device), _new_stream(self.device)
        return self.send_stream, self.recv_stream

    def _send(self, k, v, call, chunk, heads):
        exchange = self.session.exchange
        send_stream, _ = self.streams()
        index, buffer = exchange.acquire(self.side, send_stream)
        b, n, d = k.shape[0], k.shape[1], k.shape[3]
        count = b * n * len(heads) * d
        view = buffer[:2 * count * k.element_size()].view(k.dtype).view(2, b, n, len(heads), d)
        with _on(send_stream):
            view[0].copy_(k[:, :, heads.start:heads.stop], non_blocking=True)
            view[1].copy_(v[:, :, heads.start:heads.stop], non_blocking=True)
            event = _record(_event(self.device), send_stream)
        exchange.publish(self.side, call, chunk, index, event, (b, n, len(heads), d, k.dtype))

    def _partial_merge(self, query, key, value):
        """Video-to-audio cross-attention: each GPU attends the (replicated) audio queries over its own video rows, the
        two GPUs swap these partial results (output + log-sum-exp, a few MB) and merge them in the same order, so both
        get the attention over all the video rows."""
        exchange = self.session.exchange
        peer = 1 - self.side
        call = self.call
        self.call += 1
        batch, queries, heads, head_dim = query.shape
        tag = ("v2a", batch, queries, heads, head_dim)
        own = _partial_attention(query, key, value)
        del key, value
        n_out, n_lse = own[0].numel(), own[1].numel()
        nbytes = 4 * (n_out + n_lse)
        if nbytes > exchange.slot_bytes:
            raise SPFailure("video-to-audio partial results do not fit in the exchange slots")
        compute = _current_stream(self.device)
        send_stream, recv_stream = self.streams()
        if send_stream is not None:
            send_stream.wait_stream(compute)
        index, buffer = exchange.acquire(self.side, send_stream)
        view = buffer[:nbytes].view(torch.float32)
        with _on(send_stream):
            view[:n_out].copy_(own[0].reshape(-1), non_blocking=True)
            view[n_out:].copy_(own[1].reshape(-1), non_blocking=True)
            event = _record(_event(self.device), send_stream)
        if send_stream is not None:
            own[0].record_stream(send_stream)
            own[1].record_stream(send_stream)
        exchange.publish(self.side, call, 0, index, event, tag)
        slot_index, peer_event, shape = exchange.take(peer, call, 0)
        if shape != tag:
            raise SPFailure(f"exchange mismatch: got {shape}, expected {tag}")
        peer_view = exchange.slots[peer][slot_index][:nbytes].view(torch.float32)
        other = (torch.empty_like(own[0]), torch.empty_like(own[1]))
        if recv_stream is not None:
            recv_stream.wait_stream(compute)   # the buffers were allocated on the compute stream
        with _on(recv_stream):
            _wait_event(recv_stream, peer_event)
            other[0].view(-1).copy_(peer_view[:n_out], non_blocking=True)
            other[1].view(-1).copy_(peer_view[n_out:], non_blocking=True)
            received = _record(_event(self.device), recv_stream)
        exchange.release(peer, slot_index, received)
        before = _record(_event(self.device, timing=True), compute)
        _wait_event(compute, received)
        after = _record(_event(self.device, timing=True), compute)
        if before is not None:
            self.stalls.append((before, after))
        first, second = (own, other) if self.side == 0 else (other, own)
        return _merge_partial(first, second).to(query.dtype)

    def attention(self, qkv_list, role, kwargs):
        """pay_attention for an attention whose keys/values are this GPU's video rows: q (any rows, all heads), k and v
        of this GPU's rows (normalized and rotated). Returns the same tensor as pay_attention on one GPU."""
        session, exchange = self.session, self.session.exchange
        original = _state["orig_pay_attention"]
        query, key, value = qkv_list
        qkv_list.clear()
        batch, _, heads, head_dim = query.shape
        rows = key.shape[1]
        peer = 1 - self.side
        if rows != session.rows[self.side]:
            raise SPFailure(f"{role} attention got {rows} key rows, expected {session.rows[self.side]}")
        if key.shape[2] != heads or key.shape[3] != head_dim:
            raise SPFailure(f"unexpected key shape {tuple(key.shape)} for {role} attention")
        if role == "v2a" and session.v2a_merge:
            return self._partial_merge(query, key, value)
        peer_rows = session.rows[peer]
        total = rows + peer_rows
        local = slice(0, rows) if self.side == 0 else slice(peer_rows, total)   # global row order, as with one GPU
        remote = slice(rows, total) if self.side == 0 else slice(0, peer_rows)
        step = session.chunk_heads.get((role, heads, head_dim))
        if step is None:
            step = session.chunk_for(heads, head_dim)
        if 2 * batch * max(session.rows) * step * head_dim * key.element_size() > exchange.slot_bytes:
            raise SPFailure(f"{role} attention chunk does not fit in the exchange slots")
        call = self.call
        self.call += 1
        compute = _current_stream(self.device)
        send_stream, recv_stream = self.streams()
        chunks = [range(start, min(start + step, heads)) for start in range(0, heads, step)]
        if send_stream is not None:
            send_stream.wait_stream(compute)   # k and v are written by the compute stream
        for chunk in range(min(SEND_AHEAD, len(chunks))):
            self._send(key, value, call, chunk, chunks[chunk])

        width = min(step, heads)
        flats = [(key.new_empty(batch * total * width * head_dim), value.new_empty(batch * total * width * head_dim))
                 for _ in range(2)]
        if recv_stream is not None:
            recv_stream.wait_stream(compute)   # the buffers were allocated on the compute stream
        attended = [None, None]
        output = None
        for chunk, heads_range in enumerate(chunks):
            ahead = chunk + SEND_AHEAD
            if ahead < len(chunks):
                self._send(key, value, call, ahead, chunks[ahead])
            count = len(heads_range)
            slot_index, peer_event, shape = exchange.take(peer, call, chunk)
            if shape != (batch, peer_rows, count, head_dim, key.dtype):
                raise SPFailure(f"exchange mismatch: got {shape}, expected {(batch, peer_rows, count, head_dim, key.dtype)}")
            peer_view = exchange.slots[peer][slot_index][:2 * batch * peer_rows * count * head_dim * key.element_size()]
            peer_view = peer_view.view(key.dtype).view(2, batch, peer_rows, count, head_dim)
            buffer = chunk % 2
            k_full = flats[buffer][0][:batch * total * count * head_dim].view(batch, total, count, head_dim)
            v_full = flats[buffer][1][:batch * total * count * head_dim].view(batch, total, count, head_dim)
            k_full[:, local].copy_(key[:, :, heads_range.start:heads_range.stop])
            v_full[:, local].copy_(value[:, :, heads_range.start:heads_range.stop])
            with _on(recv_stream):
                _wait_event(recv_stream, attended[buffer])
                _wait_event(recv_stream, peer_event)
                k_full[:, remote].copy_(peer_view[0], non_blocking=True)
                v_full[:, remote].copy_(peer_view[1], non_blocking=True)
                received = _record(_event(self.device), recv_stream)
            exchange.release(peer, slot_index, received)
            before = _record(_event(self.device, timing=True), compute)
            _wait_event(compute, received)
            after = _record(_event(self.device, timing=True), compute)
            if before is not None:
                self.stalls.append((before, after))
            q_chunk = query[:, :, heads_range.start:heads_range.stop].contiguous()
            result = original([q_chunk, k_full, v_full], recycle_q=True, **kwargs)
            if output is None:
                output = result.new_empty(query.shape[:2] + (heads,) + result.shape[3:])
            output[:, :, heads_range.start:heads_range.stop].copy_(result)
            del result, q_chunk
            attended[buffer] = _record(_event(self.device), compute)
        if send_stream is not None:
            key.record_stream(send_stream)
            value.record_stream(send_stream)
        del key, value, query
        return output


# ----------------------------------------------------------------------------------------------------------------------
# One denoising pass
# ----------------------------------------------------------------------------------------------------------------------

def _usable_vram(device, empty):
    """VRAM an allocation can get on device: free in the driver, plus what PyTorch's allocator keeps cached. The cache
    is emptied only when asked (first pass of a phase): within a phase, the secondary GPU reuses the memory of the
    previous step instead of allocating it again from the driver."""
    torch.cuda.synchronize(device)
    if empty:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    cached = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    return free + cached


class _Session:
    def __init__(self, model, videos, audios, perturbations, cfg):
        self.model = model
        self.main_thread = threading.current_thread()
        self.cond = threading.Condition()
        self.abort, self.error = False, None
        self.started = time.perf_counter()
        self.videos, self.audios, self.perturbations = list(videos), list(audios), list(perturbations)
        self.count = len(self.videos)
        x = self.videos[0].x
        primary = x.device
        secondary = _state["test_secondary"] or torch.device("cuda", 1 if (primary.index or 0) == 0 else 0)
        self.devices = (primary, secondary)
        self.inference = torch.is_inference_mode_enabled()
        batch, tokens = x.shape[0], x.shape[1]
        self.tokens, self.batch = tokens, batch
        quantum = _quantum(self.videos, tokens)
        align = quantum if quantum > 1 else ROW_ALIGN
        if tokens < 2 * align:
            raise _Unsupported(f"{tokens} video tokens cannot be cut on a boundary of {align} tokens")
        share = cfg["ltx_denoise_share_cuda0"] / 100.0
        rows0 = int(round(tokens * share / align)) * align
        rows0 = max(align, min(tokens - align, rows0))
        if tokens - rows0 <= 0:
            raise _Unsupported("nothing left for the secondary GPU")
        self.rows = (rows0, tokens - rows0)
        self.quantum = quantum

        # Arguments of each GPU (one per CFG condition in a joint pass). The originals are never modified, so that a
        # failed pass can be run again on one GPU.
        self.local_videos = [_video_part(v, 0, rows0, tokens, quantum, own_x=True) for v in self.videos]
        self.local_audios = [_audio_copy(a) for a in self.audios]
        memo = {}
        remote = [_video_part(v, rows0, tokens, tokens, quantum, own_x=False) for v in self.videos]
        remote_audio = [None if a is None else dataclasses.replace(a, embedded_timestep=None) for a in self.audios]
        self.remote_host = deep_map((remote, remote_audio), _to_host, memo)
        self.remote_bytes = sum(tensor_bytes(t) for t in iter_tensors(self.remote_host))

        self.clones = _clones_for(model)
        self.blocks = len(self.clones.blocks)
        block = model.transformer_blocks[0]
        self.element = x.element_size()
        wanted = int(cfg.get("denoise_chunk_heads", 0) or 0) or CHUNK_HEADS
        self.wanted_heads = wanted
        self.chunk_heads, slot = {}, 0
        audio = next((a for a in self.audios if a is not None), None)
        self.v2a_merge = False
        for name, role in ROLES:
            attn = block._modules.get(name)
            if attn is None:
                continue
            heads, head_dim = int(attn.heads), int(attn.dim_head)
            if role == "v2a":
                if audio is None:
                    continue
                # partial results (output + log-sum-exp in float32) instead of the video keys/values
                blob = 4 * (batch * audio.x.shape[1] * heads * head_dim + batch * heads * audio.x.shape[1])
                if blob <= SLOT_BYTES:
                    self.v2a_merge = True
                    slot = max(slot, blob)
                    continue
            step = self.chunk_for(heads, head_dim)
            self.chunk_heads[(role, heads, head_dim)] = step
            slot = max(slot, 2 * batch * max(self.rows) * step * head_dim * self.element)
        self.slot_bytes = max(slot, 1 << 20)
        self.max_chunk_dim = max((step * key[2] for key, step in self.chunk_heads.items()), default=CHUNK_HEADS * 128)
        self._check_vram(x)
        self.exchange = _Exchange(self, self.slot_bytes, SLOTS, _is_cuda(primary))
        self.sides = (_Side(self, 0, primary), _Side(self, 1, secondary))
        self.ready, self.done = {}, {}
        self.worker = self.loader = None

    def chunk_for(self, heads, head_dim):
        head_bytes = 2 * self.batch * max(self.rows) * head_dim * self.element
        return max(1, min(self.wanted_heads, heads, SLOT_BYTES // head_bytes))

    # --- host-side waits, interruptible ------------------------------------------------------------------------------
    def check(self):
        """Raises when the pass must stop: SPFailure in WanGP's thread (the pass is then computed again on one GPU),
        _Aborted in the threads of the secondary GPU."""
        if self.error is not None:
            if threading.current_thread() is self.main_thread:
                raise SPFailure(self.error)
            raise _Aborted()
        if self.abort:
            raise _Aborted()

    def wait_for(self, predicate, take):
        with self.cond:
            while not predicate():
                self.check()
                self.cond.wait(WAIT_POLL)
            return take()

    def fail(self, exc):
        with self.cond:
            if self.error is None:
                self.error = f"{type(exc).__name__}: {exc}"
            self.cond.notify_all()

    def stop(self):
        with self.cond:
            self.abort = True
            self.cond.notify_all()

    def _check_vram(self, x):
        secondary = self.devices[1]
        if not _is_cuda(secondary):
            self.vram_need = 0
            return
        dim = x.shape[-1]
        need = (PREFETCH_BLOCKS + 2) * max(self.clones.block_bytes)
        need += 2 * self.remote_bytes                                     # the inputs and the updated hidden states
        need += 8 * self.count * self.batch * self.rows[1] * dim * self.element   # activations of the rows of cuda:1
        need += 4 * self.batch * self.tokens * self.max_chunk_dim * self.element  # full keys/values of one chunk
        need += GB // 2
        self.vram_need = need
        new_phase = _state.get("last_tokens") != self.tokens
        _state["last_tokens"], _state["last_secondary"] = self.tokens, secondary
        free = _usable_vram(secondary, empty=new_phase)
        if free < need and not new_phase:
            free = _usable_vram(secondary, empty=True)
        if free < need:
            raise _NoRoom(f"not enough free VRAM on {secondary}: {free / GB:.1f} GB usable, about {need / GB:.1f} GB needed")
        torch.cuda.reset_peak_memory_stats(secondary)
        self.vram_start = torch.cuda.memory_allocated(secondary)

    # --- secondary GPU threads ----------------------------------------------------------------------------------------
    def _grad_mode(self):
        return torch.inference_mode() if self.inference else torch.no_grad()

    def _loader_job(self):
        device = self.devices[1]
        clear_stale_cuda_error(_device_index(device))
        stream = _new_stream(device)
        loaded = []
        try:
            with self._grad_mode(), torch.device(device), _on(stream):
                for index in range(self.blocks):
                    while loaded and loaded[0] < index - PREFETCH_BLOCKS:
                        old = loaded.pop(0)
                        event = self.wait_for(lambda: old in self.done, lambda: self.done[old])
                        if event is not None:
                            event.synchronize()
                        _unload_block(self.clones, old)
                    keep = _load_block(self.clones, index, device)
                    event = _record(_event(device), stream)
                    with self.cond:
                        self.ready[index] = (event, keep)
                        self.cond.notify_all()
                    loaded.append(index)
                for old in loaded:
                    event = self.wait_for(lambda: old in self.done, lambda: self.done[old])
                    if event is not None:
                        event.synchronize()
                    _unload_block(self.clones, old)
                loaded = []
        except _Aborted:
            pass
        except BaseException as exc:
            traceback.print_exc()
            self.fail(exc)
        finally:
            if loaded:
                if _is_cuda(device):
                    torch.cuda.synchronize(device)
                for old in loaded:
                    _unload_block(self.clones, old)

    def _worker_job(self):
        device = self.devices[1]
        clear_stale_cuda_error(_device_index(device))
        side = self.sides[1]
        _tls.sp = side
        try:
            with self._grad_mode(), torch.device(device):   # as WanGP does on the primary GPU
                compute = _current_stream(device)
                videos, audios = deep_map(self.remote_host, lambda t: t.to(device, non_blocking=True))
                videos, audios = list(videos), list(audios)

                def wait_weights(index):
                    event, _ = self.wait_for(lambda: index in self.ready, lambda: self.ready[index])
                    before = _record(_event(device, timing=True), compute)
                    _wait_event(compute, event)
                    after = _record(_event(device, timing=True), compute)
                    if before is not None:
                        side.weight_stalls.append((before, after))

                for index in range(self.blocks):
                    clone = self.clones.blocks[index]
                    wait_weights(index)
                    for c in range(self.count):
                        self.check()
                        videos[c], audios[c] = clone(video=videos[c], audio=audios[c],
                                                     perturbations=self.perturbations[c])
                    done = _record(_event(device), compute)
                    with self.cond:
                        self.done[index] = done
                        self.ready.pop(index, None)
                        self.cond.notify_all()
                return [_to_host(v.x, pinned=False) for v in videos]
        except _Aborted:
            return None
        except BaseException as exc:
            traceback.print_exc()
            self.fail(exc)
            return None
        finally:
            _tls.sp = None

    def _start(self):
        index = _device_index(self.devices[1])
        self.loader = executor(index, "loader").submit(self._loader_job)
        self.worker = executor(index).submit(self._worker_job)

    # --- primary GPU: WanGP's thread runs the real blocks (managed by MMGP) on its rows -------------------------------
    def run(self):
        model = self.model
        self.setup_time = time.perf_counter() - self.started
        self._start()
        _tls.sp = self.sides[0]
        loop_started = time.perf_counter()
        try:
            for block in model.transformer_blocks:
                if model.interrupt_check is not None and model.interrupt_check():
                    raise _Interrupted()
                for c in range(self.count):
                    self.local_videos[c], self.local_audios[c] = block(video=self.local_videos[c],
                                                                       audio=self.local_audios[c],
                                                                       perturbations=self.perturbations[c])
                self.check()
        except (SPFailure, _Interrupted, _Aborted):
            raise
        except Exception as exc:
            if "Interrupt" in type(exc).__name__ or "Abort" in type(exc).__name__:
                raise
            traceback.print_exc()
            raise SPFailure(f"{type(exc).__name__}: {exc}") from exc
        finally:
            _tls.sp = None
        self.loop_time = time.perf_counter() - loop_started
        remote = self.worker.result()
        self.check()
        if remote is None:
            raise SPFailure("the secondary GPU stopped")
        self.loader.result()
        self.check()
        videos_out = []
        for c in range(self.count):
            local_x = self.local_videos[c].x
            full = torch.cat((local_x, remote[c].to(local_x.device)), dim=1)
            videos_out.append(dataclasses.replace(self.videos[c], x=full))
        self.local_videos = None
        self.finished = time.perf_counter()
        return videos_out, list(self.local_audios)

    # --- end of pass ---------------------------------------------------------------------------------------------------
    def close(self, completed):
        if not completed:
            self.stop()
        for future in (self.worker, self.loader):
            if future is not None:
                try:
                    future.result(timeout=60)
                except Exception:
                    pass
        self.clones = None
        self.remote_host = None
        if not completed:   # the block copies may be half loaded: rebuild them, and give the VRAM back
            drop_clones()
        secondary = self.devices[1]
        if _is_cuda(secondary):
            try:   # nothing may still read the pinned slots
                torch.cuda.synchronize(self.devices[0])
                torch.cuda.synchronize(secondary)
                if not completed:
                    with torch.cuda.device(secondary):
                        torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[Multi-GPU] Could not synchronize {secondary}: {exc}")
        if not completed or not hasattr(self, "finished"):
            return
        cuda = all(_is_cuda(d) for d in self.devices)
        if cuda:
            for device in self.devices:
                torch.cuda.synchronize(device)

        def total(pairs):
            return sum(before.elapsed_time(after) for before, after in pairs) / 1000.0

        wall = self.finished - self.started
        stall0, stall1 = (total(self.sides[0].stalls), total(self.sides[1].stalls)) if cuda else (0.0, 0.0)
        weights = total(self.sides[1].weight_stalls) if cuda else 0.0
        primary, secondary = (_device_index(d) for d in self.devices)
        share = 100.0 * self.rows[0] / self.tokens
        # Each GPU's own work: the pass minus its waits for the other GPU (and for weights). The share that gives both
        # GPUs the same work time is suggested.
        busy0, busy1 = max(wall - stall0, 1e-6), max(wall - stall1 - weights, 1e-6)
        rate0, rate1 = busy0 / self.rows[0], busy1 / self.rows[1]
        suggested = int(round(100.0 * rate1 / (rate0 + rate1)))
        hint = "balanced share" if abs(suggested - round(share)) <= 1 else f"suggested share of cuda:{primary}: {suggested} %"
        vram = ""
        if _is_cuda(self.devices[1]):
            peak = torch.cuda.max_memory_allocated(self.devices[1]) - getattr(self, "vram_start", 0)
            vram = f" | cuda:{secondary} peak {peak / GB:.1f} GB (estimate {self.vram_need / GB:.1f} GB)"
        heads = ", ".join(f"{role} {step}" for (role, _, _), step in self.chunk_heads.items())
        if self.v2a_merge:
            heads += ", v2a merged"
        cfg_text = f", {self.count} conditions in one pass" if self.count > 1 else ""
        audio = "" if all(a is None for a in self.audios) else f" + {self.audios[0].x.shape[1]} audio tokens on both GPUs"
        loop = max(getattr(self, "loop_time", wall), 1e-6)
        report = (f"Last LTX-2 denoising pass: {wall:.1f} s for {self.tokens} video tokens{audio}{cfg_text} (heads per "
                  f"chunk: {heads}), {self.rows[0]} on cuda:{primary} ({share:.0f} %), {self.rows[1]} on "
                  f"cuda:{secondary} | waits for the other GPU: cuda:{primary} {stall0:.2f} s, cuda:{secondary} "
                  f"{stall1:.2f} s ({hint}) | cuda:{secondary} waited {weights:.2f} s for weights{vram} | setup "
                  f"{getattr(self, 'setup_time', 0.0):.2f} s, blocks {loop:.1f} s.")
        _state["last_report"] = report
        print(f"[Multi-GPU] {report}")


# ----------------------------------------------------------------------------------------------------------------------
# Patched entry points
# ----------------------------------------------------------------------------------------------------------------------

def _ineligible(model, videos, audios, cfg):
    if not cfg["enabled"] or not cfg["ltx_denoise_enabled"]:
        return "off"
    if _state["broken"]:
        return "broken"
    if not videos or any(v is None for v in videos):
        return "a pass without video tokens"
    first = videos[0].x
    if _state["test_secondary"] is None and (not first.is_cuda or torch.cuda.device_count() < 2):
        return "needs two visible GPUs"
    if any(not v.enabled or v.x.shape != first.shape for v in videos):
        return "conditions of different shapes in one pass"
    if first.shape[0] != 1:
        return f"batch of {first.shape[0]} videos in one pass"
    if any(getattr(v, "self_attention_mask", None) is not None for v in videos):
        return "a self-attention mask is used"
    if any(getattr(t, "cross_attention_mask", None) is not None for t in list(videos) + list(audios) if t is not None):
        return "an audio/video cross-attention mask is used"
    shapes = {None if a is None else tuple(a.x.shape) for a in audios}
    if len(shapes) > 1:
        return "audio streams of different shapes in one pass"
    if first.shape[1] < MIN_TOKENS:
        return f"short sequence ({first.shape[1]} video tokens, under the {MIN_TOKENS} needed to share the work)"
    if getattr(model, "_enable_gradient_checkpointing", False) and model.training:
        return "training mode"
    if module_is_compiled(model.transformer_blocks[0]):
        return "torch.compile is on"
    if cuda_graphs_on(model.transformer_blocks[0]):
        return "CUDA graphs are on"
    if _state["test_secondary"] is None:
        primary_index = first.device.index if first.device.index is not None else torch.cuda.current_device()
        problem = kernel_compat.prepare_secondary(primary_index, 1 if primary_index == 0 else 0)
        if problem is not None:
            return f"the INT8 kernels could not be set up for the second GPU ({problem})"
    return None


def _run(model, videos, audios, perturbations, joint):
    if joint:
        def run_original():
            return _state["orig_process_joint"](model, videos, audios, perturbations)
    else:
        def run_original():
            return _state["orig_process"](model, videos[0], audios[0], perturbations)
    cfg = settings.get()
    reason = _ineligible(model, videos, audios, cfg)
    if reason is None:
        _watch_generations()
    if reason is not None:
        if reason not in ("off", "broken"):
            warn_once(f"denoise_{reason}", f"LTX-2 denoising runs on one GPU: {reason}.")
            _state["last_report"] = f"Last LTX-2 denoising pass on one GPU: {reason}."
        return run_original()
    if isinstance(perturbations, list):
        if len(perturbations) != len(videos):
            return run_original()
        perts = list(perturbations)
    else:
        perts = [perturbations] * len(videos)

    try:
        session = _Session(model, videos, audios, perts, cfg)
    except (_Unsupported, _NoRoom) as exc:
        warn_once(f"denoise_skip_{exc}", f"This denoising pass runs on one GPU: {exc}.")
        _state["last_report"] = f"Last LTX-2 denoising pass on one GPU: {exc}."
        return run_original()
    except Exception as exc:
        traceback.print_exc()
        _state["broken"] = f"{type(exc).__name__}: {exc}"
        print(f"[Multi-GPU] Could not prepare the multi-GPU denoising ({exc}): this pass runs on one GPU, and the "
              f"feature is disabled until it is switched off and on.")
        return run_original()

    completed = False
    try:
        videos_out, audios_out = session.run()
        completed = True
    except _Interrupted:
        session.close(False)
        model.interrupted = True
        if joint:
            return [None] * len(videos), [None] * len(audios)
        return None, None
    except (SPFailure, _Aborted) as exc:
        session.close(False)
        _state["broken"] = str(exc) or type(exc).__name__
        print(f"[Multi-GPU] Multi-GPU denoising failed ({exc}): this pass is computed again on one GPU, and the "
              f"feature is disabled until it is switched off and on.")
        return run_original()
    finally:
        if completed:
            session.close(True)
    if joint:
        return videos_out, audios_out
    return videos_out[0], audios_out[0]


def _watch_generations():
    """Drop the block copies when an LTX-2 generation ends, whatever happened in it. Installed at the first LTX-2
    denoising pass (the module is loaded by then; importing it when the plugin loads would load LTX-2 for every user)."""
    if _state.get("watching"):
        return
    _state["watching"] = True
    try:
        module = sys.modules.get("models.ltx2.ltx2")
        if module is None:
            raise RuntimeError("models.ltx2.ltx2 is not loaded")
        original_generate = module.LTX2.generate
        if getattr(original_generate, "_wangp_mgpu_ltx2", False):
            return

        def generate(self, *args, **kwargs):
            try:
                return original_generate(self, *args, **kwargs)
            finally:
                drop_clones()
        generate._wangp_mgpu_ltx2 = True
        module.LTX2.generate = generate
    except Exception as exc:
        warn_once("watch_generate", f"Could not watch the end of LTX-2 generations ({type(exc).__name__}: {exc}); the "
                                    f"secondary GPU's block copies are dropped at the VAE decode instead.")


def _patched_process(self, video, audio, perturbations):
    return _run(self, [video], [audio], perturbations, joint=False)


def _patched_process_joint(self, video_list, audio_list, perturbations):
    return _run(self, list(video_list), list(audio_list), perturbations, joint=True)


def _caller_role():
    """The role marker of the LTX-2 Attention module calling the attention (looked up a few frames up, past any wrapper
    another plugin may have put around the attention functions)."""
    frame = sys._getframe(2)
    for _ in range(4):
        if frame is None:
            return None
        caller = frame.f_locals.get("self")
        if isinstance(caller, nn.Module):
            return caller.__dict__.get(ROLE)
        frame = frame.f_back
    return None


def _patched_pay_attention(qkv_list, *args, **kwargs):
    """shared.attention.pay_attention as seen by LTX-2's Attention.forward. Inside a split pass, the video self-attention
    and the video-to-audio cross-attention go to the distributed attention of the calling thread's GPU."""
    side = getattr(_tls, "sp", None)
    if side is not None and not args:
        role = _caller_role()
        if role is not None:
            for name in ("attention_mask", "q_lens", "k_lens"):
                if kwargs.get(name) is not None:
                    raise SPFailure(f"unsupported attention option {name}")
            kwargs.pop("recycle_q", None)
            return side.attention(qkv_list, role, kwargs)
    return _state["orig_pay_attention"](qkv_list, *args, **kwargs)


def _kit_options(side, role, kwargs):
    """attention_kit options of a distributed attention: the plugin's attention_fn, all heads at once (both GPUs must
    exchange the same chunks, whatever their own token counts)."""
    for name in ("attention_mask", "kv_prefix", "attention_fn"):
        if kwargs.get(name) is not None:
            raise SPFailure(f"unsupported attention option {name}")
    kwargs = dict(kwargs)
    kwargs["attention_fn"] = lambda qkv_list, group, *extra: side.attention(qkv_list, role, {})
    kwargs["split_heads"] = False
    return kwargs


def _patched_kit_qkv_attention(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, *args, **kwargs):
    side = getattr(_tls, "sp", None)
    if side is not None and not args:
        role = _caller_role()
        if role is not None:
            kwargs = _kit_options(side, role, kwargs)
    return _state["orig_kit_qkv"](x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, *args, **kwargs)


def _patched_kit_cross_qkv_attention(x_list, context_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, *args,
                                     **kwargs):
    side = getattr(_tls, "sp", None)
    if side is not None and not args:
        role = _caller_role()
        if role is not None:
            kwargs = _kit_options(side, role, kwargs)
    return _state["orig_kit_cross"](x_list, context_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, *args,
                                    **kwargs)


class _PortableCudart:
    """torch.cuda.cudart() whose cudaHostRegister registers pinned memory as portable (readable by every GPU)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def cudaHostRegister(self, pointer, size, flags):
        return self._inner.cudaHostRegister(pointer, size, int(flags) | 1)   # cudaHostRegisterPortable


def install():
    """Installs the LTX-2 hooks (class and module level, so the moment a model is loaded does not matter). Call it after
    the MiniMax H3 install, which wraps attention_kit.qkv_attention too: each wrapper hands over to the other outside
    its own passes."""
    if _state["installed"]:
        return
    from shared import attention_kit
    from models.ltx2.ltx_core.model.transformer import attention as ltx_attention
    from models.ltx2.ltx_core.model.transformer.model import LTXModel

    if not getattr(torch.cuda.cudart, "_wangp_mgpu", False) and not getattr(torch.cuda.cudart, "_wangp_mgpu_ltx2", False):
        original_cudart = torch.cuda.cudart

        def cudart():
            return _PortableCudart(original_cudart())
        cudart._wangp_mgpu_ltx2 = True
        torch.cuda.cudart = cudart
    if not getattr(ltx_attention.pay_attention, "_wangp_mgpu_ltx2", False):
        _state["orig_pay_attention"] = ltx_attention.pay_attention
        _patched_pay_attention._wangp_mgpu_ltx2 = True
        ltx_attention.pay_attention = _patched_pay_attention
    if not getattr(attention_kit.qkv_attention, "_wangp_mgpu_ltx2", False):
        _state["orig_kit_qkv"] = attention_kit.qkv_attention
        _patched_kit_qkv_attention._wangp_mgpu_ltx2 = True
        attention_kit.qkv_attention = _patched_kit_qkv_attention
    if hasattr(attention_kit, "cross_qkv_attention") and not getattr(attention_kit.cross_qkv_attention, "_wangp_mgpu_ltx2", False):
        _state["orig_kit_cross"] = attention_kit.cross_qkv_attention
        _patched_kit_cross_qkv_attention._wangp_mgpu_ltx2 = True
        attention_kit.cross_qkv_attention = _patched_kit_cross_qkv_attention
    if not getattr(LTXModel._process_transformer_blocks, "_wangp_mgpu_ltx2", False):
        _state["orig_process"] = LTXModel._process_transformer_blocks
        _patched_process._wangp_mgpu_ltx2 = True
        LTXModel._process_transformer_blocks = _patched_process
    if not getattr(LTXModel._process_transformer_blocks_joint, "_wangp_mgpu_ltx2", False):
        _state["orig_process_joint"] = LTXModel._process_transformer_blocks_joint
        _patched_process_joint._wangp_mgpu_ltx2 = True
        LTXModel._process_transformer_blocks_joint = _patched_process_joint
    _state["installed"] = True
