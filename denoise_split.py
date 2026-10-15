"""Split MiniMax H3 denoising across two GPUs (sequence parallelism).

Principle
---------
H3 is a single-stream DiT: every block processes the whole token sequence (text, audio and video tokens), and only
attention mixes tokens. The plugin cuts the sequence in two: the first rows stay on the primary GPU (where WanGP runs),
the last rows go to the secondary GPU. Each GPU runs all 50 blocks on its own rows; in every attention, the two GPUs
swap their keys and values through pinned RAM, so that each query still attends to every key of the sequence. Nothing is
approximated: no stale activations, no skipped steps, the same weights, the same LoRAs. It works for any number of steps,
distilled models and turbo LoRAs included.

Weights
-------
The primary GPU keeps using WanGP's memory manager (MMGP) untouched. The secondary GPU runs light copies of the blocks
(the same module objects rebuilt without MMGP's hooks) whose weights are streamed by a loader thread from the very same
pinned RAM that MMGP uses for the primary GPU (no second copy of the model in RAM), a few blocks ahead of the compute.
LoRAs are mirrored the same way. For the secondary GPU to read MMGP's pinned RAM at full speed, the plugin asks CUDA to
register pinned memory as "portable" (visible to every GPU), which changes nothing for the primary GPU.

Limits (version 0.3)
--------------------
Standard H3 attention path only. A denoising pass falls back to one GPU when it uses Sol (sparse) attention, VEDA, VDN,
ControlNet, Spectrum or first block caching, CUDA graphs, or a sequence shorter than MIN_TOKENS. Any failure of the
multi-GPU path reruns the pass on one GPU and disables the feature until it is switched off and on again.
"""
import contextlib
import functools
import threading
import time
import traceback
import types
import weakref

import torch
import torch.nn as nn

from . import settings
from .common import executor, tensor_bytes, warn_once

GB = 1024 ** 3
MIN_TOKENS = 4096          # shorter sequences (images, very short clips) stay on one GPU: little to gain
ROW_ALIGN = 128            # split point aligned to attention/quantization tiles: same tiles as with one GPU
CHUNK_HEADS = 4            # attention heads exchanged and attended per chunk: smaller chunks, shorter wait for the first one
SLOT_BYTES = 64 << 20      # cap on the keys/values of one chunk: long sequences use fewer heads per chunk
SEND_AHEAD = 3             # chunks of keys/values sent ahead of the chunk being attended
SLOTS = SEND_AHEAD + 3     # pinned RAM slots per GPU for the exchange
PREFETCH_BLOCKS = 2        # blocks of weights loaded on the secondary GPU ahead of the block being computed
WAIT_POLL = 0.05

_tls = threading.local()
_state = {
    "installed": False,
    "orig_model_forward": None,
    "orig_block_forward": None,
    "orig_qkv_attention": None,
    "clones": {},                 # id(model) -> _Clones
    "broken": None,               # reason why the multi-GPU path was disabled after a failure
    "last_report": "No multi-GPU denoising pass yet.",
    "test_secondary": None,       # tests only: a device standing for the secondary GPU
}


class SPFailure(RuntimeError):
    """The multi-GPU path cannot run this pass: it is run again on one GPU."""


class _Aborted(Exception):
    pass


class _NoRoom(Exception):
    """Not enough VRAM on the secondary GPU for this pass: it runs on one GPU, the feature stays on."""


def last_report():
    if _state["broken"]:
        return f"Multi-GPU denoising disabled after an error: {_state['broken']} (switch it off and on to retry)."
    return _state["last_report"]


def reset_broken():
    _state["broken"] = None


def release():
    """Drop the block copies (they hold no VRAM between passes; their weights are loaded per pass)."""
    had = bool(_state["clones"])
    _state["clones"].clear()
    return had


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


def _host_slots(nbytes, count, cuda):
    """The exchange's pinned slots, kept between passes and reallocated only when a longer sequence needs larger ones
    (pinned RAM cannot be swapped out by Windows: the plugin keeps as little of it as possible)."""
    pool = _state.get("slot_pool")
    if pool is not None and pool[0] >= nbytes and pool[1] == count and pool[2] == cuda:
        return pool[3]
    _state["slot_pool"] = None
    pool = None
    if cuda:
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
        if empty_host_cache is not None:
            empty_host_cache()
    slots = [[torch.empty(nbytes, dtype=torch.uint8, pin_memory=cuda) for _ in range(count)] for _ in range(2)]
    _state["slot_pool"] = (nbytes, count, cuda, slots)
    return slots


def _to_host(tensor, pinned=True):
    """Synchronous copy to host memory: pinned for small tensors read back asynchronously, pageable for the hidden
    states handed over once per pass (hundreds of MB on long videos)."""
    if not tensor.is_cuda:
        return tensor.detach().clone()
    if not pinned:
        return tensor.to("cpu")
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    host.copy_(tensor, non_blocking=True)
    torch.cuda.current_stream(tensor.device).synchronize()
    return host


def _map(value, fn):
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map(v, fn) for v in value)
    if isinstance(value, list):
        return [_map(v, fn) for v in value]
    return value


def _take(x_list):
    x = x_list[0]
    x_list.clear()
    return x


# ----------------------------------------------------------------------------------------------------------------------
# Copies of the blocks for the secondary GPU
# ----------------------------------------------------------------------------------------------------------------------

_HOOK_DICTS = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks",
               "_forward_pre_hooks_with_kwargs", "_backward_hooks", "_backward_pre_hooks")


def _rebind(callable_, original, clone):
    """A callable bound to `original` (bound method or functools.partial with it as first argument), bound to `clone`."""
    if isinstance(callable_, types.MethodType) and callable_.__self__ is original:
        return types.MethodType(callable_.__func__, clone)
    if isinstance(callable_, functools.partial) and callable_.args and callable_.args[0] is original:
        return functools.partial(callable_.func, clone, *callable_.args[1:], **(callable_.keywords or {}))
    raise SPFailure(f"unsupported hook on module {type(original).__name__}")


def _is_mmgp_loader(callable_):
    func = callable_.func if isinstance(callable_, functools.partial) else callable_
    name = getattr(func, "__name__", "")
    return name.startswith("_mm_wrap_") or name.startswith("check_load_into_GPU_needed") or name == "check_change_module"


def _clone_module(module, memo, shared):
    if id(module) in shared:
        return module
    if id(module) in memo:
        return memo[id(module)]
    clone = module.__class__.__new__(module.__class__)
    values = dict(module.__dict__)
    for key in _HOOK_DICTS:
        if key in values:
            if len(values[key]):
                warn_once(f"hooks_{type(module).__name__}_{key}",
                          f"Denoising copies ignore {len(values[key])} {key} of {type(module).__name__}.")
            values[key] = type(values[key])()
    values["_parameters"] = type(module._parameters)(module._parameters)
    values["_buffers"] = type(module._buffers)(module._buffers)
    values["_modules"] = type(module._modules)()
    values.pop("_mm_forward", None)
    forward = values.pop("forward", None)
    clone.__dict__.update(values)
    memo[id(module)] = clone
    for name, child in module._modules.items():
        clone._modules[name] = None if child is None else _clone_module(child, memo, shared)

    old_forward = module.__dict__.get("_mm_lora_old_forward")
    if old_forward is not None:
        from mmgp import offload
        clone.__dict__["_mm_lora_old_forward"] = _rebind(old_forward, module, clone)
        clone.__dict__["_mm_lora_data"] = {}
        linear = isinstance(module, nn.Linear) or getattr(module, "is_nvfp4", False)
        clone.__dict__["forward"] = functools.partial(
            offload._mm_lora_linear_forward if linear else offload._mm_lora_generic_forward, clone)
    elif forward is not None:
        inner = module.__dict__.get("_mm_forward", forward) if _is_mmgp_loader(forward) else forward
        if isinstance(inner, types.MethodType) and inner.__self__ is module and inner.__func__ is getattr(type(module), "forward", None):
            pass  # the class forward: nothing to keep
        elif _is_mmgp_loader(inner):
            pass
        else:
            clone.__dict__["forward"] = _rebind(inner, module, clone)
    return clone


class _Clones:
    """Hook-free copies of the blocks of one H3 model and, per block, where each of their tensors comes from."""

    def __init__(self, model):
        blocks = list(model.blocks)
        counts = {}
        for block in blocks:
            for module in block.modules():
                counts[id(module)] = counts.get(id(module), 0) + 1
        shared = {key for key, count in counts.items() if count > 1}
        sol = getattr(model, "sol_attention", None)
        if sol is not None:
            shared.add(id(sol))
        for block in blocks:
            for module in block.modules():
                if id(module) in shared and (len(module._parameters) or len(module._buffers)):
                    raise SPFailure(f"module {type(module).__name__} with weights is shared between blocks")
        self.blocks, self.entries, self.loras, self.block_bytes = [], [], [], []
        for block in blocks:
            memo = {}
            clone = _clone_module(block, memo, shared)
            entries = self._registry(block, memo)
            self.blocks.append(clone)
            self.entries.append(entries)
            self.loras.append([(memo[id(m)], m) for m in block.modules()
                               if id(m) in memo and "_mm_lora_old_forward" in m.__dict__])
            self.block_bytes.append(sum(tensor_bytes(entry[2]) for entry in entries))
        self.model_ref = weakref.ref(model)

    @staticmethod
    def _registry(block, memo):
        """(clone module, name, CPU tensor, is_buffer, (tied clone, name) or None) for every tensor of the block. The CPU
        tensors are those MMGP keeps (pinned) for the block, so that nothing is copied twice in RAM."""
        modules = {id(m): m for m in block.modules() if id(m) in memo}
        entries, covered = [], set()
        manager = block.__dict__.get("_mm_manager")
        if manager is not None:
            for params in manager.blocks_of_modules.values():
                for parent, name, tensor, is_buffer, tied in params:
                    if id(parent) not in modules:
                        continue
                    tied_clone = None
                    if tied is not None and id(tied[0]) in modules:
                        tied_clone = (memo[id(tied[0])], tied[1])
                        covered.add((id(tied[0]), tied[1]))
                    entries.append((memo[id(parent)], name, tensor, is_buffer, tied_clone))
                    covered.add((id(parent), name))
        for module in modules.values():
            for is_buffer, tensors in ((False, module._parameters), (True, module._buffers)):
                for name, tensor in tensors.items():
                    if tensor is None or (id(module), name) in covered:
                        continue
                    if tensor.is_cuda:  # kept in VRAM by MMGP without a CPU copy: one copy in pinned RAM
                        tensor = _to_host(tensor.data if isinstance(tensor, nn.Parameter) else tensor)
                    entries.append((memo[id(module)], name, tensor, is_buffer, None))
        return entries


def _clones_for(model):
    clones = _state["clones"].get(id(model))
    if clones is None or clones.model_ref() is not model:
        _state["clones"].clear()
        started = time.perf_counter()
        clones = _state["clones"][id(model)] = _Clones(model)
        print(f"[Multi-GPU] Prepared {len(clones.blocks)} H3 block copies for the secondary GPU "
              f"({max(clones.block_bytes) / GB:.2f} GB for the largest block) in {time.perf_counter() - started:.1f} s.")
    return clones


def _install(module, name, tensor, is_buffer):
    if is_buffer:
        module._buffers[name] = tensor
    else:
        module._parameters[name] = tensor


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
        model = original.__dict__.get("_mm_lora_model")
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

class _Exchange:
    """Each side writes its keys/values, chunk by chunk, into its own ring of pinned slots; the other side reads them.
    GPU-side ordering uses CUDA events (a stream of one GPU may wait on an event of the other); host-side handshakes make
    sure an event has been recorded before the other side waits on it."""

    def __init__(self, session, slot_bytes, slot_count, cuda):
        self.session = session
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
    """What one GPU needs during the pass: its rows, segments, rope, streams, and the distributed attention."""

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
        n, d = k.shape[1], k.shape[3]
        count = n * len(heads) * d
        view = buffer[:2 * count * k.element_size()].view(k.dtype).view(2, 1, n, len(heads), d)
        with _on(send_stream):
            view[0].copy_(k[:, :, heads.start:heads.stop], non_blocking=True)
            view[1].copy_(v[:, :, heads.start:heads.stop], non_blocking=True)
            event = _record(_event(self.device), send_stream)
        exchange.publish(self.side, call, chunk, index, event, (n, len(heads)))

    def attention(self, qkv_list, group):
        """attention_fn for shared.attention_kit: q, k, v of this GPU's rows (normalized and rotated), all heads."""
        from shared.attention import pay_attention
        session, exchange = self.session, self.session.exchange
        query, key, value = qkv_list
        qkv_list.clear()
        _, rows, heads, head_dim = query.shape
        peer = 1 - self.side
        peer_rows = session.rows[peer]
        total = rows + peer_rows
        local = slice(0, rows) if self.side == 0 else slice(peer_rows, total)   # global row order, as with one GPU
        remote = slice(rows, total) if self.side == 0 else slice(0, peer_rows)
        call = self.call
        self.call += 1
        compute = _current_stream(self.device)
        send_stream, recv_stream = self.streams()
        step = session.chunk_heads
        chunks = [range(start, min(start + step, heads)) for start in range(0, heads, step)]
        if send_stream is not None:
            send_stream.wait_stream(compute)   # k and v are written by the compute stream
        for chunk in range(min(SEND_AHEAD, len(chunks))):
            self._send(key, value, call, chunk, chunks[chunk])

        width = min(step, heads)
        flats = [(key.new_empty(total * width * head_dim), value.new_empty(total * width * head_dim)) for _ in range(2)]
        if recv_stream is not None:
            recv_stream.wait_stream(compute)   # the buffers were allocated on the compute stream
        attended = [None, None]
        output = query.new_empty(query.shape)
        for chunk, heads_range in enumerate(chunks):
            ahead = chunk + SEND_AHEAD
            if ahead < len(chunks):
                self._send(key, value, call, ahead, chunks[ahead])
            count = len(heads_range)
            slot_index, peer_event, shape = exchange.take(peer, call, chunk)
            if shape != (peer_rows, count):
                raise SPFailure(f"exchange mismatch: got {shape}, expected {(peer_rows, count)}")
            peer_view = exchange.slots[peer][slot_index][:2 * peer_rows * count * head_dim * key.element_size()]
            peer_view = peer_view.view(key.dtype).view(2, 1, peer_rows, count, head_dim)
            buffer = chunk % 2
            k_full = flats[buffer][0][:total * count * head_dim].view(1, total, count, head_dim)
            v_full = flats[buffer][1][:total * count * head_dim].view(1, total, count, head_dim)
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
            result = pay_attention([q_chunk, k_full, v_full], recycle_q=True)
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

def _usable_vram(device):
    """VRAM an allocation can get on device: free in the driver, plus what PyTorch's allocator keeps cached (WanGP's VMM
    allocator keeps freed memory mapped, so the driver alone reports too little)."""
    torch.cuda.synchronize(device)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(device)
    cached = max(0, torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    return free + cached


def _rows_slice(value, start, stop, sequence):
    """Rows [start, stop) of the tensors of rope (token axis 1, or 0)."""
    def cut(tensor):
        if tensor.dim() >= 2 and tensor.shape[1] == sequence:
            return tensor[:, start:stop].contiguous()
        if tensor.shape[0] == sequence:
            return tensor[start:stop].contiguous()
        raise SPFailure(f"unexpected rope shape {tuple(tensor.shape)}")
    return _map(value, cut)


def _rebase_segments(segments, start, stop):
    out = []
    for first, last, row in segments:
        first, last = max(first, start), min(last, stop)
        if last > first:
            out.append((first - start, last - start, row))
    return out


class _Session:
    def __init__(self, ctx, hidden, segments, rope):
        model, cfg = ctx.model, ctx.cfg
        self.ctx, self.model = ctx, model
        self.cond = threading.Condition()
        self.abort, self.error = False, None
        self.started = time.perf_counter()
        primary = hidden.device
        secondary = _state["test_secondary"] or torch.device("cuda", 1 if (primary.index or 0) == 0 else 0)
        self.devices = (primary, secondary)
        self.inference = torch.is_inference_mode_enabled()
        sequence = hidden.shape[0]
        share = cfg["denoise_share_cuda0"] / 100.0
        rows0 = int(round(sequence * share / ROW_ALIGN)) * ROW_ALIGN
        rows0 = max(ROW_ALIGN, min(sequence - ROW_ALIGN, rows0))
        self.sequence, self.rows = sequence, (rows0, sequence - rows0)
        self.segments = (_rebase_segments(segments, 0, rows0), _rebase_segments(segments, rows0, sequence))
        self.rope_local = None if rope is None else _rows_slice(rope, 0, rows0, sequence)
        self.rope_remote_host = None if rope is None else _map(_rows_slice(rope, rows0, sequence, sequence), _to_host)
        self.clones = _clones_for(model)
        attn = model.blocks[0].attn
        self.heads, self.head_dim = attn.heads, attn.head_dim
        self.blocks = len(model.blocks)
        element = hidden.element_size()
        head_bytes = max(self.rows) * self.head_dim * 2 * element
        self.chunk_heads = max(1, min(CHUNK_HEADS, self.heads, SLOT_BYTES // head_bytes))
        self._check_vram(hidden)
        self.exchange = _Exchange(self, head_bytes * self.chunk_heads, SLOTS, _is_cuda(primary))
        self.sides = (_Side(self, 0, primary), _Side(self, 1, secondary))
        self.messages, self.temb_host = {}, {}
        self.ready, self.done = {}, {}
        self.worker = self.loader = None

    # --- host-side waits, interruptible ------------------------------------------------------------------------------
    def check(self):
        """Raises when the pass must stop: SPFailure in WanGP's thread (the pass is then computed again on one GPU),
        _Aborted in the threads of the secondary GPU."""
        if self.error is not None:
            if threading.current_thread() is self.ctx.main_thread:
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

    def _check_vram(self, hidden):
        secondary = self.devices[1]
        if not _is_cuda(secondary):
            return
        hidden_size = hidden.shape[1]
        need = (PREFETCH_BLOCKS + 2) * max(self.clones.block_bytes)
        need += 7 * self.rows[1] * hidden_size * hidden.element_size()   # measured peak: about 6x the rows of cuda:1
        need += 4 * self.sequence * self.chunk_heads * self.head_dim * hidden.element_size()
        need += GB // 2
        self.vram_need = need
        free = _usable_vram(secondary)
        if free < need:
            from . import te_split
            if te_split.release():
                free = _usable_vram(secondary)
                print(f"[Multi-GPU] Freed the text encoder layers on {secondary} to make room for denoising.")
        if free < need:
            raise _NoRoom(f"not enough free VRAM on {secondary}: {free / GB:.1f} GB usable, about {need / GB:.1f} GB needed")
        torch.cuda.reset_peak_memory_stats(secondary)
        self.vram_start = torch.cuda.memory_allocated(secondary)

    # --- secondary GPU threads ----------------------------------------------------------------------------------------
    def _grad_mode(self):
        return torch.inference_mode() if self.inference else torch.no_grad()

    def _loader_job(self):
        device = self.devices[1]
        stream = _new_stream(device)
        loaded = []
        try:
            with self._grad_mode(), _on(stream):
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

    def _worker_job(self, hidden_host):
        device = self.devices[1]
        side = self.sides[1]
        original = _state["orig_block_forward"]
        _tls.sp = side
        try:
            with self._grad_mode():
                compute = _current_stream(device)
                hidden = hidden_host.to(device, non_blocking=True)
                rope = None if self.rope_remote_host is None else _map(self.rope_remote_host, lambda t: t.to(device, non_blocking=True))
                tembs = {}
                for index in range(self.blocks):
                    key = self.wait_for(lambda: index in self.messages, lambda: self.messages[index])
                    if key not in tembs:
                        tembs[key] = self.temb_host[key].to(device, non_blocking=True)
                    event, _ = self.wait_for(lambda: index in self.ready, lambda: self.ready[index])
                    before = _record(_event(device, timing=True), compute)
                    _wait_event(compute, event)
                    after = _record(_event(device, timing=True), compute)
                    if before is not None:
                        side.weight_stalls.append((before, after))
                    hidden = original(self.clones.blocks[index], [hidden], tembs[key], self.segments[1], rope)
                    done = _record(_event(device), compute)
                    with self.cond:
                        self.done[index] = done
                        self.ready.pop(index, None)
                        self.cond.notify_all()
                return _to_host(hidden, pinned=False)
        except _Aborted:
            return None
        except BaseException as exc:
            traceback.print_exc()
            self.fail(exc)
            return None
        finally:
            _tls.sp = None

    # --- primary GPU, called for each block by WanGP's loop ------------------------------------------------------------
    def _send_temb(self, temb):
        key = id(temb)
        if key not in self.temb_host:
            self.temb_host[key] = _to_host(temb)
        return key

    def block(self, index, block, x_list, temb):
        try:
            return self._block(index, block, x_list, temb)
        except (SPFailure, _Aborted):
            raise
        except Exception as exc:
            if "Interrupt" in type(exc).__name__ or "Abort" in type(exc).__name__:
                raise
            traceback.print_exc()
            raise SPFailure(f"{type(exc).__name__}: {exc}") from exc

    def _block(self, index, block, x_list, temb):
        original = _state["orig_block_forward"]
        if index == 0:
            hidden = _take(x_list)
            local = hidden[:self.rows[0]].clone()
            remote_host = _to_host(hidden[self.rows[0]:], pinned=False)
            del hidden
            self.loader = executor(self.devices[1].index or 0, "loader").submit(self._loader_job)
            self.worker = executor(self.devices[1].index or 0).submit(self._worker_job, remote_host)
        else:
            local = _take(x_list)
        key = self._send_temb(temb)
        with self.cond:
            self.messages[index] = key
            self.cond.notify_all()
        _tls.sp = self.sides[0]
        try:
            output = original(block, [local], temb, self.segments[0], self.rope_local)
        finally:
            _tls.sp = None
        if index < self.blocks - 1:
            self.check()
            return output
        remote_host = self.worker.result()
        self.check()
        if remote_host is None:
            raise SPFailure("the secondary GPU stopped")
        remote = remote_host.to(output.device, non_blocking=True)
        full = torch.cat((output, remote), dim=0)
        del output, remote
        self.loader.result()
        self.check()
        self.finished = time.perf_counter()
        return full

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
        secondary = self.devices[1]
        if _is_cuda(secondary):
            try:   # nothing may still read the pinned slots; then give the cached VRAM of the pass back
                torch.cuda.synchronize(self.devices[0])
                torch.cuda.synchronize(secondary)
                with torch.cuda.device(secondary):
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[Multi-GPU] Could not empty the VRAM cache of {secondary}: {exc}")
        if not completed or not hasattr(self, "finished"):
            return
        for device in self.devices:
            if _is_cuda(device):
                torch.cuda.synchronize(device)

        def total(pairs):
            return sum(before.elapsed_time(after) for before, after in pairs) / 1000.0

        wall = self.finished - self.started
        stall0, stall1 = total(self.sides[0].stalls), total(self.sides[1].stalls)
        weights = total(self.sides[1].weight_stalls)
        primary, secondary = (d.index or 0 for d in self.devices)
        share = 100.0 * self.rows[0] / self.sequence
        if abs(stall0 - stall1) < max(0.02 * wall, 0.15 * max(stall0, stall1)):
            hint = "balanced share"
        elif stall0 > stall1:
            hint = f"cuda:{primary} waits more: raise its share"
        else:
            hint = f"cuda:{secondary} waits more: lower the share of cuda:{primary}"
        vram = ""
        if _is_cuda(self.devices[1]):
            peak = torch.cuda.max_memory_allocated(self.devices[1]) - getattr(self, "vram_start", 0)
            vram = f" | cuda:{secondary} peak {peak / GB:.1f} GB (estimate {self.vram_need / GB:.1f} GB)"
        report = (f"Last H3 denoising pass: {wall:.1f} s for {self.sequence} tokens, {self.rows[0]} on cuda:{primary} "
                  f"({share:.0f} %), {self.rows[1]} on cuda:{secondary} | waits for the other GPU: cuda:{primary} "
                  f"{stall0:.2f} s, cuda:{secondary} {stall1:.2f} s ({hint}) | cuda:{secondary} waited {weights:.2f} s "
                  f"for weights{vram}.")
        _state["last_report"] = report
        print(f"[Multi-GPU] {report}")


class _ForwardContext:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.main_thread = threading.current_thread()
        self.index = {id(block): i for i, block in enumerate(model.blocks)}
        self.session = None
        self.decided = False


# ----------------------------------------------------------------------------------------------------------------------
# Patched entry points
# ----------------------------------------------------------------------------------------------------------------------

def _ineligible(model, video_x, payload, spectrum, first_block_cache, cfg):
    if not cfg["enabled"] or not cfg["denoise_enabled"]:
        return "off"
    if _state["broken"]:
        return "broken"
    if _state["test_secondary"] is None and (not video_x.is_cuda or torch.cuda.device_count() < 2):
        return "needs two visible GPUs"
    if spectrum is not None or first_block_cache is not None:
        return "Spectrum or first block cache is on"
    if payload.get("control_rows") is not None:
        return "ControlNet is on"
    attn = model.blocks[0].attn
    if not hasattr(attn, "q_proj"):
        return "the q/k/v projections are fused in this checkpoint"
    if hasattr(attn, "vdn"):
        return "VDN attention is on"
    if hasattr(model, "veda_attention") or hasattr(attn, "veda"):
        return "VEDA attention is on"
    if getattr(model, "control_layers", ()):
        return "this model has ControlNet layers"
    try:
        from models.minimax_h3.transformer import _graph_manager
        if _graph_manager(model.blocks[0]) is not None:
            return "CUDA graphs are on"
    except Exception:
        pass
    return None


def _patched_model_forward(self, video_x, audio_x, sigma_video, sigma_audio, context, payload, spectrum=None,
                           first_block_cache=None):
    original = _state["orig_model_forward"]
    cfg = settings.get()
    reason = _ineligible(self, video_x, payload, spectrum, first_block_cache, cfg)
    args = (video_x, audio_x, sigma_video, sigma_audio, context, payload)
    kwargs = {"spectrum": spectrum, "first_block_cache": first_block_cache}
    if reason is not None:
        if reason not in ("off", "broken"):
            warn_once(f"denoise_{reason}", f"H3 denoising runs on one GPU: {reason}.")
        return original(self, *args, **kwargs)
    return _split_forward(self, cfg, args, kwargs)


def _split_forward(self, cfg, args, kwargs):
    original = _state["orig_model_forward"]
    video_x, audio_x, sigma_video, sigma_audio, context, payload = args
    spectrum, first_block_cache = kwargs["spectrum"], kwargs["first_block_cache"]
    ctx = _ForwardContext(self, cfg)
    previous = getattr(_tls, "forward", None)
    _tls.forward = ctx
    completed = False
    try:
        result = original(self, video_x, audio_x, sigma_video, sigma_audio, context, payload, spectrum=spectrum,
                          first_block_cache=first_block_cache)
        completed = True
        return result
    except SPFailure as exc:
        _state["broken"] = str(exc)
        print(f"[Multi-GPU] Multi-GPU denoising failed ({exc}): this pass is computed again on one GPU, and the feature is "
              f"disabled until it is switched off and on.")
        if ctx.session is not None:
            ctx.session.close(False)
            ctx.session = None
        _tls.forward = previous
        return original(self, video_x, audio_x, sigma_video, sigma_audio, context, payload, spectrum=spectrum,
                        first_block_cache=first_block_cache)
    finally:
        if ctx.session is not None:
            ctx.session.close(completed)
        _tls.forward = previous


def _patched_block_forward(self, x_list, temb, segments, rope, residual_signature_elements=0):
    original = _state["orig_block_forward"]
    ctx = getattr(_tls, "forward", None)
    if ctx is None or residual_signature_elements or getattr(_tls, "sp", None) is not None:
        if residual_signature_elements:
            return original(self, x_list, temb, segments, rope, residual_signature_elements)
        return original(self, x_list, temb, segments, rope)
    index = ctx.index.get(id(self))
    if index is None:
        return original(self, x_list, temb, segments, rope)
    if index == 0 and not ctx.decided:
        ctx.decided = True
        hidden = x_list[0]
        sol = getattr(ctx.model, "sol_attention", None)
        if hidden.shape[0] < MIN_TOKENS:
            pass
        elif sol is not None and getattr(sol, "enabled", False):
            warn_once("denoise_sol", "H3 denoising runs on one GPU: Sol (sparse) attention is on.")
        else:
            try:
                ctx.session = _Session(ctx, hidden, segments, rope)
            except _NoRoom as exc:
                ctx.session = None
                _state["last_report"] = f"Last H3 denoising pass on one GPU: {exc}."
                print(f"[Multi-GPU] This denoising pass runs on one GPU: {exc}.")
            except SPFailure:
                raise
            except Exception as exc:
                traceback.print_exc()
                raise SPFailure(f"{type(exc).__name__}: {exc}") from exc
    if ctx.session is None:
        return original(self, x_list, temb, segments, rope)
    return ctx.session.block(index, self, x_list, temb)


def _patched_qkv_attention(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, **kwargs):
    side = getattr(_tls, "sp", None)
    original = _state["orig_qkv_attention"]
    if side is None:
        return original(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, **kwargs)
    for name in ("attention_fn", "attention_mask", "kv_prefix"):
        if kwargs.get(name) is not None:
            raise SPFailure(f"unsupported attention option {name}")
    kwargs = dict(kwargs, attention_fn=side.attention, split_heads=False)
    return original(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, **kwargs)


class _PortableCudart:
    """torch.cuda.cudart() whose cudaHostRegister registers pinned memory as portable (readable by every GPU)."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def cudaHostRegister(self, pointer, size, flags):
        return self._inner.cudaHostRegister(pointer, size, int(flags) | 1)   # cudaHostRegisterPortable


def install():
    if _state["installed"]:
        return
    from shared import attention_kit
    from models.minimax_h3.transformer import DiTBlock, MiniMaxH3Model

    if not getattr(torch.cuda.cudart, "_wangp_mgpu", False):
        original_cudart = torch.cuda.cudart

        def cudart():
            return _PortableCudart(original_cudart())
        cudart._wangp_mgpu = True
        torch.cuda.cudart = cudart
    if not getattr(attention_kit.qkv_attention, "_wangp_mgpu", False):
        _state["orig_qkv_attention"] = attention_kit.qkv_attention
        _patched_qkv_attention._wangp_mgpu = True
        attention_kit.qkv_attention = _patched_qkv_attention
    if not getattr(DiTBlock.forward, "_wangp_mgpu", False):
        _state["orig_block_forward"] = DiTBlock.forward
        _patched_block_forward._wangp_mgpu = True
        DiTBlock.forward = _patched_block_forward
    if not getattr(MiniMaxH3Model.forward, "_wangp_mgpu", False):
        _state["orig_model_forward"] = MiniMaxH3Model.forward
        _patched_model_forward._wangp_mgpu = True
        MiniMaxH3Model.forward = _patched_model_forward
    _state["installed"] = True
