"""Run the last layers of the MiniMax H3 text encoder on a second GPU.

Why
---
H3's text encoder is Qwen3-VL-32B cut after its 50th language layer (about 23 GB in INT8, 465 MB per layer). With WanGP's
usual memory profiles it is not kept in VRAM: every new prompt streams all 50 layers from RAM to the GPU, and the GPU
waits for the weights (the prompt itself is only a few hundred tokens, so the compute is tiny). Encoding a new prompt is
limited by that transfer (about 4.7 s on an RTX 5080, PCIe 5.0 x8, text encoder not pinned).

How
---
The secondary GPU is idle during text encoding and has free VRAM, so it keeps the last N layers resident. A new prompt
then streams only the first 50 - N layers to the primary GPU, as WanGP normally does, hands the hidden state (a few MB)
to the secondary GPU through pinned RAM, which runs its N layers straight from VRAM, and gets the result back. The layers
are the same weights from the same checkpoint, so the output is the same as with one GPU (up to floating point rounding
differences between two GPU models).

The copy of the last N layers is built once, from the checkpoint file, at the first new prompt after enabling the
feature, and stays on the secondary GPU until the plugin is disabled, N is changed, or "Free secondary GPU memory" is
pressed. If the secondary GPU does not have room for N layers (plus a reserve for the VAE copy), fewer layers are used.
"""
import threading
import time
import traceback

import torch
import torch.nn as nn

from . import settings, kernel_compat
from .common import (clear_stale_cuda_error, executor, device_index, gpu_name, map_tensors, match_float_dtypes, module_bytes, other_gpu,
                     to_pinned_host, warn_once)

GB = 1024 ** 3
# VRAM left free on the secondary GPU when it also decodes part of the VAE: the decoder copy (about 2 GB) and its work.
RESERVE_WITH_VAE_GB = 5.0
RESERVE_ALONE_GB = 1.5
# With the denoising split on, the secondary GPU also needs room for streamed blocks and its share of the activations.
RESERVE_WITH_DENOISE_GB = 6.5

_lock = threading.Lock()
_state = {
    "installed": False,
    "orig_encode": None,       # original MiniMaxH3TextEncoder.encode
    "tail": None,              # {"key", "layers", "first", "device", "bytes"}
    "last_report": "No multi-GPU H3 text encoding yet.",
}


def last_report():
    return _state["last_report"]


def release():
    """Drop the text encoder layers held on the secondary GPU and give their memory back."""
    with _lock:
        tail, _state["tail"] = _state["tail"], None
    if tail is None:
        return False
    index = tail["device"]
    tail = None
    if torch.cuda.is_available():
        with torch.cuda.device(index):
            torch.cuda.empty_cache()
    print(f"[Multi-GPU] Freed the text encoder layers on cuda:{index}.")
    return True


# ----------------------------------------------------------------------------------------------------------------------
# Copy of the last layers on the secondary GPU
# ----------------------------------------------------------------------------------------------------------------------

def _min_first_layer(text_encoder):
    """Layers that receive DeepStack visual features (the first 3 for Qwen3-VL) must stay on the primary GPU."""
    indexes = getattr(text_encoder.config.vision_config, "deepstack_visual_indexes", None) or []
    return max(1, len(indexes))


def _load_tail(text_encoder, first):
    """Load language layers [first, count) from the text encoder checkpoint, on the CPU (memory-mapped)."""
    from accelerate import init_empty_weights
    from mmgp import offload
    from models.ideogram4.qwen3_vl_transformers import Qwen3VLTextDecoderLayer
    from models.minimax_h3 import minimax_h3_main

    filename, dtype = text_encoder._wangp_mgpu_te_args
    text_config = text_encoder.config.text_config
    count = len(text_encoder.language_model.layers)

    holder = nn.Module()
    holder.language_model = nn.Module()
    with init_empty_weights(include_buffers=True):
        holder.language_model.layers = nn.ModuleList(
            [nn.Identity() if i < first else Qwen3VLTextDecoderLayer(text_config, i) for i in range(count)])
    prefixes = tuple(f"language_model.layers.{i}." for i in range(first, count))

    def keep_tail(state_dict, quantization_map=None, tied_weights_map=None):
        state_dict = {k: v for k, v in state_dict.items() if k.startswith(prefixes)}
        if quantization_map:
            quantization_map = {k: v for k, v in quantization_map.items() if (k + ".").startswith(prefixes)}
        if tied_weights_map:
            tied_weights_map = {k: [t for t in v if t.startswith(prefixes)] for k, v in tied_weights_map.items()
                                if k.startswith(prefixes)}
        return state_dict, quantization_map, tied_weights_map

    offload.load_model_data(holder, filename, writable_tensors=False, default_dtype=dtype,
                            preprocess_sd=minimax_h3_main._strip_text_encoder_wrapper, postprocess_sd=keep_tail,
                            ignore_unused_weights=True)
    holder.eval().requires_grad_(False)
    return holder


def _tail_for(text_encoder, wanted, index, reserve_gb):
    """The copy of the last layers on cuda:index, built if needed. Returns None when nothing fits."""
    load_args = getattr(text_encoder, "_wangp_mgpu_te_args", None)
    if load_args is None:
        raise RuntimeError("this text encoder was loaded before the plugin was active: restart WanGP")
    count = len(text_encoder.language_model.layers)
    wanted = max(0, min(wanted, count - _min_first_layer(text_encoder)))
    if wanted == 0:
        return None
    key = (repr(load_args), wanted, index, reserve_gb)
    with _lock:
        tail = _state["tail"]
        if tail is not None and tail["key"] == key:
            return tail
    release()

    started = time.perf_counter()
    first = count - wanted
    print(f"[Multi-GPU] Loading text encoder layers {first}-{count - 1} for cuda:{index} (first use only)...")
    holder = _load_tail(text_encoder, first)
    layers = holder.language_model.layers

    # Keep as many of the last layers as the secondary GPU can hold, leaving the reserve free.
    torch.cuda.synchronize(index)
    with torch.cuda.device(index):
        torch.cuda.empty_cache()
    free_bytes, _ = torch.cuda.mem_get_info(index)
    free_bytes += max(0, torch.cuda.memory_reserved(index) - torch.cuda.memory_allocated(index))  # cached by the allocator
    budget = free_bytes - reserve_gb * GB
    sizes = {i: module_bytes(layers[i]) for i in range(first, count)}
    while first < count and sum(sizes[i] for i in range(first, count)) > budget:
        layers[first] = nn.Identity()
        first += 1
    if first >= count:
        warn_once(f"te_vram_{index}", f"Not enough free VRAM on cuda:{index} for any text encoder layer "
                                      f"({free_bytes / GB:.1f} GB free, {reserve_gb:.1f} GB kept in reserve): text encoding stays on one GPU.")
        return None
    if count - first < wanted:
        print(f"[Multi-GPU] Only {count - first} of the {wanted} requested text encoder layers fit on cuda:{index} "
              f"({free_bytes / GB:.1f} GB free, {reserve_gb:.1f} GB kept in reserve).")

    match_float_dtypes(holder, text_encoder)
    holder.to(torch.device("cuda", index))
    tail_layers = [layers[i] for i in range(first, count)]
    total = sum(sizes[i] for i in range(first, count))
    tail = {"key": key, "holder": holder, "layers": tail_layers, "first": first, "device": index, "bytes": total}
    with _lock:
        _state["tail"] = tail
    print(f"[Multi-GPU] Text encoder layers {first}-{count - 1} ready on cuda:{index} ({total / GB:.1f} GB) "
          f"in {time.perf_counter() - started:.1f} s.")
    return tail


# ----------------------------------------------------------------------------------------------------------------------
# Running the last layers on the secondary GPU
# ----------------------------------------------------------------------------------------------------------------------

def _run_tail(layers, host_hidden, host_kwargs, index, owner):
    clear_stale_cuda_error(index)
    device = torch.device("cuda", index)
    with torch.no_grad():
        hidden = host_hidden.to(device, non_blocking=True)
        kwargs = map_tensors(host_kwargs, lambda t: t.to(device, non_blocking=True))
        for layer in layers:
            layer_input = [hidden]
            hidden = None
            hidden = layer(layer_input, **kwargs)
            if getattr(owner, "_interrupt", False):
                break
        out = to_pinned_host(hidden)
    return out


class _RemoteTail(nn.Module):
    """Stands in for the first of the layers held by the secondary GPU, in the layer list of the language model: sends
    the hidden state there, runs all those layers, and brings the result back to the primary GPU."""

    def __init__(self, tail, owner, stats):
        super().__init__()
        self._tail_layers = list(tail["layers"])   # a plain list: not submodules, so WanGP never sees or moves them
        self._index = tail["device"]
        object.__setattr__(self, "_owner", owner)  # the language model: kept out of the submodules, it holds this module
        self._stats = stats

    def forward(self, hidden_states, **kwargs):
        if isinstance(hidden_states, list):
            hidden = hidden_states[0]
            hidden_states.clear()
        else:
            hidden = hidden_states
        primary = hidden.device
        started = time.perf_counter()
        host_hidden = to_pinned_host(hidden)
        hidden = None
        host_kwargs = map_tensors(kwargs, lambda t: to_pinned_host(t) if t.is_cuda else t)
        future = executor(self._index).submit(_run_tail, self._tail_layers, host_hidden, host_kwargs, self._index,
                                              self._owner)
        out = future.result().to(primary, non_blocking=True)
        self._stats["remote_time"] += time.perf_counter() - started
        self._stats["calls"] += 1
        return out


class _ProgressMarker(nn.Module):
    """Takes the place of the other layers held by the secondary GPU: passes the hidden state through, so that WanGP's
    progress bar still counts 50 layers."""

    def forward(self, hidden_states, **kwargs):
        return hidden_states[0] if isinstance(hidden_states, list) else hidden_states


def _patched_encode(self, prompt, items, device, dtype):
    orig = _state["orig_encode"]
    cfg = settings.get()
    if not cfg["enabled"] or not cfg["te_enabled"] or cfg["te_layers_cuda1"] <= 0 or torch.cuda.device_count() < 2:
        return orig(self, prompt, items, device, dtype)
    if cfg.get("denoise_enabled"):
        # The secondary GPU's VRAM goes to denoising, worth far more than the text encoder layers (a fraction of a second).
        if release():
            print("[Multi-GPU] Text encoder layers dropped from the secondary GPU: its VRAM is kept for the denoising split.")
        warn_once("te_vs_denoise", "Text encoder split skipped while the denoising split is on (the secondary GPU's VRAM "
                                   "goes to denoising).")
        return orig(self, prompt, items, device, dtype)
    try:
        primary_index = device_index(device)
        index = other_gpu(primary_index)
        if kernel_compat.prepare_secondary(primary_index, index) is not None:
            return orig(self, prompt, items, device, dtype)
        reserve = RESERVE_WITH_VAE_GB if cfg["vae_enabled"] else RESERVE_ALONE_GB
        if cfg.get("denoise_enabled"):
            reserve = max(reserve, RESERVE_WITH_DENOISE_GB)
        tail = _tail_for(self, cfg["te_layers_cuda1"], index, reserve)
    except Exception as exc:
        warn_once("te_tail", f"Multi-GPU text encoding unavailable, falling back to one GPU: {exc}")
        traceback.print_exc()
        return orig(self, prompt, items, device, dtype)
    if tail is None:
        return orig(self, prompt, items, device, dtype)

    language_model = self.language_model
    original_layers = language_model.layers
    count = len(original_layers)
    first = tail["first"]
    stats = {"remote_time": 0.0, "calls": 0}
    runner = _RemoteTail(tail, language_model, stats)
    language_model.layers = nn.ModuleList(list(original_layers[:first]) + [runner] +
                                          [_ProgressMarker() for _ in range(count - first - 1)])
    started = time.perf_counter()
    try:
        result = orig(self, prompt, items, device, dtype)
    finally:
        language_model.layers = original_layers
        if stats["calls"]:
            total = time.perf_counter() - started
            report = (f"Last H3 text encoding: {total:.1f} s, layers 0-{first - 1} on cuda:{primary_index} "
                      f"({gpu_name(primary_index)}), layers {first}-{count - 1} on cuda:{index} ({gpu_name(index)}, "
                      f"{stats['remote_time']:.2f} s including transfers).")
            _state["last_report"] = report
            print(f"[Multi-GPU] {report}")
    return result


# ----------------------------------------------------------------------------------------------------------------------
# Installation (once, at WanGP startup, before any model is loaded)
# ----------------------------------------------------------------------------------------------------------------------

def _wrap_loader(orig_loader):
    def load_text_encoder(filename, dtype, *args, **kwargs):
        text_encoder = orig_loader(filename, dtype, *args, **kwargs)
        try:
            text_encoder._wangp_mgpu_te_args = (tuple(filename) if isinstance(filename, list) else filename, dtype)
        except Exception:
            pass
        return text_encoder
    load_text_encoder._wangp_mgpu = True
    return load_text_encoder


def install():
    if _state["installed"]:
        return
    from models.minimax_h3 import minimax_h3_main
    from models.minimax_h3.text_encoder import MiniMaxH3TextEncoder

    if not getattr(MiniMaxH3TextEncoder.encode, "_wangp_mgpu", False):
        _state["orig_encode"] = MiniMaxH3TextEncoder.encode
        _patched_encode._wangp_mgpu = True
        MiniMaxH3TextEncoder.encode = _patched_encode
    if not getattr(minimax_h3_main._load_text_encoder, "_wangp_mgpu", False):
        minimax_h3_main._load_text_encoder = _wrap_loader(minimax_h3_main._load_text_encoder)
    _state["installed"] = True
