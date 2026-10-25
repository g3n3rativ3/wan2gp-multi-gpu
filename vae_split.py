"""Split MiniMax H3 VAE decoding across two GPUs.

How it works
------------
H3's VAE decodes each temporal clip as one batch of spatial tiles (``AutoencoderKLMiniMaxH3._decode_clip`` stacks the tiles
along the batch axis and calls the ViT decoder once). Every tile is decoded independently (self-attention never crosses
tiles), so the batch can be cut in two: the first part runs on the primary GPU (where WanGP put the VAE), the rest on a
secondary GPU that holds its own copy of the decoder. The two parts run at the same time, then the secondary part is
copied back and concatenated. The tile stitching and blending code of WanGP is untouched, so the output is the same as
with one GPU (up to floating point rounding differences between two GPU models).

The secondary copy of the decoder is built once, from the same checkpoint file as the primary one, and stays on the
secondary GPU until the plugin is disabled or "Free secondary GPU memory" is pressed.
"""
import threading
import time
import traceback

import torch

from . import settings, kernel_compat
from .common import clear_stale_cuda_error, executor as _executor, gpu_name, match_float_dtypes, other_gpu, to_pinned_host as _to_pinned_host
from .common import warn_once as _warn_once

_lock = threading.Lock()
_state = {
    "installed": False,
    "orig_forward": None,       # original MiniMaxH3VideoViTDecoder3d.forward
    "orig_loader": None,        # original minimax_h3_main._load_video_vae
    "replica": None,            # decoder copy on the secondary GPU
    "replica_key": None,
    "decode": None,             # statistics of the VAE decode in progress
    "last_report": "No multi-GPU H3 VAE decode yet.",
}


def last_report():
    return _state["last_report"]


# ----------------------------------------------------------------------------------------------------------------------
# Secondary decoder copy
# ----------------------------------------------------------------------------------------------------------------------

def release():
    """Drop the decoder copy held on the secondary GPU and give its memory back."""
    with _lock:
        replica_key = _state["replica_key"]
        _state["replica"] = None
        _state["replica_key"] = None
    if replica_key is not None and torch.cuda.is_available():
        device_index = replica_key[1]
        with torch.cuda.device(device_index):
            torch.cuda.empty_cache()
        print(f"[Multi-GPU] Freed the VAE decoder copy on cuda:{device_index}.")
        return True
    return False


def _replica_for(decoder, device_index):
    load_args = getattr(decoder, "_wangp_mgpu_load_args", None)
    if load_args is None:
        raise RuntimeError("this VAE was loaded before the plugin was active: restart WanGP")
    key = (load_args, device_index)
    with _lock:
        if _state["replica_key"] == key and _state["replica"] is not None:
            return _state["replica"]
    release()
    started = time.perf_counter()
    print(f"[Multi-GPU] Building a copy of the H3 VAE decoder on cuda:{device_index} (first use only)...")
    vae = _state["orig_loader"](*load_args[0], **dict(load_args[1]))
    replica = vae.decoder
    vae = None  # the encoder and the rest of the copy are not needed
    match_float_dtypes(replica, decoder)
    replica.to(torch.device("cuda", device_index))
    replica.eval().requires_grad_(False)
    replica._wangp_mgpu_is_replica = True
    with _lock:
        _state["replica"], _state["replica_key"] = replica, key
    print(f"[Multi-GPU] Decoder copy ready on cuda:{device_index} in {time.perf_counter() - started:.1f} s.")
    return replica


# ----------------------------------------------------------------------------------------------------------------------
# The split itself
# ----------------------------------------------------------------------------------------------------------------------

def split_count(batch, primary_index, share_cuda0):
    """Number of batch items (tiles) kept on the primary GPU."""
    share = share_cuda0 / 100.0 if primary_index == 0 else 1.0 - share_cuda0 / 100.0
    return max(0, min(batch, int(round(batch * share))))


def _run_secondary(replica, host_input, tokens, device_index):
    clear_stale_cuda_error(device_index)
    with torch.no_grad():
        hidden_states = host_input.to(torch.device("cuda", device_index), non_blocking=True)
        out = _state["orig_forward"](replica, hidden_states, tokens)
        return _to_pinned_host(out)


def _patched_forward(self, hidden_states, tokens=False):
    orig = _state["orig_forward"]
    cfg = settings.get()
    if (not cfg["enabled"] or not cfg["vae_enabled"] or getattr(self, "_wangp_mgpu_is_replica", False) or not torch.is_tensor(hidden_states)
            or not hidden_states.is_cuda or hidden_states.dim() != 5 or hidden_states.shape[0] < 2
            or torch.cuda.device_count() < 2):
        return orig(self, hidden_states, tokens)
    batch = hidden_states.shape[0]
    primary = hidden_states.device
    primary_index = primary.index if primary.index is not None else torch.cuda.current_device()
    secondary_index = other_gpu(primary_index)
    keep = split_count(batch, primary_index, cfg["share_cuda0"])
    if keep >= batch or kernel_compat.prepare_secondary(primary_index, secondary_index) is not None:
        return orig(self, hidden_states, tokens)
    try:
        replica = _replica_for(self, secondary_index)
    except Exception as exc:
        _warn_once("replica", f"Multi-GPU VAE decoding unavailable, falling back to one GPU: {exc}")
        traceback.print_exc()
        return orig(self, hidden_states, tokens)

    started = time.perf_counter()
    remote_input = _to_pinned_host(hidden_states[keep:])
    replica._interrupt = getattr(self, "_interrupt", False)
    future = _executor(secondary_index).submit(_run_secondary, replica, remote_input, tokens, secondary_index)
    try:
        local = orig(self, hidden_states[:keep], tokens) if keep > 0 else None
    except BaseException:
        try:   # nothing left running on the secondary GPU when the decode stops here
            future.result()
        except BaseException:
            pass
        raise
    try:
        remote = future.result().to(primary, non_blocking=True)
    except Exception as exc:   # the secondary GPU failed: its tiles are decoded here, the video is still complete
        _warn_once(f"remote_{type(exc).__name__}", f"The secondary GPU could not decode its VAE tiles ({type(exc).__name__}: "
                                                   f"{exc}): they are decoded on cuda:{primary_index}.")
        traceback.print_exc()
        remote = orig(self, hidden_states[keep:], tokens)
        if _state["decode"] is not None:
            _state["decode"]["failed"] = f"{type(exc).__name__}: {exc}"
        keep = batch
    out = remote if local is None else torch.cat((local, remote), dim=0)

    stats = _state["decode"]
    if stats is not None:
        stats["calls"] += 1
        stats["tiles"] = batch
        stats["local"] = keep
        stats["split_time"] += time.perf_counter() - started
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Installation (once, at WanGP startup, before any model is loaded)
# ----------------------------------------------------------------------------------------------------------------------

def _wrap_loader(orig_loader):
    def load_video_vae(*args, **kwargs):
        vae = orig_loader(*args, **kwargs)
        try:
            vae.decoder._wangp_mgpu_load_args = (tuple(args), tuple(sorted(kwargs.items())))
        except Exception:
            pass
        return vae
    load_video_vae._wangp_mgpu = True
    return load_video_vae


def _wrap_decode(orig_decode):
    def decode(self, *args, **kwargs):
        cfg = settings.get()
        if not cfg["enabled"] or not cfg["vae_enabled"]:
            return orig_decode(self, *args, **kwargs)
        _state["decode"] = {"calls": 0, "tiles": 0, "local": 0, "split_time": 0.0}
        started = time.perf_counter()
        failed = None
        try:
            return orig_decode(self, *args, **kwargs)
        except BaseException as exc:
            failed = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            stats, _state["decode"] = _state["decode"], None
            total = time.perf_counter() - started
            if failed is not None:
                report = f"Last H3 VAE decode stopped after {total:.1f} s by an error ({failed[:200]})."
            elif stats and stats["calls"]:
                primary = torch.cuda.current_device()
                secondary = other_gpu(primary)
                report = (f"Last H3 VAE decode: {total:.1f} s, {stats['calls']} clip(s) of {stats['tiles']} tiles: "
                          f"{stats['local']} on cuda:{primary} ({gpu_name(primary)}), "
                          f"{stats['tiles'] - stats['local']} on cuda:{secondary} ({gpu_name(secondary)})"
                          + (f"; the secondary GPU failed ({stats['failed'][:200]}), its tiles were decoded on "
                             f"cuda:{primary}." if stats.get("failed") else "."))
            else:
                report = f"Last H3 VAE decode: {total:.1f} s on one GPU (nothing to split: tiling off, a single tile, or share at 100/0)."
            _state["last_report"] = report
            print(f"[Multi-GPU] {report}")
    decode._wangp_mgpu = True
    return decode


def install():
    if _state["installed"]:
        return
    from models.minimax_h3 import minimax_h3_main
    from models.minimax_h3.components.video_autoencoder import MiniMaxH3VideoViTDecoder3d
    from models.minimax_h3.video_vae import MiniMaxH3VideoVAE

    if not getattr(MiniMaxH3VideoViTDecoder3d.forward, "_wangp_mgpu", False):
        _state["orig_forward"] = MiniMaxH3VideoViTDecoder3d.forward
        _patched_forward._wangp_mgpu = True
        MiniMaxH3VideoViTDecoder3d.forward = _patched_forward
    if not getattr(minimax_h3_main._load_video_vae, "_wangp_mgpu", False):
        _state["orig_loader"] = minimax_h3_main._load_video_vae
        minimax_h3_main._load_video_vae = _wrap_loader(minimax_h3_main._load_video_vae)
    if not getattr(MiniMaxH3VideoVAE.decode, "_wangp_mgpu", False):
        MiniMaxH3VideoVAE.decode = _wrap_decode(MiniMaxH3VideoVAE.decode)
    _state["installed"] = True
