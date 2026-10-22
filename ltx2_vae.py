"""Split LTX-2 VAE decoding across two GPUs (WanGP v18.17+, streamed decoder).

How LTX-2 decodes (WanGP v18.17+)
---------------------------------
``VideoDecoder.tiled_decode`` cuts the frame into spatial tiles (the fewest that fit the "VAE tiling" setting: a 720p or
1080p frame is often a single tile) and decodes each tile on its own, streamed latent frame after latent frame through a
``DecoderStream`` that carries the state of its temporal convolutions: every output frame is exactly the one the whole
video would give, without temporal tiles. The frames of a tile go to a ``sink`` that writes what no other tile covers and
blends the overlaps on the CPU.

The split
---------
The time axis cannot be cut (each frame depends on the state left by the previous ones), so the tiles are shared: tile 0
stays on the primary GPU (so that both GPUs start at once), the others are spread evenly according to the VAE share. The
secondary GPU decodes its tiles ahead of time with its own copy of the decoder, recording what its stream hands to the
sink at each latent frame. When WanGP's loop reaches one of these tiles, the plugin feeds the recorded frames to WanGP's
own sink in the same order instead of decoding them again: the blending is WanGP's, unchanged, and the video is the same
as on one GPU. With a single tile there is nothing to share and the decode stays on one GPU.

If the secondary GPU fails to decode a tile, that tile and the remaining ones are decoded on the primary GPU (a tile is
handed over only once the secondary GPU has decoded all of it, so a failure never leaves a tile half written).

The copy of the decoder on the secondary GPU is built once from the tensors WanGP's memory manager already holds in RAM,
and stays there until the plugin is disabled, "Free secondary GPU memory" is pressed or another decoder is used.
"""
import threading
import time
import traceback
import weakref

import torch
import torch.nn as nn

from . import kernel_compat, ltx2_denoise, settings
from .clones import CloneError, clone_module, mmgp_index, own_modules, registry, strip_mmgp
from .common import clear_stale_cuda_error, executor, gpu_name, other_gpu, warn_once

WAIT_POLL = 0.05
_MISSING = object()

_lock = threading.Lock()
_state = {
    "installed": False,
    "orig_tiled_decode": None,
    "replica": None,           # (weakref to the decoder, device, copy)
    "last_report": "No multi-GPU LTX-2 VAE decode yet.",
    "test_secondary": None,    # tests only: a device standing for the secondary GPU
}


def last_report():
    return _state["last_report"]


# ----------------------------------------------------------------------------------------------------------------------
# Decoder copy on the secondary GPU
# ----------------------------------------------------------------------------------------------------------------------

def release():
    """Drop the decoder copy held on the secondary GPU and give its memory back."""
    with _lock:
        replica, _state["replica"] = _state["replica"], None
    if replica is None:
        return False
    device = replica[1]
    replica = None
    if torch.device(device).type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    print(f"[Multi-GPU] Freed the LTX-2 VAE decoder copy on {device}.")
    return True


def _replica_for(decoder, device):
    with _lock:
        held = _state["replica"]
        if held is not None and held[0]() is decoder and held[1] == device:
            return held[2]
    release()
    started = time.perf_counter()
    print(f"[Multi-GPU] Building a copy of the LTX-2 VAE decoder on {device} (first use only)...")
    memo = {}
    try:
        replica = clone_module(decoder, memo)
    except CloneError as exc:
        raise RuntimeError(str(exc)) from exc
    entries = registry(decoder, memo, mmgp_index(own_modules(decoder)))
    moved = {}
    for module, name, tensor, is_buffer, tied in entries:
        key = id(tensor)
        if key not in moved:
            source = tensor.data if isinstance(tensor, nn.Parameter) else tensor
            copy = source.to(device)
            moved[key] = copy if is_buffer else nn.Parameter(copy, requires_grad=False)
        value = moved[key]
        if is_buffer:
            module._buffers[name] = value
        else:
            module._parameters[name] = value
        if tied is not None:
            if tied[1] in tied[0]._buffers:
                tied[0]._buffers[tied[1]] = value
            else:
                tied[0]._parameters[tied[1]] = value
    entries = moved = None
    strip_mmgp(replica)
    for name in ("stream_step", "tiled_decode"):   # instance wrappers of the original (MMGP's load hook): not for the copy
        replica.__dict__.pop(name, None)
    replica.__dict__["_wangp_mgpu_ltx2_replica"] = True
    replica.eval()
    with _lock:
        _state["replica"] = (weakref.ref(decoder), device, replica)
    print(f"[Multi-GPU] LTX-2 decoder copy ready on {device} in {time.perf_counter() - started:.1f} s.")
    return replica


# ----------------------------------------------------------------------------------------------------------------------
# One tiled decode
# ----------------------------------------------------------------------------------------------------------------------

def _remote_tiles(count, share_remote):
    """Spread the remote tiles evenly (tile 0 always stays on the primary GPU, so both GPUs start at once)."""
    remote = []
    for i in range(1, count):
        if int((i + 1) * share_remote) > int(i * share_remote):
            remote.append(i)
    return remote


class _Plan:
    """Which tiles the secondary GPU decodes, and what its streams handed to their sink, per tile and latent frame."""

    def __init__(self, latent, tiles, remote, device, piece_frames):
        self.tiles = tiles
        self.frames = latent.shape[2]
        self.remote = remote                  # sorted indexes of the tiles decoded on the secondary GPU
        self.remote_set = set(remote)
        self.device = device
        self.piece_frames = piece_frames
        self.cond = threading.Condition()
        self.results = {}                     # tile -> [outputs of each latent frame (list of CPU tensors)] or exception
        self.abort = False
        self.disabled = None                  # reason why the remaining remote tiles are decoded on the primary GPU
        self.calls = 0
        self.started = time.perf_counter()
        self.local_tiles = self.remote_tiles = 0
        self.local_time = self.remote_time = self.wait_time = 0.0
        self.tile_started = None
        self.share0 = 50
        self.primary_is_cuda0 = True
        self.latent_host = latent.detach().to("cpu")
        self.future = None

    def stop(self):
        with self.cond:
            self.abort = True
            self.cond.notify_all()

    def job(self, replica, stream_class):
        index = getattr(self.device, "index", None)
        clear_stale_cuda_error(-1 if self.device.type != "cuda" else index)
        try:
            with torch.inference_mode(), torch.device(self.device):
                latent = self.latent_host.to(self.device)
                for tile_index in self.remote:
                    with self.cond:   # at most one decoded tile waiting in RAM ahead of its use
                        while not self.abort and sum(1 for v in self.results.values() if not isinstance(v, Exception)) >= 1:
                            self.cond.wait(WAIT_POLL)
                        if self.abort:
                            return
                    started = time.perf_counter()
                    try:
                        steps, current = [], []
                        stream = stream_class(replica, self.piece_frames, lambda out: current.append(out.to("cpu")))
                        part = latent[self.tiles[tile_index].in_coords]
                        for latent_index in range(self.frames):
                            if self.abort:
                                return
                            stream.step(part[:, :, latent_index:latent_index + 1], latent_index == self.frames - 1)
                            steps.append(current)
                            current = []
                        result = steps
                    except Exception as exc:   # this tile (and the next ones) are decoded on the primary GPU
                        traceback.print_exc()
                        result = exc
                    with self.cond:
                        self.results[tile_index] = result
                        self.remote_time += time.perf_counter() - started
                        self.cond.notify_all()
                    if isinstance(result, Exception):
                        return
        finally:
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()

    def take(self, tile_index, interrupt_check):
        """The secondary GPU's decode of a whole tile (outputs per latent frame), or None (decode it on the primary GPU)."""
        started = time.perf_counter()
        with self.cond:
            while tile_index not in self.results:
                if self.abort or (self.future is not None and self.future.done()):
                    break
                if interrupt_check is not None and interrupt_check():
                    break
                self.cond.wait(WAIT_POLL)
            result = self.results.pop(tile_index, None)
            self.cond.notify_all()
        self.wait_time += time.perf_counter() - started
        if isinstance(result, Exception):
            self.disabled = f"{type(result).__name__}: {result}"
            warn_once(f"ltx_vae_remote_{self.disabled}", f"The secondary GPU could not decode an LTX-2 VAE tile "
                                                         f"({self.disabled}): the remaining tiles are decoded on the "
                                                         f"primary GPU.")
            return None
        return result


def _make_plan(decoder, latent, tiling_config):
    cfg = settings.get()
    if not cfg["enabled"] or not cfg["ltx_vae_enabled"] or decoder.__dict__.get("_wangp_mgpu_ltx2_replica"):
        return None
    test = _state["test_secondary"]
    if test is None and (not torch.is_tensor(latent) or not latent.is_cuda or torch.cuda.device_count() < 2):
        return None
    if getattr(decoder, "timestep_conditioning", False):
        return None
    from models.ltx2.ltx_core.model.video_vae import video_vae
    tiles = video_vae._stream_tiles(latent.shape, tiling_config, video_vae.STREAM_TILE_AREA, video_vae.STREAM_FRAME_SHARE,
                                    decoder.video_downscale_factors.height, latent.device)
    if len(tiles) < 2:
        _report_skip("Last LTX-2 VAE decode on one GPU: a single spatial tile, nothing to share (WanGP's Low VRAM VAE "
                     "tiling, Configuration > RAM/VRAM Management > VAE Tiling, gives more tiles).")
        return None
    if test is not None:
        device = test
    else:
        primary_index = latent.device.index if latent.device.index is not None else torch.cuda.current_device()
        secondary_index = other_gpu(primary_index)
        if kernel_compat.prepare_secondary(primary_index, secondary_index) is not None:
            return None
        device = torch.device("cuda", secondary_index)
    share0 = cfg["ltx_vae_share_cuda0"] / 100.0
    if test is None and device.index == 0:   # WanGP runs on cuda:1: the share setting is about cuda:0
        share0 = 1.0 - share0
    remote = _remote_tiles(len(tiles), 1.0 - share0)
    if not remote:
        _report_skip(f"Last LTX-2 VAE decode on one GPU: with {len(tiles)} spatial tiles, a share of "
                     f"{cfg['ltx_vae_share_cuda0']} % gives no tile to the secondary GPU.")
        return None
    try:
        replica = _replica_for(decoder, device)
    except Exception as exc:
        warn_once("ltx_vae_replica", f"Multi-GPU LTX-2 VAE decoding unavailable, falling back to one GPU: {exc}")
        traceback.print_exc()
        return None
    plan = _Plan(latent, tiles, remote, device, video_vae.STREAM_PIECE_FRAMES)
    plan.share0 = cfg["ltx_vae_share_cuda0"]
    plan.primary_is_cuda0 = test is not None or device.index != 0
    plan.future = executor(-1 if device.type != "cuda" else device.index).submit(plan.job, replica, video_vae.DecoderStream)
    return plan


def _report_skip(report):
    """A decode the split does not apply to is reported too, so that the console shows why both GPUs were not used."""
    _state["last_report"] = report
    print(f"[Multi-GPU] {report}")


def _plan_step(plan, local_step, interrupt_check):
    """stream_step of the decoder during a split decode: local tiles go to WanGP's (MMGP-hooked) step, remote tiles get
    the frames the secondary GPU decoded, fed to WanGP's own sink."""
    def step(stream, latent, last):
        call = plan.calls
        plan.calls += 1
        tile_index, latent_index = divmod(call, plan.frames)
        if latent_index == 0:
            plan.current = None
            if tile_index in plan.remote_set and plan.disabled is None:
                plan.current = plan.take(tile_index, interrupt_check)
            if plan.current is None:
                plan.local_tiles += 1
            else:
                plan.remote_tiles += 1
            plan.tile_started = time.perf_counter()
        if plan.current is None:
            out = local_step(stream, latent, last)
            if last:
                if latent.is_cuda:
                    torch.cuda.synchronize(latent.device)
                plan.local_time += time.perf_counter() - plan.tile_started
            return out
        if last != (latent_index == plan.frames - 1):
            raise RuntimeError("unexpected LTX-2 VAE stream order")
        for out in plan.current[latent_index]:
            stream.sink(out.to(latent.device))
        plan.current[latent_index] = None
        return None
    return step


def _patched_tiled_decode(self, latent, tiling_config, write, interrupt_check=None, *args, **kwargs):
    original = _state["orig_tiled_decode"]
    try:   # the denoising is over: the secondary GPU's VRAM goes to the decoder copy
        ltx2_denoise.end_of_denoising()
    except Exception as exc:
        warn_once("ltx_vae_end_denoise", f"Could not free the denoising memory of the secondary GPU: {exc}")
    plan = None
    try:
        plan = _make_plan(self, latent, tiling_config)
    except Exception as exc:
        warn_once("ltx_vae_plan", f"Multi-GPU LTX-2 VAE decoding unavailable, falling back to one GPU: {exc}")
        traceback.print_exc()
    if plan is None:
        return original(self, latent, tiling_config, write, interrupt_check, *args, **kwargs)
    saved = self.__dict__.get("stream_step", _MISSING)
    local_step = saved if saved is not _MISSING else type(self).stream_step.__get__(self)
    self.__dict__["stream_step"] = _plan_step(plan, local_step, interrupt_check)
    completed = False
    try:
        result = original(self, latent, tiling_config, write, interrupt_check, *args, **kwargs)
        completed = bool(result)
        return result
    finally:
        if saved is _MISSING:
            self.__dict__.pop("stream_step", None)
        else:
            self.__dict__["stream_step"] = saved
        plan.stop()
        try:
            plan.future.result(timeout=120)
        except Exception:
            pass
        total = time.perf_counter() - plan.started
        if completed:
            secondary = plan.device
            name = gpu_name(secondary.index) if secondary.type == "cuda" else str(secondary)
            note = f" ({plan.disabled}: finished on one GPU)" if plan.disabled else ""
            report = (f"Last LTX-2 VAE decode: {total:.1f} s, {len(plan.tiles)} spatial tiles: {plan.local_tiles} on the "
                      f"primary GPU ({plan.local_time:.1f} s of decoding), {plan.remote_tiles} on {secondary} ({name}, "
                      f"{plan.remote_time:.1f} s of work); the primary GPU waited {plan.wait_time:.1f} s for them{note}.")
            _state["last_report"] = report
            print(f"[Multi-GPU] {report}")


def install():
    if _state["installed"]:
        return
    from models.ltx2.ltx_core.model.video_vae.video_vae import VideoDecoder
    if not hasattr(VideoDecoder, "stream_step"):
        raise RuntimeError("this WanGP version decodes LTX-2 without the streamed decoder (WanGP v18.17+ needed)")
    if not getattr(VideoDecoder.tiled_decode, "_wangp_mgpu_ltx2", False):
        _state["orig_tiled_decode"] = VideoDecoder.tiled_decode
        _patched_tiled_decode._wangp_mgpu_ltx2 = True
        VideoDecoder.tiled_decode = _patched_tiled_decode
    _state["installed"] = True
