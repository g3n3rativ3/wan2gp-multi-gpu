"""CPU check of the LTX-2 VAE split (WanGP v18.17+, streamed decoder): spatial tiles shared between two "GPUs" (the CPU
and a thread with its own decoder copy) must give exactly the video of the one-device streamed decode."""
import importlib
import os
import sys
import types

import torch

WANGP = sys.argv[1]
PLUGINS = sys.argv[2]
sys.path.insert(0, WANGP)
sys.path.insert(0, PLUGINS)
_props = types.SimpleNamespace(name="CPU stand-in", total_memory=16 << 30, L2_cache_size=4 << 20, multi_processor_count=64,
                               major=8, minor=9, gcnArchName="", shared_memory_per_block=48 << 10,
                               shared_memory_per_block_optin=99 << 10)
torch.cuda.get_device_capability = lambda *a, **k: (8, 9)
torch.cuda.current_device = lambda: 0
torch.cuda.get_device_properties = lambda *a, **k: _props
shared = types.ModuleType("shared")
shared.__path__ = [os.path.join(WANGP, "shared")]
sys.modules["shared"] = shared
attention_stub = types.ModuleType("shared.attention")
attention_stub.pay_attention = None
attention_stub.sage_staged = lambda *args, **kwargs: None
attention_stub.sage_attention_mask = lambda mask, dtype: mask
sys.modules["shared.attention"] = attention_stub

from models.ltx2.ltx_core.model.video_vae import video_vae  # noqa: E402
from models.ltx2.ltx_core.model.video_vae.streaming import DecoderStream  # noqa: E402
from models.ltx2.ltx_core.model.video_vae.tiling import SpatialTilingConfig, TilingConfig  # noqa: E402

PLUGIN = os.environ.get("MULTIGPU_PLUGIN", "wan2gp-multi-gpu")
vs = importlib.import_module(f"{PLUGIN}.ltx2_vae")
settings = importlib.import_module(f"{PLUGIN}.settings")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strict_device import StrictDeviceContext, violations  # noqa: E402
torch.manual_seed(0)

decoder = video_vae.VideoDecoder(in_channels=16, out_channels=3, base_channels=4, patch_size=4,
                                 decoder_blocks=[("compress_all", {"multiplier": 2}), ("res_x", {"num_layers": 1}),
                                                 ("compress_all", {"multiplier": 1}), ("compress_all", {"multiplier": 1})])
with torch.no_grad():
    for p in decoder.parameters():
        p.copy_(torch.randn_like(p) * 0.1)
decoder.eval()
latent = torch.randn(1, 16, 5, 32, 48)   # 33 frames, 1024x1536

configs = {
    "tiling 64 px": TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=64, tile_overlap_in_pixels=32),
                                 temporal_config=None),
    "tiling 64 px, no overlap": TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=64,
                                                                                 tile_overlap_in_pixels=0),
                                             temporal_config=None),
    "tiling 128 px": TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=128, tile_overlap_in_pixels=32),
                                  temporal_config=None),
}

vs.install()
vs._state["test_secondary"] = torch.device("cpu")
all_ok = True
for name, cfg in configs.items():
    tiles = len(video_vae._stream_tiles(latent.shape, cfg, video_vae.STREAM_TILE_AREA, video_vae.STREAM_FRAME_SHARE, 32,
                                        latent.device))
    with torch.inference_mode(), StrictDeviceContext("cpu"):
        settings.update(enabled=False)
        ref = video_vae.decode_video_to_tensor(latent, decoder, cfg)
        for share in (50, 30, 75):
            settings.update(enabled=True, ltx_vae_enabled=True, ltx_vae_share_cuda0=share)
            vs.release()
            vs._state["last_report"] = "none"
            out = video_vae.decode_video_to_tensor(latent, decoder, cfg)
            diff = (out.int() - ref.int()).abs().max().item()
            report = vs.last_report()
            remote = len(vs._remote_tiles(tiles, 1.0 - share / 100.0))
            expected = (f" {remote} on cpu" in report and report.startswith("Last LTX-2 VAE decode:")) if remote else \
                "gives no tile to the secondary GPU" in report
            ok = tiles >= 2 and out.shape == ref.shape and diff == 0 and expected
            all_ok &= ok
            print(f"{'OK  ' if ok else 'FAIL'} {name} ({tiles} tiles), share {share}: {tuple(out.shape)}, max diff {diff} | "
                  f"{report}")

# a single tile: nothing to share, one device, same output
cfg = TilingConfig(spatial_config=SpatialTilingConfig(tile_size_in_pixels=2048, tile_overlap_in_pixels=128), temporal_config=None)
with torch.inference_mode():
    settings.update(enabled=False)
    ref = video_vae.decode_video_to_tensor(latent, decoder, cfg)
    settings.update(enabled=True, ltx_vae_share_cuda0=50)
    out = video_vae.decode_video_to_tensor(latent, decoder, cfg)
ok = torch.equal(out, ref) and "single spatial tile" in vs.last_report()
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} single tile stays on one device | {vs.last_report()}")

# a failing secondary decode: its tiles are decoded on the primary device, same output
orig_step = DecoderStream.step


def flaky(self, latent, last):
    if self.decoder.__dict__.get("_wangp_mgpu_ltx2_replica"):
        raise RuntimeError("simulated out of memory on the secondary GPU")
    return orig_step(self, latent, last)


DecoderStream.step = flaky
cfg = configs["tiling 64 px"]
with torch.inference_mode():
    settings.update(enabled=False)
    ref = video_vae.decode_video_to_tensor(latent, decoder, cfg)
    settings.update(enabled=True, ltx_vae_share_cuda0=50)
    out = video_vae.decode_video_to_tensor(latent, decoder, cfg)
DecoderStream.step = orig_step
ok = torch.equal(out, ref) and "finished on one GPU" in vs.last_report()
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} failure on the secondary side: same output | {vs.last_report()}")

# interruption in the middle of the decode: nothing left running
calls = {"n": 0}


def interrupt():
    calls["n"] += 1
    return calls["n"] > 3


with torch.inference_mode():
    settings.update(enabled=True, ltx_vae_share_cuda0=50)
    out = video_vae.decode_video_to_tensor(latent, decoder, configs["tiling 64 px"], interrupt_check=interrupt)
ok = out is None
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} interruption")

ok = not violations
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} no allocation without an explicit device inside WanGP's device context {violations[:3]}")
print("ALL OK" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
