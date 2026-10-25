"""Multi-GPU plugin for WanGP.

MiniMax H3: denoising split across cuda:0 and cuda:1 by token rows (denoise_split.py), VAE decoding split by tiles
(vae_split.py), and the last layers of the text encoder kept on cuda:1 (te_split.py).
LTX-2 (since v0.5): denoising split by video rows (ltx2_denoise.py) and VAE decoding split by spatial tiles
(ltx2_vae.py), ported from the "Multi-GPU LTX-2" plugin for WanGP 11.22.
See README.md. The settings live in a "Multi-GPU" tab added after the "Misc." tab of the Media Generator advanced settings.
"""
import traceback

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import denoise_split, settings, te_split, vae_split

try:   # LTX-2 support: optional, a WanGP without LTX-2 (or with another version of it) still gets the H3 features
    from . import ltx2_denoise, ltx2_vae
except Exception as _ltx_import_error:   # noqa: N816
    ltx2_denoise = ltx2_vae = None
    print(f"[Multi-GPU] LTX-2 support unavailable: {_ltx_import_error}")
from .common import gpu_list

PLUGIN_NAME = "Multi-GPU"


def _gpu_summary():
    gpus = gpu_list()
    if not gpus:
        return "**No CUDA GPU visible.** Multi-GPU is unavailable."
    lines = [f"- `cuda:{i}`: {name} ({total:.0f} GB)" for i, name, total in gpus]
    text = "**GPUs visible to WanGP**\n\n" + "\n".join(lines)
    if len(gpus) < 2:
        text += ("\n\n**Only one GPU is visible.** If you start WanGP with `--gpu`, remove that option: it hides the other "
                 "GPUs (it sets `CUDA_VISIBLE_DEVICES`).")
    return text


def _split_summary(share):
    gpus = {i: name for i, name, _ in gpu_list()}
    return (f"Work split: **{share} %** on `cuda:0` ({gpus.get(0, 'not found')}), "
            f"**{100 - share} %** on `cuda:1` ({gpus.get(1, 'not found')}).")


def _te_summary(layers):
    if layers <= 0:
        return "Text encoder: all 50 layers on `cuda:0`, as usual."
    return (f"Text encoder: layers 0-{49 - layers} streamed to `cuda:0` as usual, the last **{layers}** kept on `cuda:1` "
            f"(about {layers * 0.47:.1f} GB of `cuda:1` VRAM with the INT8 text encoder, {layers * 0.93:.1f} GB with BF16).")


def _status(cfg=None):
    cfg = cfg or settings.get()
    if not cfg["enabled"]:
        state = "**Disabled**: WanGP uses one GPU, as usual."
    else:
        parts = [name for name, on in (("H3 denoising" + (" (videos and images)" if cfg["denoise_images"] else " (videos)"),
                                        cfg["denoise_enabled"]),
                                       ("H3 VAE decoding", cfg["vae_enabled"]),
                                       ("H3 text encoder", cfg["te_enabled"] and cfg["te_layers_cuda1"] > 0)) if on]
        parts += [name for name, on in (("LTX-2 denoising", cfg["ltx_denoise_enabled"]),
                                        ("LTX-2 VAE decoding", cfg["ltx_vae_enabled"])) if on and ltx2_denoise is not None]
        state = f"**Enabled** for: {', '.join(parts)}." if parts else "**Enabled**, but no feature is selected."
    lines = [te_split.last_report(), denoise_split.last_report(), vae_split.last_report()]
    if ltx2_denoise is not None:
        lines += [ltx2_denoise.last_report(), ltx2_vae.last_report()]
    return state + "\n\n" + "\n".join(f"- {line}" for line in lines)


def _release_all():
    freed = [te_split.release(), vae_split.release(), denoise_split.release()]
    if ltx2_denoise is not None:
        freed += [ltx2_vae.release(), ltx2_denoise.release()]
    return any(freed)


class MultiGPUPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PLUGIN_NAME
        self.version = "0.5.4"
        self.description = ("Experimental multi-GPU support: MiniMax H3 (denoising, VAE decoding, text encoder) and LTX-2 "
                            "(denoising, VAE decoding) on two GPUs.")

    def setup_ui(self):
        modules = [("H3 VAE decoding", vae_split), ("H3 text encoder", te_split), ("H3 denoising", denoise_split)]
        if ltx2_denoise is not None:   # after the H3 denoising: both wrap attention_kit.qkv_attention (see ltx2_denoise.install)
            modules += [("LTX-2 denoising", ltx2_denoise), ("LTX-2 VAE decoding", ltx2_vae)]
        for name, module in modules:
            try:
                module.install()
                print(f"[Multi-GPU] Installed: {name} can use cuda:0 and cuda:1 (Media Generator > Multi-GPU tab).")
            except Exception as exc:
                print(f"[Multi-GPU] Could not install the {name} hooks: {exc}")
                traceback.print_exc()
        try:
            from . import kernel_compat
            locks = kernel_compat.install_triton_locks()
            if locks:
                print(f"[Multi-GPU] Serialized for two GPUs: {', '.join(locks)} (see README, Known issues of WanGP).")
            if kernel_compat.install_wangp_fixes():
                print("[Multi-GPU] Installed a fix for WanGP's shared INT8 projections with WGPKernels on non-ConvRot "
                      "checkpoints (see README, Known issues of WanGP).")
        except Exception as exc:
            print(f"[Multi-GPU] Could not install the WanGP INT8 fix: {exc}")
        self.insert_after("misc_tab", self._build_tab)

    def _build_tab(self):
        cfg = settings.get()
        with gr.Tab(PLUGIN_NAME) as tab:
            gr.Markdown("Experimental. **Uses the second GPU for MiniMax H3 (denoising, VAE decoding, text encoding) and "
                        "LTX-2 (denoising, VAE decoding).** Settings apply to the next denoising step, prompt or decode, "
                        "and are saved automatically.")
            gr.Markdown(_gpu_summary())
            enabled = gr.Checkbox(label="Enable multi-GPU", value=cfg["enabled"],
                                  info="Master switch, off at every WanGP start. Each feature below also needs it.")
            chunk_heads = gr.Dropdown(choices=[("Automatic (4, fewer on long videos)", 0), ("1", 1), ("2", 2), ("4", 4), ("8", 8)],
                                      value=cfg["denoise_chunk_heads"], label="Attention heads per exchanged chunk (H3 and LTX-2)",
                                      info="Smaller chunks shorten the waits between the GPUs, larger ones mean fewer, "
                                           "bigger attention calls. Long videos are capped at 64 MB per chunk.")
            gr.Markdown("### MiniMax H3")
            with gr.Group():
                gr.Markdown("**Denoising** (the main part of a render)")
                denoise_enabled = gr.Checkbox(label="Split H3 denoising across cuda:0 and cuda:1",
                                              value=cfg["denoise_enabled"])
                denoise_share = gr.Slider(minimum=10, maximum=90, step=1, value=cfg["denoise_share_cuda0"],
                                          label="Denoising work share of cuda:0 (%)",
                                          info="Share of the video tokens denoised on cuda:0. Same rule as below; the console "
                                               "line of each denoising pass says which GPU waits for the other.")
                denoise_text = gr.Markdown(_split_summary(cfg["denoise_share_cuda0"]))
                denoise_images = gr.Checkbox(label="Also split still images (Text to Image)", value=cfg["denoise_images"],
                                             info="May be slower than one GPU: an image has too few tokens to make up for "
                                                  "the exchanges between the GPUs (2048x2048: 2.4 s per step on two GPUs, "
                                                  "2.1 s on one). Needs Enable multi-GPU and the denoising split.")
            with gr.Group():
                gr.Markdown("**VAE decoding** (the final step that turns latents into video frames)")
                vae_enabled = gr.Checkbox(label="Split H3 VAE decoding across cuda:0 and cuda:1", value=cfg["vae_enabled"])
                share = gr.Slider(minimum=0, maximum=100, step=1, value=cfg["share_cuda0"], label="Work share of cuda:0 (%)",
                                  info="cuda:1 receives the rest. Match it to the relative speed of your GPUs: 50 for two "
                                       "identical GPUs, about 56 for an RTX 5080 (cuda:0) with an RTX 5070 Ti (cuda:1).")
                split_text = gr.Markdown(_split_summary(cfg["share_cuda0"]))
            with gr.Group():
                gr.Markdown("**Text encoder** (Qwen3-VL, 50 layers, run for each new prompt)")
                te_enabled = gr.Checkbox(label="Keep the last text encoder layers on cuda:1 (ignored while denoising is split)",
                                         value=cfg["te_enabled"])
                te_layers = gr.Slider(minimum=0, maximum=47, step=1, value=cfg["te_layers_cuda1"],
                                      label="Text encoder layers kept on cuda:1",
                                      info="These layers stay in cuda:1 VRAM, so each new prompt streams fewer layers from "
                                           "RAM. More layers: faster prompts, less free VRAM on cuda:1. If they do not fit "
                                           "(5 GB are kept free for the VAE copy), fewer are used.")
                te_text = gr.Markdown(_te_summary(cfg["te_layers_cuda1"]))
            ltx = ltx2_denoise is not None
            gr.Markdown("### LTX-2" + ("" if ltx else " (unavailable with this WanGP version: see the console)"))
            with gr.Group(visible=ltx):
                ltx_denoise_enabled = gr.Checkbox(label="Split LTX-2 denoising across cuda:0 and cuda:1",
                                                  value=cfg["ltx_denoise_enabled"])
                ltx_denoise_share = gr.Slider(minimum=10, maximum=90, step=1, value=cfg["ltx_denoise_share_cuda0"],
                                              label="LTX-2 denoising work share of cuda:0 (%)",
                                              info="Share of the video tokens denoised on cuda:0 (the audio tokens are "
                                                   "computed by both GPUs). The console line of each pass suggests a share.")
                ltx_denoise_text = gr.Markdown(_split_summary(cfg["ltx_denoise_share_cuda0"]))
                ltx_vae_enabled = gr.Checkbox(label="Split LTX-2 VAE decoding across cuda:0 and cuda:1",
                                              value=cfg["ltx_vae_enabled"],
                                              info="Shares the spatial tiles of the decoder. A frame decoded as a single "
                                                   "tile (often with the default VAE tiling) stays on cuda:0.")
                ltx_vae_share = gr.Slider(minimum=0, maximum=100, step=1, value=cfg["ltx_vae_share_cuda0"],
                                          label="LTX-2 VAE work share of cuda:0 (%)")
                ltx_vae_text = gr.Markdown(_split_summary(cfg["ltx_vae_share_cuda0"]))
            status = gr.Markdown(_status(cfg))
            with gr.Row():
                refresh_btn = gr.Button("Refresh status", size="sm")
                free_btn = gr.Button("Free secondary GPU memory", size="sm")

        def on_enabled(value):
            denoise_split.reset_broken()
            if ltx2_denoise is not None:
                ltx2_denoise.reset_broken()
            cfg_now = settings.update(enabled=bool(value))
            if not cfg_now["enabled"]:
                _release_all()
            return _status(cfg_now)

        def on_denoise_enabled(value):
            denoise_split.reset_broken()
            cfg_now = settings.update(denoise_enabled=bool(value))
            return _status(cfg_now)

        def on_denoise_images(value):
            cfg_now = settings.update(denoise_images=bool(value))
            return _status(cfg_now)

        def on_denoise_share(value):
            cfg_now = settings.update(denoise_share_cuda0=int(value))
            return _split_summary(cfg_now["denoise_share_cuda0"])

        def on_chunk_heads(value):
            cfg_now = settings.update(denoise_chunk_heads=int(value))
            return _status(cfg_now)

        def on_vae_enabled(value):
            cfg_now = settings.update(vae_enabled=bool(value))
            if not cfg_now["vae_enabled"]:
                vae_split.release()
            return _status(cfg_now)

        def on_share(value):
            cfg_now = settings.update(share_cuda0=int(value))
            return _split_summary(cfg_now["share_cuda0"])

        def on_te_enabled(value):
            cfg_now = settings.update(te_enabled=bool(value))
            if not cfg_now["te_enabled"]:
                te_split.release()
            return _status(cfg_now)

        def on_te_layers(value):
            cfg_now = settings.update(te_layers_cuda1=int(value))
            if cfg_now["te_layers_cuda1"] <= 0:
                te_split.release()
            return _te_summary(cfg_now["te_layers_cuda1"]), _status(cfg_now)

        def on_ltx_denoise_enabled(value):
            ltx2_denoise.reset_broken()
            cfg_now = settings.update(ltx_denoise_enabled=bool(value))
            if not cfg_now["ltx_denoise_enabled"]:
                ltx2_denoise.release()
            return _status(cfg_now)

        def on_ltx_denoise_share(value):
            cfg_now = settings.update(ltx_denoise_share_cuda0=int(value))
            return _split_summary(cfg_now["ltx_denoise_share_cuda0"])

        def on_ltx_vae_enabled(value):
            cfg_now = settings.update(ltx_vae_enabled=bool(value))
            if not cfg_now["ltx_vae_enabled"]:
                ltx2_vae.release()
            return _status(cfg_now)

        def on_ltx_vae_share(value):
            cfg_now = settings.update(ltx_vae_share_cuda0=int(value))
            return _split_summary(cfg_now["ltx_vae_share_cuda0"])

        def on_free():
            message = "Secondary GPU memory freed." if _release_all() else "Nothing to free."
            return f"{message}\n\n{_status()}"

        enabled.change(fn=on_enabled, inputs=[enabled], outputs=[status], show_progress="hidden")
        denoise_enabled.change(fn=on_denoise_enabled, inputs=[denoise_enabled], outputs=[status], show_progress="hidden")
        denoise_images.change(fn=on_denoise_images, inputs=[denoise_images], outputs=[status], show_progress="hidden")
        denoise_share.release(fn=on_denoise_share, inputs=[denoise_share], outputs=[denoise_text], show_progress="hidden")
        chunk_heads.change(fn=on_chunk_heads, inputs=[chunk_heads], outputs=[status], show_progress="hidden")
        vae_enabled.change(fn=on_vae_enabled, inputs=[vae_enabled], outputs=[status], show_progress="hidden")
        share.release(fn=on_share, inputs=[share], outputs=[split_text], show_progress="hidden")
        te_enabled.change(fn=on_te_enabled, inputs=[te_enabled], outputs=[status], show_progress="hidden")
        te_layers.release(fn=on_te_layers, inputs=[te_layers], outputs=[te_text, status], show_progress="hidden")
        if ltx:
            ltx_denoise_enabled.change(fn=on_ltx_denoise_enabled, inputs=[ltx_denoise_enabled], outputs=[status],
                                       show_progress="hidden")
            ltx_denoise_share.release(fn=on_ltx_denoise_share, inputs=[ltx_denoise_share], outputs=[ltx_denoise_text],
                                      show_progress="hidden")
            ltx_vae_enabled.change(fn=on_ltx_vae_enabled, inputs=[ltx_vae_enabled], outputs=[status], show_progress="hidden")
            ltx_vae_share.release(fn=on_ltx_vae_share, inputs=[ltx_vae_share], outputs=[ltx_vae_text], show_progress="hidden")
        refresh_btn.click(fn=lambda: _status(), inputs=None, outputs=[status], show_progress="hidden")
        free_btn.click(fn=on_free, inputs=None, outputs=[status], show_progress="hidden")
        return tab
