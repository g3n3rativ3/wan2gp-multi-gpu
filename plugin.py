"""Multi-GPU plugin for WanGP.

v0.3: MiniMax H3 denoising split across cuda:0 and cuda:1 by token rows (denoise_split.py, experimental), VAE decoding
split by tiles (vae_split.py), and the last layers of the text encoder kept on cuda:1 (te_split.py). See README.md.
The settings live in a "Multi-GPU" tab added after the "Misc." tab of the Media Generator advanced settings.
"""
import traceback

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from . import denoise_split, settings, te_split, vae_split
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
        parts = [name for name, on in (("H3 denoising", cfg["denoise_enabled"]),
                                       ("H3 VAE decoding", cfg["vae_enabled"]),
                                       ("H3 text encoder", cfg["te_enabled"] and cfg["te_layers_cuda1"] > 0)) if on]
        state = f"**Enabled** for: {', '.join(parts)}." if parts else "**Enabled**, but no feature is selected."
    return (f"{state}\n\n- {te_split.last_report()}\n- {denoise_split.last_report()}\n- {vae_split.last_report()}")


def _release_all():
    freed_te = te_split.release()
    freed_vae = vae_split.release()
    freed_denoise = denoise_split.release()
    return freed_te or freed_vae or freed_denoise


class MultiGPUPlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PLUGIN_NAME
        self.version = "0.3.5"
        self.description = "Experimental multi-GPU support (v0.3: MiniMax H3 denoising, VAE decoding and text encoder on two GPUs)."

    def setup_ui(self):
        for name, module in (("H3 VAE decoding", vae_split), ("H3 text encoder", te_split), ("H3 denoising", denoise_split)):
            try:
                module.install()
                print(f"[Multi-GPU] Installed: {name} can use cuda:0 and cuda:1 (Media Generator > Multi-GPU tab).")
            except Exception as exc:
                print(f"[Multi-GPU] Could not install the {name} hooks: {exc}")
                traceback.print_exc()
        self.insert_after("misc_tab", self._build_tab)

    def _build_tab(self):
        cfg = settings.get()
        with gr.Tab(PLUGIN_NAME) as tab:
            gr.Markdown("Experimental. **v0.3 uses the second GPU for MiniMax H3 denoising (experimental), VAE decoding "
                        "and text encoding.** Settings apply to the next denoising step, prompt or decode, and are saved "
                        "automatically.")
            gr.Markdown(_gpu_summary())
            enabled = gr.Checkbox(label="Enable multi-GPU", value=cfg["enabled"])
            with gr.Group():
                gr.Markdown("**Denoising** (the main part of a render) - experimental")
                denoise_enabled = gr.Checkbox(label="Split H3 denoising across cuda:0 and cuda:1",
                                              value=cfg["denoise_enabled"])
                denoise_share = gr.Slider(minimum=10, maximum=90, step=1, value=cfg["denoise_share_cuda0"],
                                          label="Denoising work share of cuda:0 (%)",
                                          info="Share of the video tokens denoised on cuda:0. Same rule as below; the console "
                                               "line of each denoising pass says which GPU waits for the other.")
                denoise_text = gr.Markdown(_split_summary(cfg["denoise_share_cuda0"]))
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
            status = gr.Markdown(_status(cfg))
            with gr.Row():
                refresh_btn = gr.Button("Refresh status", size="sm")
                free_btn = gr.Button("Free secondary GPU memory", size="sm")

        def on_enabled(value):
            denoise_split.reset_broken()
            cfg_now = settings.update(enabled=bool(value))
            if not cfg_now["enabled"]:
                _release_all()
            return _status(cfg_now)

        def on_denoise_enabled(value):
            denoise_split.reset_broken()
            cfg_now = settings.update(denoise_enabled=bool(value))
            return _status(cfg_now)

        def on_denoise_share(value):
            cfg_now = settings.update(denoise_share_cuda0=int(value))
            return _split_summary(cfg_now["denoise_share_cuda0"])

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

        def on_free():
            message = "Secondary GPU memory freed." if _release_all() else "Nothing to free."
            return f"{message}\n\n{_status()}"

        enabled.change(fn=on_enabled, inputs=[enabled], outputs=[status], show_progress="hidden")
        denoise_enabled.change(fn=on_denoise_enabled, inputs=[denoise_enabled], outputs=[status], show_progress="hidden")
        denoise_share.release(fn=on_denoise_share, inputs=[denoise_share], outputs=[denoise_text], show_progress="hidden")
        vae_enabled.change(fn=on_vae_enabled, inputs=[vae_enabled], outputs=[status], show_progress="hidden")
        share.release(fn=on_share, inputs=[share], outputs=[split_text], show_progress="hidden")
        te_enabled.change(fn=on_te_enabled, inputs=[te_enabled], outputs=[status], show_progress="hidden")
        te_layers.release(fn=on_te_layers, inputs=[te_layers], outputs=[te_text, status], show_progress="hidden")
        refresh_btn.click(fn=lambda: _status(), inputs=None, outputs=[status], show_progress="hidden")
        free_btn.click(fn=on_free, inputs=None, outputs=[status], show_progress="hidden")
        return tab
