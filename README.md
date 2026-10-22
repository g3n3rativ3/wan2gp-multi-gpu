# Multi-GPU for WanGP (experimental)

Use two NVIDIA GPUs in **the same render** in [WanGP](https://github.com/deepbeepmeep/Wan2GP), with a manual work split
so that GPUs of different speeds (same generation) can be paired, and without giving up WanGP's RAM offloading.

> **Status: v0.5.3, experimental.** The plugin uses the second GPU for:
> - **MiniMax H3**: **denoising** (both GPUs work on the same step, each on part of the tokens), **VAE decoding** and the
>   **text encoder** (its last layers kept and run on `cuda:1`);
> - **LTX-2** (2.3 Distilled 1.0 and 1.1, 2.5 Distilled, and their Dev versions; new in v0.5): **denoising** (each GPU on part of the video tokens) and **VAE
>   decoding** (spatial tiles).
>
> Contributions are welcome.

## What it does (MiniMax H3)

### Denoising (since v0.3, experimental)

H3 is a single-stream transformer: each of its 50 blocks processes the whole token sequence (text, audio and video
tokens), and only attention mixes tokens. The plugin cuts the sequence in two: the first part of the tokens stays on
`cuda:0`, the rest is denoised on `cuda:1`. Both GPUs run every block on their own tokens at the same time. In each
attention, they swap their keys and values through pinned RAM (GeForce cards have no direct GPU-to-GPU link), chunk by
chunk while the previous chunk is being attended, so that every token still attends to every other token.

- Accuracy (measured with the accuracy check of v0.3.2, since removed), relative error of the first step's video output
  at 720p: one GPU vs one GPU with only the attention call path changed (Sage quantization noise alone): 2.2e-2; one GPU
  vs two GPUs: 2.8e-2. The split stays at the noise level of WanGP's own attention kernels (a bug would show 1e-1 or
  more). The text encoder split agrees to 1.8e-3. Few-step distilled models amplify such differences, so the video
  drifts like a slightly different seed (same composition, different details): different, not worse.
- Same computation: nothing is approximated or skipped, so it works with any number of steps, distilled models and turbo LoRAs
  included. The keys are put back in their original order and the split point is aligned on 128 tokens, so the
  attention kernels see the same data as with one GPU. Expect the same video, up to tiny floating point differences.
- Memory: `cuda:0` keeps WanGP's memory manager (MMGP) exactly as before. `cuda:1` has no copy of the model: a loader
  thread streams each block's weights (and active LoRAs) from the same Reserved RAM that MMGP uses for `cuda:0`, two
  blocks ahead of the compute. `cuda:1` needs about 2 to 4 GB of free VRAM; if the text encoder layers kept there leave
  too little, they are freed automatically.
- Work share: a separate slider, like the VAE one. After each denoising pass the console line suggests a share computed
  from each GPU's own work time (or says "balanced share").
- First measurement (v0.3.0, RTX 5080 `cuda:0` + RTX 5070 Ti `cuda:1`, share 56 %, H3 FL2VA Pruned 20B INT8 + turbo
  LoRA, 1280x704, 124 frames, 33022 tokens): **16.6 s -> 11.4 to 11.7 s per denoising step (-30 %)**, with the
  waits of the two GPUs balanced (about 1.2 s each per step, the time spent waiting for keys and values). The physical
  ceiling of this pair is about 9.3 s per step. v0.3.1 sends smaller chunks to shorten these waits.
- Real use (v0.3.3, same GPUs, H3 Ref2VA Pruned INT8 with 10 reference images, 15 s at 832x1088, two phases with the
  "2 Phases Plus" plugin): phase 1 (37703 tokens) 20.4 -> 13.2 s per step (-35 %), phase 2 (116634 tokens) 97.2 -> 61.8 s
  per step (-36 %), VAE decoding 50.4 -> 40.1 s, **whole render 345 s -> 235 s (-32 %)**. Peak VRAM on `cuda:1`: 2.5 GB
  and 5.2 GB.
- Two identical GPUs (v0.4.5, two RTX 5080, H3 Ref2VA Pruned INT8 with 5 RefMods, 15 s at 1088x832, two phases, 50/50
  share): phase 1 (35182 tokens) 12.0 -> 7.5 s per step (-37 %), phase 2 (106123 tokens) 70.7 -> 37.5 s per step
  (-47 %), VAE decoding 23.7 -> 18.4 s, **whole render 216.5 s -> 128.9 s (-40 %)**.
- Long sequence (v0.3.5, same setup, 20 s without sliding window, 481 frames): phase 1 (45836 tokens) 16.4 s per step,
  phase 2 (147972 tokens) 81.9 s per step with 5.6 GB peak on `cuda:1`; smaller exchange chunks cut the waits of
  phase 2 from 3.7 / 6.5 s to 0.7 / 1.4 s per step (87.1 -> 81.9 s).
- Safety net: if anything fails on the multi-GPU path (not enough VRAM, an unsupported option, an error on `cuda:1`),
  the denoising pass is computed again on one GPU, the render goes on, and the feature stays off until you switch it off
  and on again. The tab shows the reason.
- ControlNet-Union models are split too: WanGP's FL2VA ControlNet-Union (33B and Pruned 20B) and the Ref2VA
  ControlNet-Union models added by the `wan2gp-minimax-h3-ref2va-controlnet` plugin. Their control blocks run on the same
  rows split as the main blocks (each GPU computes its rows of the control stream, with the same key/value exchange), and
  each GPU adds the control skips to its own rows. The console line of each pass then ends with "ControlNet: N control
  blocks split too". Without a control video the control blocks stay idle, as on one GPU.
- A pass runs on one GPU when it uses an option this version does not split: Sol (sparse) or VEDA attention, VDN,
  Spectrum or first block caching, CUDA graphs, or a sequence shorter than 2048 tokens (images smaller than
  about 720x720). The console says so once.
- Still images (H3's Text to Image mode) stay on one GPU unless **Also split still images** is ticked: an image has too
  few tokens (about 4100 at 2048x2048) to make up for the exchanges between the GPUs. Measured on an RTX 5080 + RTX 5070
  Ti at 2048x2048, 20 steps: 2.06 s per step on one GPU, 2.4 s on two. The VAE decoding of images is still split
  (9.0 s down to 6 to 7.5 s at 2048x2048).

### VAE decoding (since v0.1)

MiniMax H3 decodes each temporal clip of the video as a batch of independent spatial tiles. With the plugin enabled, that
batch is cut in two: one part is decoded on `cuda:0` (where WanGP runs), the other on `cuda:1`, at the same time. The tiles
are then stitched exactly as WanGP does with one GPU.

- Output: the same video as with one GPU, up to tiny floating point rounding differences between two GPU models.
- Measured gain (RTX 5080 `cuda:0` + RTX 5070 Ti `cuda:1`, share 56 %, H3 FL2VA Pruned 20B INT8, 1280x704, 124 frames,
  4 steps, Windows 11): **VAE decoding 19.7 s -> 13.4 s (-32 %)**, H3 pipeline 86.8 s -> 80.9 s (-7 %). The output video
  is identical. The part of the decode that stays on `cuda:0` (projecting and stitching the tiles, writing frames to RAM)
  limits the gain; see Roadmap.
- Cost: a copy of the VAE decoder (about 2 GB) is kept on `cuda:1` after the first decode. It is built once (3.3 s in the
  test above), from the same checkpoint file, and freed when you disable the plugin or press **Free secondary GPU memory**.

### Text encoder (since v0.2)

H3's text encoder is Qwen3-VL-32B cut after its 50th language layer: about 23 GB in INT8, 465 MB per layer. With
WanGP's usual memory profiles it is not kept in VRAM, so every **new** prompt streams all 50 layers from RAM to the GPU.
The prompt is only a few hundred tokens, so the GPU spends that time waiting for weights, not computing (2.4 s per new
prompt in the test setup below, 4.6 s for the first prompt after loading the model, which also reads the file from disk;
an identical prompt is cached by WanGP and costs nothing).

The second GPU is idle at that moment and has free VRAM, so the plugin keeps the **last N layers** of the text encoder
resident on `cuda:1`. A new prompt then streams only the first 50 - N layers to `cuda:0`, as WanGP normally does; the
hidden state (a few MB) is handed to `cuda:1` through pinned RAM, `cuda:1` runs its N layers straight from VRAM, and the
result comes back to `cuda:0`.

- Output: the same embeddings as with one GPU (same weights, same checkpoint), up to floating point rounding
  differences between two GPU models.
- Measured gain (RTX 5080 + RTX 5070 Ti, 16 layers on `cuda:1`, the most that fitted next to the VAE copy): **2.4 s ->
  1.7 to 2.0 s per new prompt** (-17 to -29 %). The streamed volume drops by N/50; the 16 layers take 0.13 to 0.40 s on
  `cuda:1`, transfers included.
- Cost: about 0.47 GB of `cuda:1` VRAM per layer with the INT8 text encoder (0.93 GB with BF16). The layers are loaded
  once from the checkpoint file at the first new prompt after enabling the feature (expect several seconds), and stay on
  `cuda:1` until you disable the feature, change N, or press **Free secondary GPU memory**.
- If `cuda:1` does not have room for N layers while keeping 5 GB free for the VAE decoder copy and its work (1.5 GB when
  the VAE split is off), the plugin uses as many of the last layers as fit and says so in the console.
- The first 3 layers (which receive Qwen3-VL's DeepStack image features) always stay on `cuda:0`, so N is at most 47.

## What it does (LTX-2, since v0.5)

Ported from the "Multi-GPU LTX-2" plugin (v0.2.1, written for WanGP 11.22) and adapted to WanGP v18.17+ (mmgp 4.x).
That plugin measured, on two RTX 5080 with WanGP 11.22 and LTX-2.3 Distilled INT8, a 10 s 1080p render going from
2 min 37 s on one GPU to 1 min 45 s on two (-33 %; stage 2 of the distilled pipeline -40 %).

Models (WanGP v18.21): **LTX-2.3 Distilled 1.0** and **1.1** (and Dev), **LTX-2.5 Distilled** (and Dev). They share
WanGP's LTX-2 transformer and video decoder code; what differs is handled as follows:

| Difference | Effect on the split |
| --- | --- |
| Checkpoint formats: quanto INT8 (2.3 1.0, `quanto_int8`; 2.3 1.1, `quanto_bf16_int8`), **INT8 ConvRot** (2.5, `int8_convrot`), BF16, NVFP4 (2.5) | the block copies of `cuda:1` take the weights in their own format (ConvRot: the rotation group of each layer comes with the copy) |
| LTX-2.5: feed-forward without bias, learned **keyframe embedding** (start/end images, injected frames) | the embedding is added before the blocks, on the full sequence: nothing to split |
| LTX-2.5: ancestral sampler of the distilled pipeline, Gemma 4 text encoder | outside the blocks (the text encoder is not split for LTX-2) |
| LTX-2.3 1.1: HDR IC-LoRA added automatically | IC-LoRA reference tokens are video tokens like the others |

Tested on the CPU against WanGP v18.21's code (INT8 ConvRot loaded through mmgp's quanto router, as WanGP loads it).
NVFP4 checkpoints are not covered by the tests.

Measured with this version on two RTX 5080 (PCIe 5.0 x8 each, Windows), WanGP v18.21, LTX-2.3 Distilled quanto INT8,
WGPKernels, 121 frames: **1 min 34 s → 1 min 18 s (-18 %)**.

| Stage | Video tokens | One GPU | Two GPUs |
| --- | --- | --- | --- |
| Stage 1 (8 steps, low resolution) | 15 004 | 2.89 s/step | 2.9 to 3.0 s/step (no gain) |
| Stage 2 (3 steps, full resolution) | 60 016 | 17.5 s/step | 11.0 s/step (-37 %) |
| Rest (text encoder, VAE single tile, audio) | | about 20 s | about 20 s |

Why stage 1 gains nothing: the INT8 model (18.5 GB) does not fit in 16 GB, so each GPU streams all the weights from RAM
at every step (about 18 GB), and the video self-attention exchange adds about 12 GB per GPU and step (6 GB each way).
Windows gives each GeForce a single copy engine (28.7 GB/s on PCIe 5.0 x8, sending and receiving share it), so about
30 GB per step and GPU take at least 1.05 s, against about 1.45 s of compute each GPU has left with half the tokens. The
exchange can only hide behind the attention of the same block, which is short at low resolution: each GPU waits about
1.1 s per step. At full resolution the attention costs 16 times more and hides the exchange.

### Denoising

LTX-2 is a **dual-stream** transformer: each of its 48 blocks updates a **video stream** (thousands of tokens) and an
**audio stream** (a few hundred tokens). The plugin cuts the **video** sequence in two: the first rows stay on `cuda:0`,
the rest are denoised on `cuda:1`.

- Everything that works token by token (projections, norms, AdaLN modulation, feed-forward, text cross-attention,
  audio-to-video cross-attention) runs on each GPU for its own rows only. The **audio stream is small, so each GPU
  computes all of it**.
- Only two attentions need keys/values from *all* the video rows:
  - the **video self-attention**: the two GPUs swap their keys and values through pinned RAM, chunk by chunk, as for H3;
  - the **video-to-audio cross-attention**: its queries are the audio tokens, which both GPUs hold. Each GPU attends them
    over its own video rows and keeps the log-sum-exp of the scores; the GPUs swap these partial results (a few MB
    instead of hundreds of MB of keys/values) and merge them exactly (in float32).
- **Rotary embeddings** (LTX-2's `RopeCache`) and **per-frame timesteps** are cut exactly at the split point; when the
  timesteps are given per latent frame (the usual case), the split falls on a frame boundary.
- Same computation: any number of steps, distilled or dev models, LoRAs (IC-LoRAs included), CFG "joint" passes and STG
  perturbations. Expect the same video up to tiny floating point differences.
- Memory: `cuda:0` keeps WanGP's memory manager; `cuda:1` streams each block's weights from the same RAM, two blocks
  ahead, and keeps the copies of the blocks between the steps of a generation (dropped when it ends, when the VAE decode
  starts, after an interruption or a failure). `cuda:1` needs about 2 to 5 GB of free VRAM; the plugin checks it before
  each pass.
- WanGP v18 attention: LTX-2 now reaches the attention through `attention_kit` (self-attentions, and cross-attentions
  with **Attention Head Split**) or by calling `pay_attention` itself (other cross-attentions, a backend forced by the
  model). Both paths are intercepted; a split attention always uses all its heads at once (both GPUs must exchange the
  same chunks), so Attention Head Split does not apply to the two distributed attentions.
- A pass stays on one GPU (with a console message) with fewer than 2048 video tokens, `torch.compile`, CUDA graphs, a
  batch of several videos, or a self-attention or audio/video cross-attention mask (some audio-paired modes).
- Work share: its own slider. The console line of each pass suggests a share; about 48 % was the advice for two
  identical cards with WanGP 11.22 (`cuda:0` also runs WanGP's memory manager).

### VAE decoding

WanGP v18 rewrote LTX-2's video decoder: it now cuts the frame into **spatial tiles** (the fewest that fit the **VAE
tiling** setting) and decodes each tile **streamed latent frame after latent frame**, with no temporal tiles (every
frame is exactly the one the whole video would give). The time axis can therefore not be shared; the tiles are:

- tile 0 stays on `cuda:0` (so both GPUs start at once), the others are spread according to the VAE share. `cuda:1`
  decodes its tiles ahead of time with its own copy of the decoder and records, latent frame by latent frame, what its
  stream produces; WanGP's loop then feeds these frames to its own blending code in the same order. The video is
  identical to the one-GPU decode (checked bit for bit in the tests).
- **A frame decoded as a single tile stays on one GPU** — often the case with the default VAE tiling, which fits a
  1080p frame in one tile. A smaller VAE tiling size gives 2 tiles or more. With 2 tiles, the share must be 50 % or
  less for `cuda:1` to get one.
- A decoded tile waits in RAM until its turn (up to a few GB for a long 1080p video, one tile at a time).
- If `cuda:1` fails to decode a tile, that tile and the remaining ones are decoded on `cuda:0`: the video is still
  complete and identical.
- Cost: a copy of the decoder on `cuda:1` (built once from the tensors MMGP holds in RAM), freed by **Free secondary
  GPU memory**, by unticking the LTX-2 VAE split or by disabling the plugin.

### Text encoder

Not split for LTX-2 (Gemma 3 runs once per new prompt and is cached by WanGP).

## Requirements

- WanGP v18 or later, with a MiniMax H3 model (FL2VA, Ref2VA, pruned or full; INT8, FP8 or BF16 VAE). LTX-2 support
  needs WanGP v18.17 or later (tested against v18.17 and v18.21); with an older WanGP the LTX-2 section of the tab says it
  is unavailable and the H3 features work as before.
- Two NVIDIA GPUs of the same generation (tested: RTX 5080 + RTX 5070 Ti, and two RTX 5080, Blackwell).
- Windows or Linux. No NCCL, no `torch.distributed`, no peer-to-peer GPU access needed: everything runs in WanGP's process.
- VAE tiling on (the H3 default). With tiling off there is only one tile per clip and nothing to split.
- Any INT8 kernel backend of WanGP (PyTorch, Triton, Comfy Kitchen, WGPKernels). WGPKernels 1.1.2 is written for one GPU
  per process (its native library raises each GEMM kernel's shared memory limit once, but that limit is per GPU), so the
  plugin gives each secondary GPU its own copy of that library: a file named like `wgp_int8_mgpu_cuda1.dll` written next
  to the original (or in the temp folder), loaded only for the INT8 work of that GPU. WanGP's Triton INT8 kernels time
  their configurations inside a CUDA graph the first time a shape is seen; the plugin runs these tuning sessions one at
  a time and in CUDA's "thread_local" capture mode, so the other GPU can keep working meanwhile.

## Install

1. Copy the `wan2gp-multi-gpu` folder into `WanGP/plugins/`.
2. Make sure WanGP sees both GPUs: **do not start WanGP with `--gpu`**. That option sets `CUDA_VISIBLE_DEVICES` and hides
   the other GPUs.
3. Start WanGP, enable **Multi-GPU** in the plugin manager, restart WanGP.
4. The console should print `[Multi-GPU] Installed`.

## Use

In the Media Generator, switch to Advanced Mode and open the **Multi-GPU** tab (after **Misc.**):

- **Enable multi-GPU**: master on/off switch. Off means WanGP behaves exactly as without the plugin. It is always off
  when WanGP starts, so the second GPU is never used by accident; the other settings of the tab are remembered.
- **Attention heads per exchanged chunk** (H3 and LTX-2): automatic by default (4, fewer on long videos so that a chunk
  stays under 64 MB). Smaller chunks shorten the waits between the GPUs; larger ones mean fewer, bigger attention calls.

**MiniMax H3**

- **Denoising** (experimental)
  - **Split H3 denoising across cuda:0 and cuda:1**: off by default.
  - **Denoising work share of cuda:0 (%)**: share of the tokens denoised on `cuda:0`. Default 50; about 56 for an
    RTX 5080 (`cuda:0`) with an RTX 5070 Ti (`cuda:1`).
  - **Also split still images (Text to Image)**: off by default; needs **Enable multi-GPU** and the denoising split. May
  be slower than one GPU (see above).
- **VAE decoding**
  - **Split H3 VAE decoding across cuda:0 and cuda:1**: on by default (once the master switch is on).
  - **Work share of cuda:0 (%)**: share of the tiles decoded on `cuda:0`; `cuda:1` gets the rest. Default 50.
- **Text encoder**
  - **Keep the last text encoder layers on cuda:1**: off by default, and ignored while denoising is split.
  - **Text encoder layers kept on cuda:1**: N, from 0 to 47. Default 20 (about 9.4 GB with the INT8 text encoder).

**LTX-2**

- **Split LTX-2 denoising across cuda:0 and cuda:1**: on by default (once the master switch is on).
- **LTX-2 denoising work share of cuda:0 (%)**: share of the video tokens denoised on `cuda:0`. Default 50.
- **Split LTX-2 VAE decoding across cuda:0 and cuda:1**: on by default (once the master switch is on).
- **LTX-2 VAE work share of cuda:0 (%)**: share of the spatial tiles decoded on `cuda:0`. Default 50.

The tab lists the GPUs WanGP sees, with their `cuda:N` numbers, and the result of the last pass and decode of each
feature.

Settings are saved in `settings.json` next to the plugin and apply to the next prompt or decode, including queued jobs.

### Choosing the number of text encoder layers

More layers on `cuda:1` means faster new prompts and less free VRAM on `cuda:1`. The text encoder split is skipped while
the H3 denoising split is on (that VRAM goes to denoising, worth far more). Without it, 20 is a safe value on a 16 GB
card, and the plugin trims the number if it does not fit.

### Choosing the work share

Set the share to the relative speed of the two GPUs: `share = speed(cuda:0) / (speed(cuda:0) + speed(cuda:1))`.

| GPUs | Share of cuda:0 |
| --- | --- |
| Two identical GPUs | 50 |
| RTX 5080 (`cuda:0`) + RTX 5070 Ti (`cuda:1`), measured 410 vs 325 INT8 TOPS | 56 |

To fine-tune, enable the H3 Profiler plugin, render the same video with a few values and keep the fastest. By default,
CUDA numbers GPUs from fastest to slowest, so `cuda:0` is usually the faster one; the tab shows which is which.

## How it works (for contributors)

| File | Role |
| --- | --- |
| `plugin.py` | Gradio tab, settings events. |
| `settings.py` | Persistent settings (`settings.json`). |
| `common.py` | Shared helpers: GPU list, one worker thread per secondary GPU, copies through pinned RAM, tensor sizes. |
| `vae_split.py` | VAE split: patches `MiniMaxH3VideoViTDecoder3d.forward`, builds and keeps the decoder copy, runs the second part in a worker thread bound to the secondary GPU. |
| `te_split.py` | Text encoder split: loads the last N language layers from the checkpoint onto the secondary GPU and runs them there. |
| `denoise_split.py` | H3 denoising split: token rows per GPU, hook-free copies of the blocks for the secondary GPU, weight streaming, key/value exchange. |
| `kernel_compat.py` | WanGP's INT8 kernels on two GPUs: a copy of the WGPKernels library per secondary GPU, Triton tuning one session at a time. |
| `clones.py` | Hook-free copies of MMGP-managed modules (LTX-2 blocks and decoder), tensors pointing to MMGP's CPU tensors. |
| `ltx2_denoise.py` | LTX-2 denoising split: video rows per GPU, audio on both, RoPE/timestep slicing, key/value exchange, partial attention merge. |
| `ltx2_vae.py` | LTX-2 VAE split: spatial tiles of the streamed decoder, decoder copy, secondary decode thread. |
| `tests/` | CPU checks of the LTX-2 splits (no GPU needed): `python -I tests/test_ltx2_denoise.py <WanGP folder> <plugins folder>`, same for `test_ltx2_vae.py`. |

Hooks installed at WanGP startup, before any model is loaded:

1. `minimax_h3_main._load_video_vae` and `minimax_h3_main._load_text_encoder` are wrapped to remember their arguments on
   the loaded model, so the plugin can load the same checkpoint a second time for `cuda:1`.
2. `MiniMaxH3VideoViTDecoder3d.forward` is replaced: when enabled and the batch has at least 2 tiles, it decodes the first
   `round(batch x share)` tiles locally and the rest on the secondary GPU in parallel, then concatenates the results.
3. `MiniMaxH3VideoVAE.decode` is wrapped to report the split and the time of each decode.
4. `MiniMaxH3TextEncoder.encode` is wrapped: for the duration of the call, the layer list of the language model is
   replaced by layers `0..49-N` (unchanged, still managed by WanGP's memory manager), one stand-in module that runs layers
   `50-N..49` on the secondary GPU, and `N-1` pass-through modules so that the progress bar still counts 50 layers. The
   original layer list is put back as soon as the call returns. The copy of the last layers is held in a plain Python
   list, outside the model, so WanGP's memory manager never sees or moves it.

5. `MiniMaxH3Model.forward` is wrapped: it decides whether the pass can be split and, if a failure occurs, computes the
   pass again on one GPU.
6. `DiTBlock.forward` is replaced: at block 0 the hidden states are cut in two (the second part is sent to the secondary
   GPU, whose worker thread then runs all 50 blocks on it); every block then runs on the primary GPU with its own rows,
   segments and rotary embeddings; the last block gathers both parts back. WanGP's loop, MMGP's loading of `cuda:0` and
   everything before and after the blocks are unchanged.
7. `shared.attention_kit.qkv_attention` is wrapped: inside a split block it hands WanGP's normalized and rotated q, k, v
   to the plugin's attention, which exchanges keys and values and calls WanGP's `pay_attention` (your attention
   backend, Sage or SDPA) per chunk of 8 heads.
8. `torch.cuda.cudart().cudaHostRegister` registers pinned RAM as portable, so that the secondary GPU reads MMGP's
   Reserved RAM with fast asynchronous copies. It changes nothing for `cuda:0`.

The block copies for `cuda:1` are the same module objects rebuilt without MMGP's hooks: their tensors point to MMGP's
pinned CPU tensors, swapped for `cuda:1` copies while a block runs. LoRA linears keep MMGP's LoRA forward with
`cuda:1` copies of the adapter weights; scales and step schedules are read from the model, as on `cuda:0`.

LTX-2 hooks (installed after the H3 ones):

9. `LTXModel._process_transformer_blocks` and `_process_transformer_blocks_joint` are replaced: when the split applies,
   the blocks' arguments (`TransformerArgs`) are cut, `cuda:0` runs WanGP's real blocks on its rows in the same loop as
   WanGP, and a worker thread runs hook-free copies of the 48 blocks on the rows of `cuda:1`.
10. `pay_attention` as imported by LTX-2's attention module, and `attention_kit.qkv_attention` /
    `attention_kit.cross_qkv_attention` (above the H3 wrapper) are wrapped: inside a split pass, when the calling
    Attention module is a block's `attn1` or `video_to_audio_attn`, the attention goes to the plugin's distributed
    attention.
11. `VideoDecoder.tiled_decode` is wrapped: when the frame has 2 tiles or more, the decoder's `stream_step` is replaced
    for the duration of the decode so that the tiles of `cuda:1` are fed from its recorded output.
12. At the first split pass, `LTX2.generate` is wrapped so that the block copies kept between steps are dropped when a
    generation ends.

When the plugin is disabled, every hook calls the original WanGP code directly.

## Roadmap

1. **v0.1**: H3 VAE decoding on two GPUs. Possible improvements: project and stitch the remote tiles on `cuda:1`, and
   decode clip N+1 while clip N is being stitched.
2. **v0.2 (this release)**: H3 text encoder, last N layers resident on `cuda:1`. Possible improvement: let `cuda:1`
   stream the layers it does not hold from RAM in parallel with `cuda:0` (two PCIe links at once).
3. **v0.3 (this release)**: denoising split by token rows, standard attention path. Next: an automatic work share
   (from the waits measured at each pass), Sol attention and VDN support, and measurements on more GPU pairs.

## Background: feasibility measurements

Measured on Windows 11, RTX 5080 + RTX 5070 Ti, both PCIe 5.0 x8 from an Intel Core Ultra 7 270K, 96 GB DDR5-5600,
PyTorch 2.10 + CUDA 13.0.

| Measurement | Result |
| --- | --- |
| RAM to GPU or GPU to RAM, one GPU | 28.7 GB/s per direction (Gen5 x8) |
| Copy engines per GPU under Windows | 1: receiving and sending share the 28.7 GB/s |
| Both GPUs loading from RAM at once | 38 to 43 GB/s in total (mixed 32 GB + 64 GB RAM sticks, Intel Flex mode) |
| GPU to RAM to GPU chain, one direction | 24 to 26 GB/s, data intact |
| GPU writes to RAM with its own cores (Triton kernel, 4 blocks) | 26 GB/s; send + receive at once reach 35 to 40 GB/s |
| Exchange hidden behind compute (copy engines) | 98 to 99 % |
| INT8 compute | RTX 5080: 410 TOPS, RTX 5070 Ti: 325 TOPS (79 %) |

Estimated gains for a full H3 render (4 steps, new prompt), RTX 5080 + RTX 5070 Ti versus the RTX 5080 alone:
-21 % (480p) / -27 % (720p) with denoising only, -33 % / -37 % with denoising + VAE + text encoder. The physical ceiling of
this GPU pair is -44 % (the RTX 5070 Ti brings 79 % of the RTX 5080's compute). CFG parallelism brings nothing here:
distilled models run at guidance 1.

## Single-GPU tip: pinning the text encoder

Independently of this plugin, the text encoder is slow mainly because WanGP's default memory profile (Profile 4) keeps
only the main model in *Reserved RAM* (pinned memory, the fastest path to the GPU). **Profile 2** (Configuration > RAM/VRAM Management, video memory profile) keeps every model there,
the text encoder included, which should make new prompts several times faster on one GPU (not measured yet). It needs a
lot of RAM: about 20 GB for the H3 transformer plus 23 GB for the text encoder, and WanGP reserves at most 40 % of the
RAM on Windows by default, so raise **Reserved RAM for Pinning** (Configuration > RAM/VRAM Management; 50 to 60 % on a 96 GB PC) if the
text encoder does not fit. The two approaches add up:
with the text encoder pinned, the plugin still removes the transfer of the N layers held by `cuda:1`.

## Report your results

Please open an issue or a discussion with: your GPUs and PCIe links, Windows or Linux, the WanGP version, the share you
used, and the `[Multi-GPU]` and `[H3 Profiler]` console lines for the same render with the plugin on and off.

## Known limitations

- MiniMax H3 and LTX-2 only.
- Denoising: standard attention path only (see the lists of options that keep a pass on one GPU above).
- Two GPUs: `cuda:0` and `cuda:1`. More GPUs are not used.
- The first decode after enabling the plugin also builds the decoder copy on `cuda:1` (a few seconds, plus 2 to 3 GB of RAM
  while loading).
- The text encoder layers stay on `cuda:1` when you switch to another model; press **Free secondary GPU memory** to give
  that VRAM back. A different text encoder file is detected and the copy is rebuilt.
- Image and video references given to H3 are encoded by the Qwen3-VL vision tower on `cuda:0` as before; only the
  language layers are split.
- VAEs loaded by other code paths than the H3 model loader (for example some post-processing tools) are decoded on one GPU.

## Known issues of WanGP worked around by the plugin

- **WanGP v18.21, WGPKernels + a plain quanto INT8 checkpoint** (for example `ltx-2.3-22b-distilled_..._quanto_int8`):
  `AttributeError: 'QLinearQuantoRouter' object has no attribute '_convrot_group_size'` at the first LTX-2 step, with or
  without multi-GPU. WanGP's `int8_backend.linear_multi` (one input shared by several projections) reads that attribute
  without a default, which only ConvRot INT8 layers have. The plugin installs a guard at startup, whatever its switches:
  in that case only, each projection runs on its own, the path WanGP takes for projections it cannot share (same
  result). Without the plugin, choose another INT8 backend (Comfy Kitchen or Triton) for such checkpoints.

## Changelog

- **0.5.3**: LTX-2.3 Distilled 1.1 and LTX-2.5 Distilled (INT8 ConvRot checkpoint, keyframe embedding, feed-forward
  without bias) checked and documented; new LTX-2.5 cases in the tests. No change was needed in the split itself.
- **0.5.2**: an LTX-2 VAE decode that stays on one GPU (single spatial tile, no tile for `cuda:1`) now prints its
  reason in the console. First LTX-2 measurements on WanGP v18.21 in the README.
- **0.5.1**: guard for the WanGP v18.21 bug above (LTX-2.3 quanto INT8 with WGPKernels failed at the first step, plugin
  on or off).
- **0.5.0**: **LTX-2 support**, from the "Multi-GPU LTX-2" plugin (v0.2.1, WanGP 11.22), adapted to WanGP v18.17+:
  - denoising split (video rows per GPU, audio on both, exact RoPE and timestep slicing, key/value exchange for the
    video self-attention, partial attention merge for the video-to-audio cross-attention), now catching both attention
    paths of WanGP v18 (`attention_kit` and `pay_attention`), and staying on one GPU with CUDA graphs or audio/video
    cross-attention masks;
  - VAE split rewritten for WanGP v18's streamed decoder (spatial tiles instead of temporal ones);
  - its own section in the tab (denoising and VAE switches and shares); the attention chunk setting is shared.
  - CPU tests for LTX-2 in `tests/` (all checks pass on WanGP v18.17 and v18.21; the decode is bit-identical).
  H3: the CUDA graph check of the denoising split works again with WanGP v18.17+ (the function it used was removed, so a
  pass could have been split with CUDA graphs on); the plugin's RAM buffers name the CPU explicitly (WanGP may generate
  inside `with torch.device("cuda")`); console status lines say which model they are about.
- **0.4.6**: the console line of each denoising pass suggests a share ("suggested share of cuda:0: 54 %") computed from
  each GPU's own work time, instead of comparing their waits, which pointed the wrong way (a GPU also waits for the
  other's keys and values in the middle of every block). Measured with two RTX 5080: Ref2VA Pruned, 15 s at 1088x832, two
  phases, 216.5 s on one GPU, 128.9 s on two at 50/50 (-40 %; denoising -44 %, VAE -22 %).
- **0.4.5**: the 0.4.4 transfer of the Ref2VA ControlNet plugin's control timestep embedding read a tensor version
  counter, which inference tensors do not have ("Inference tensors do not track version counter"). Fixed; the test now
  runs in inference mode, as WanGP does.
- **0.4.4**: with the Ref2VA ControlNet-Union plugin, the denoising split failed at the first control block
  (`CUDA error: invalid argument`): the plugin's control timestep embedding lives on the primary GPU and was copied
  directly to the secondary one, which these cards cannot do. It now goes through RAM, once per pass. The failed copy
  also left a CUDA error pending in the secondary GPU's thread, which the VAE decoding then reported and stopped the
  render: every job on the secondary GPU now first clears any error left by a previous one.
- **0.4.3**: ControlNet-Union support for the denoising split (it ran on one GPU before): WanGP's FL2VA ControlNet-Union
  models and the Ref2VA ControlNet-Union models of the `wan2gp-minimax-h3-ref2va-controlnet` plugin. The control blocks
  are split like the main blocks and each GPU applies the control skips to its own rows; same output as one GPU in the
  tests (control at 30, 56 and 80 % shares, no control video, and the Ref2VA plugin's replaced control AdaLN). Layers
  called by a forward that another plugin replaced now take their input on the device of their own weights.
- **0.4.2**: with the Triton INT8 kernels, the VAE decoding failed on the second GPU with `CUDA error: the operation
  cannot be performed in the present state`: the first-time tuning of a Triton kernel on one GPU captured a CUDA graph in
  "global" mode, which forbids allocations on every other thread. Triton tuning sessions now run one at a time, in
  "thread_local" capture mode. Other CUDA graphs and the tuning results are unchanged.
- **0.4.1**: WGPKernels (WanGP's own INT8 kernels) failed on the second GPU with `INT8 GEMM launch failed (CUDA error 1)`,
  in the denoising and the VAE decoding: its native library sets the shared memory limit of each kernel once per
  process, for the first GPU only. The secondary GPU now loads its own copy of the library; the library's workspace size
  query and tuning cache are also made safe for two GPUs working at the same time. Nothing changes with the other INT8
  backends or on one GPU.
- **0.4.0**: still images measured slower on two GPUs (2048x2048: 2.4 s per step instead of 2.06 s), so their denoising
  stays on one GPU unless the new **Also split still images** box is ticked (off by default). Their VAE decoding is still
  split, which is faster.
- **0.3.9**: still images (Text to Image) can use both GPUs: the split now starts at 2048 tokens instead of 4096, so
  1024x1024 and 1280x720 images are split; smaller ones stay on one GPU and the console says why.
- **0.3.8**: **Enable multi-GPU** is always off when WanGP starts (the other settings are still remembered). The model
  switch failure of 0.3.7 turned out to come from another plugin that kept the previous H3 pipeline alive; WanGP's
  `--ram-debug` report shows which object holds a released model, if it happens again. Real-use measurement with 1 head
  per chunk: Ref2VA 15 s, two phases, 244 to 247 s (about 240 s with the automatic setting, 345 to 349 s on one GPU).
- **0.3.7**: switching to another H3 model in the same WanGP session failed while loading it ("Reserved RAM is short",
  `cudaHostRegister failed`, then `CUDA error: out of memory`): the block copies kept between passes still pointed to the
  previous model's pinned RAM, so it could not be freed. The copies now live for one pass only (they take no time to
  build). Chunk size test at 720p: 1, 2 and 4 heads per chunk all give 11.9 to 12.0 s per step; the automatic setting stays.
- **0.3.6**: **Attention heads per exchanged chunk** setting (automatic, 1, 2, 4 or 8), to compare chunk sizes; the
  console line of each denoising pass shows the size used.
- **0.3.5**: long sequences (148k tokens: 20 s, 832x1088, no sliding window) crashed at the end of a step with
  `CUDA error: out of memory` on `cuda:1`. The key/value exchange now caps each chunk at 64 MB (fewer heads per chunk on
  long sequences) and keeps 6 pinned slots per GPU instead of 19: at 148k tokens, pinned RAM for the exchange drops from
  about 6.4 GB to 0.5 GB, and the full-sequence buffers on each GPU from 0.6 GB to 0.15 GB. The hidden states handed over
  once per step use pageable RAM, and `cuda:1` gives its cached VRAM back after each step.
- **0.3.4**: the text encoder split is off by default and skipped while denoising is split: its gain is a fraction of a
  second, and its 8 GB on `cuda:1` kept long renders (116k tokens) from splitting denoising. Closer VRAM estimate for
  denoising (measured peak about 6x the size of the rows of `cuda:1`). Real-use measurement added (Ref2VA, two phases).
- **0.3.3**: the accuracy check is removed (its job is done, see Accuracy above). No other change.
- **0.3.2**: **Check accuracy on the next render** button: the next new prompt is encoded on one GPU and on two, and the
  first denoising step is computed three times (one GPU as WanGP does it, one GPU with the attention path of the split,
  two GPUs); the console prints the relative error of each pair. First 720p measurement: denoising 66.6 s -> 45.8 s,
  render 86.8 s -> 59.8 s (-31 %).
- **0.3.1**: free VRAM on `cuda:1` counts the memory cached by the allocator: v0.3.0 saw too little after two steps,
  freed the text encoder layers for nothing, then switched denoising off. A pass without room on `cuda:1` now just runs
  on one GPU and the feature stays on. Chunks of 4 heads with 3 sent ahead (was 8 and 2). The console line reports the
  peak VRAM used on `cuda:1`. The balance hint ignores small differences.
- **0.3.0**: denoising on two GPUs (experimental, off by default): sequence parallelism with a key/value exchange
  through pinned RAM, weights and LoRAs streamed to `cuda:1` from MMGP's Reserved RAM, automatic fallback to one GPU.
  Separate work share slider. The text encoder keeps more VRAM free on `cuda:1` when denoising is split.
- **0.2.0**: text encoder on two GPUs: the last N layers of the H3 text encoder (Qwen3-VL) are kept on `cuda:1`. Separate
  switches for the VAE and text encoder features. Shared helpers moved to `common.py`.
- **0.1.1**: GPU-to-GPU transfers go through pinned RAM. A direct `cuda:0` to `cuda:1` copy failed with `CUDA error: invalid argument` on Windows (no peer-to-peer access on GeForce cards, and WanGP's VMM allocator memory is private to its GPU).
- **0.1.0**: first release: H3 VAE decoding split across two GPUs.

## License

To be chosen by the author before publishing (for example MIT). WanGP itself is under the WanGP Community License.
Using MiniMax H3 is subject to its own license, which restricts the territories where it may be used.
