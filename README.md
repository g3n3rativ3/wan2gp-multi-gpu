# Multi-GPU for WanGP (experimental)

Use two NVIDIA GPUs in **the same render** in [WanGP](https://github.com/deepbeepmeep/Wan2GP), with a manual work split
so that GPUs of different speeds (same generation) can be paired, and without giving up WanGP's RAM offloading.

> **Status: v0.3.5, experimental.** The plugin uses the second GPU for the three stages of a **MiniMax H3** render:
> **denoising** (new in v0.3, experimental, off by default: both GPUs work on the same step, each on part of the tokens),
> **VAE decoding** (split across `cuda:0` and `cuda:1`) and the **text encoder** (its last layers kept and run on
> `cuda:1`). Contributions are welcome.

## What it does

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
- Work share: a separate slider, like the VAE one. After each denoising pass the console says which GPU waited for the
  other; raise the share of `cuda:0` when `cuda:0` waits, lower it when `cuda:1` waits.
- First measurement (v0.3.0, RTX 5080 `cuda:0` + RTX 5070 Ti `cuda:1`, share 56 %, H3 FL2VA Pruned 20B INT8 + turbo
  LoRA, 1280x704, 124 frames, 33022 tokens): **16.6 s -> 11.4 to 11.7 s per denoising step (-30 %)**, with the
  waits of the two GPUs balanced (about 1.2 s each per step, the time spent waiting for keys and values). The physical
  ceiling of this pair is about 9.3 s per step. v0.3.1 sends smaller chunks to shorten these waits.
- Real use (v0.3.3, same GPUs, H3 Ref2VA Pruned INT8 with 10 reference images, 15 s at 832x1088, two phases with the
  "2 Phases Plus" plugin): phase 1 (37703 tokens) 20.4 -> 13.2 s per step (-35 %), phase 2 (116634 tokens) 97.2 -> 61.8 s
  per step (-36 %), VAE decoding 50.4 -> 40.1 s, **whole render 345 s -> 235 s (-32 %)**. Peak VRAM on `cuda:1`: 2.5 GB
  and 5.2 GB.
- Safety net: if anything fails on the multi-GPU path (not enough VRAM, an unsupported option, an error on `cuda:1`),
  the denoising pass is computed again on one GPU, the render goes on, and the feature stays off until you switch it off
  and on again. The tab shows the reason.
- A pass runs on one GPU when it uses an option this version does not split: Sol (sparse) or VEDA attention, VDN,
  ControlNet, Spectrum or first block caching, CUDA graphs, or a sequence shorter than 4096 tokens (images). The console
  says so once.

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

## Requirements

- WanGP v18 or later, with a MiniMax H3 model (FL2VA, Ref2VA, pruned or full; INT8, FP8 or BF16 VAE).
- Two NVIDIA GPUs of the same generation (tested target: RTX 5080 + RTX 5070 Ti, Blackwell).
- Windows or Linux. No NCCL, no `torch.distributed`, no peer-to-peer GPU access needed: everything runs in WanGP's process.
- VAE tiling on (the H3 default). With tiling off there is only one tile per clip and nothing to split.

## Install

1. Copy the `wan2gp-multi-gpu` folder into `WanGP/plugins/`.
2. Make sure WanGP sees both GPUs: **do not start WanGP with `--gpu`**. That option sets `CUDA_VISIBLE_DEVICES` and hides
   the other GPUs.
3. Start WanGP, enable **Multi-GPU** in the plugin manager, restart WanGP.
4. The console should print `[Multi-GPU] Installed`.

## Use

In the Media Generator, switch to Advanced Mode and open the **Multi-GPU** tab (after **Misc.**):

- **Enable multi-GPU**: master on/off switch. Off means WanGP behaves exactly as without the plugin.
- **Denoising** (experimental)
  - **Split H3 denoising across cuda:0 and cuda:1**: off by default.
  - **Denoising work share of cuda:0 (%)**: share of the tokens denoised on `cuda:0`. Default 50; about 56 for an
    RTX 5080 (`cuda:0`) with an RTX 5070 Ti (`cuda:1`).
- **VAE decoding**
  - **Split H3 VAE decoding across cuda:0 and cuda:1**: on by default (once the master switch is on).
  - **Work share of cuda:0 (%)**: share of the tiles decoded on `cuda:0`; `cuda:1` gets the rest. Default 50.
- **Text encoder**
  - **Keep the last text encoder layers on cuda:1**: off by default, and ignored while denoising is split.
  - **Text encoder layers kept on cuda:1**: N, from 0 to 47. Default 20 (about 9.4 GB with the INT8 text encoder).
- The tab lists the GPUs WanGP sees, with their `cuda:N` numbers, and the result of the last text encoding and decode.

Settings are saved in `settings.json` next to the plugin and apply to the next prompt or decode, including queued jobs.

### Choosing the number of text encoder layers

More layers on `cuda:1` means faster new prompts and less free VRAM on `cuda:1`. Today `cuda:1` is only used for the VAE
copy and the text encoder, so on a 16 GB card 20 is a safe value and the plugin trims the number if it does not fit.
When denoising is split across both GPUs (next milestone), that VRAM will be needed for denoising and this default will
change.

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
| `denoise_split.py` | Denoising split: token rows per GPU, hook-free copies of the blocks for the secondary GPU, weight streaming, key/value exchange. |

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

- MiniMax H3 only.
- Denoising: standard attention path only (see the list of options that keep a pass on one GPU above).
- Two GPUs: `cuda:0` and `cuda:1`. More GPUs are not used.
- The first decode after enabling the plugin also builds the decoder copy on `cuda:1` (a few seconds, plus 2 to 3 GB of RAM
  while loading).
- The text encoder layers stay on `cuda:1` when you switch to another model; press **Free secondary GPU memory** to give
  that VRAM back. A different text encoder file is detected and the copy is rebuilt.
- Image and video references given to H3 are encoded by the Qwen3-VL vision tower on `cuda:0` as before; only the
  language layers are split.
- VAEs loaded by other code paths than the H3 model loader (for example some post-processing tools) are decoded on one GPU.

## Changelog

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
