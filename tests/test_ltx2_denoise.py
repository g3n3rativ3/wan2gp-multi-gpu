"""CPU check of the LTX-2 denoising split (WanGP v18.17+): the split forward must match the one-device forward.

Both "GPUs" are the CPU here (the secondary side runs in its own thread, with its own block copies and the real key/value
exchange), so this checks the arithmetic of the split: row cutting, RoPE cache slicing, timestep slicing, audio
replication, key/value exchange and reordering, joint (CFG) passes, fallbacks.
"""
import dataclasses
import importlib
import os
import sys
import types

import torch
import torch.nn.functional as F

WANGP = sys.argv[1]
PLUGINS = sys.argv[2]
sys.path.insert(0, WANGP)
sys.path.insert(0, PLUGINS)

# --- stub of shared.attention (the real one needs CUDA at import time) ---------------------------------------------------
calls = {"n": 0}


def pay_attention(qkv_list, dropout_p=0., softmax_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                  version=None, force_attention=None, attention_mask=None, recycle_q=False, q_lens=None, k_lens=None):
    q, k, v = qkv_list
    qkv_list.clear()
    calls["n"] += 1
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attention_mask)
    return out.transpose(1, 2).contiguous()


# CPU stand-ins for the CUDA queries WanGP makes even on CPU tensors
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
attention_stub.pay_attention = pay_attention
attention_stub.sage_staged = lambda *args, **kwargs: None
attention_stub.sage_attention_mask = lambda mask, dtype: mask
sys.modules["shared.attention"] = attention_stub

from models.ltx2.ltx_core.model.transformer.model import LTXModel, LTXModelType  # noqa: E402
from models.ltx2.ltx_core.model.transformer.modality import Modality  # noqa: E402
from models.ltx2.ltx_core.model.transformer.rope import LTXRopeType, apply_rope_cache_inplace  # noqa: E402
from models.ltx2.ltx_core.components.patchifiers import VideoLatentPatchifier, AudioPatchifier  # noqa: E402
from models.ltx2.ltx_core.tools import VideoLatentTools, AudioLatentTools  # noqa: E402
from models.ltx2.ltx_core.types import VideoLatentShape, AudioLatentShape  # noqa: E402
from models.ltx2.ltx_core.guidance.perturbations import (BatchedPerturbationConfig, Perturbation,  # noqa: E402
                                                        PerturbationConfig, PerturbationType)

PLUGIN = os.environ.get("MULTIGPU_PLUGIN", "wan2gp-multi-gpu")
ds = importlib.import_module(f"{PLUGIN}.ltx2_denoise")
kit = importlib.import_module("shared.attention_kit")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strict_device import StrictDeviceContext, violations  # noqa: E402
settings = importlib.import_module(f"{PLUGIN}.settings")

torch.manual_seed(0)
DT = torch.float32


def build_model(rope_type):
    model = LTXModel(model_type=LTXModelType.AudioVideo, num_attention_heads=4, attention_head_dim=32, in_channels=16,
                     out_channels=16, num_layers=3, cross_attention_dim=128, caption_channels=128,
                     audio_num_attention_heads=4, audio_attention_head_dim=16, audio_in_channels=16,
                     audio_out_channels=16, audio_cross_attention_dim=64, rope_type=rope_type,
                     apply_gated_attention=True, cross_attention_adaln=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.05 if p.dim() > 1 else 0.1))
    return model.eval()


# helpers of the pipeline (copied from ltx_pipelines/utils/helpers.py to avoid its heavy imports)
helpers_src = open(os.path.join(WANGP, "models/ltx2/ltx_pipelines/utils/helpers.py")).read()
start = helpers_src.index("def build_timestep_compression_plan(")
end = helpers_src.index("def _prepare_conditioning_context(")
ns = {"torch": torch, "math": __import__("math"), "TimestepCompressionPlan":
      importlib.import_module("models.ltx2.ltx_core.types").TimestepCompressionPlan}
exec(helpers_src[start:end], ns)


def modality(state, context, sigma):
    plan = ns["build_timestep_compression_plan"](state.denoise_mask, state.positions)
    timesteps, frame_indices = ns["timesteps_from_mask"](state.denoise_mask, sigma, plan=plan)
    s = torch.full((state.latent.shape[0],), sigma, dtype=state.latent.dtype)
    return Modality(latent=state.latent, sigma=s, timesteps=timesteps, positions=state.positions, context=context,
                    frame_indices=frame_indices)


def inputs(mask_mode, frames=6, h=8, w=12):
    vtools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), VideoLatentShape(1, 16, frames, h, w), fps=24.0)
    vstate = vtools.create_initial_state("cpu", DT, torch.randn(1, 16, frames, h, w))
    if mask_mode == "first_frame":
        mask = vstate.denoise_mask.clone()
        mask[:, : h * w] = 0.0
        vstate = dataclasses.replace(vstate, denoise_mask=mask)
    elif mask_mode == "per_token":
        vstate = dataclasses.replace(vstate, denoise_mask=torch.rand_like(vstate.denoise_mask))
    ashape = AudioLatentShape(1, 8, 43, 2)
    atools = AudioLatentTools(AudioPatchifier(patch_size=1), ashape)
    astate = atools.create_initial_state("cpu", DT, torch.randn(*ashape.to_torch_shape()))
    vctx = torch.randn(1, 20, 128)
    actx = torch.randn(1, 20, 64)
    return modality(vstate, vctx, 0.7), modality(astate, actx, 0.7)


def run(model, video, audio, perturbations=None, split=False, share=50):
    settings.update(enabled=split, ltx_denoise_enabled=True, ltx_denoise_share_cuda0=share)
    ds._state["broken"] = None
    with torch.inference_mode(), StrictDeviceContext("cpu"):   # WanGP generates inside `with torch.device(...)`
        return model(video, audio, perturbations)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def check(name, ref, out, tol=1e-5):
    vr, ar = ref
    vo, ao = out
    if isinstance(vr, list):
        errs = [rel(o, r) for o, r in zip(vo, vr)] + [rel(o, r) for o, r in zip(ao, ar) if r is not None]
    else:
        errs = [rel(vo, vr)] + ([rel(ao, ar)] if ar is not None else [])
    split_ran = ds._state["last_report"].startswith("Last LTX-2 denoising pass:")   # not "... pass on one GPU"
    ok = max(errs) < tol and ds._state["broken"] is None and split_ran
    print(f"{'OK  ' if ok else 'FAIL'} {name}: max rel err {max(errs):.2e} | {ds._state['last_report'][:150]}")
    if ds._state["broken"]:
        print("   broken:", ds._state["broken"])
    return ok


# the MiniMax H3 split wraps attention_kit.qkv_attention too (installed first by the plugin, so its wrapper sits under
# the LTX-2 one); a pass-through wrapper stands for it here (importing the H3 model needs the real shared.attention)
_h3_original = kit.qkv_attention


def _h3_like_wrapper(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, **kwargs):
    return _h3_original(x_list, q_proj, k_proj, v_proj, heads, head_dim, norm_rope, **kwargs)


kit.qkv_attention = _h3_like_wrapper
ds.install()
ds._state["test_secondary"] = torch.device("cpu")
ds.MIN_TOKENS = 64
ds.ROW_ALIGN = 32
all_ok = True

# RoPE cache slicing alone: rotating a slice with the sliced cache == slicing the rotated full tensor
for rope_type in (LTXRopeType.SPLIT, LTXRopeType.INTERLEAVED):
    model = build_model(rope_type)
    video, audio = inputs("uniform")
    args = model.video_args_preprocessor.prepare(video)
    for pe_name in ("positional_embeddings", "cross_positional_embeddings"):
        pe = getattr(args, pe_name)
        dim = 128 if pe_name == "positional_embeddings" else 64
        x = torch.randn(1, pe.token_count, dim)
        full = apply_rope_cache_inplace(x.clone(), pe)
        worst = 0.0
        for s, e in ((0, 37), (37, 576), (5, 101), (96, 480), (0, 576), (300, 301), (95, 97)):
            part = apply_rope_cache_inplace(x[:, s:e].clone(), ds._slice_rope(pe, s, e, pe.token_count))
            worst = max(worst, (part - full[:, s:e]).abs().max().item())
        ok = worst == 0.0
        all_ok &= ok
        print(f"{'OK  ' if ok else 'FAIL'} rope slice {rope_type.value} {pe_name}: max abs diff {worst:.1e}")

model = build_model(LTXRopeType.SPLIT)
for mode in ("uniform", "first_frame", "per_token"):
    video, audio = inputs(mode)
    ref = run(model, video, audio)
    for share in (50, 30, 80):
        out = run(model, video, audio, split=True, share=share)
        all_ok &= check(f"{mode} share {share}", ref, out)

# video only (no audio stream)
video, audio = inputs("uniform")
ref = run(model, video, None)
all_ok &= check("video only", ref, run(model, video, None, split=True))

# joint pass (CFG: two conditions in one call), with an STG-like perturbation on the second one
video, audio = inputs("first_frame")
video_neg = dataclasses.replace(video, context=torch.randn(1, 20, 128))
audio_neg = dataclasses.replace(audio, context=torch.randn(1, 20, 64))
perts = [BatchedPerturbationConfig.empty(1),
         BatchedPerturbationConfig([PerturbationConfig([Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, [1]),
                                                        Perturbation(PerturbationType.SKIP_V2A_CROSS_ATTN, None)])])]
ref = run(model, [video, video_neg], [audio, audio_neg], perts)
all_ok &= check("joint CFG + perturbations", ref, run(model, [video, video_neg], [audio, audio_neg], perts, split=True))

# attention routes of WanGP v18.17+: with Attention Head Split, the video-to-audio cross-attention goes through
# attention_kit.cross_qkv_attention; with a backend forced by the model, every attention calls pay_attention itself
video, audio = inputs("first_frame")
kit.MIN_SPLIT_TOKENS, saved_min = 0, kit.MIN_SPLIT_TOKENS
kit.configure(2)
ref = run(model, video, audio)
routed = {"n": 0}
_cross = ds._state["orig_kit_cross"]


def _counting_cross(*args, **kwargs):
    routed["n"] += kwargs.get("attention_fn") is not None
    return _cross(*args, **kwargs)


ds._state["orig_kit_cross"] = _counting_cross
ok = check("Attention Head Split on (attention_kit cross-attention route)", ref, run(model, video, audio, split=True))
ds._state["orig_kit_cross"] = _cross
print(f"     video-to-audio attentions distributed through cross_qkv_attention: {routed['n']}")
all_ok &= ok and routed["n"] > 0
kit.configure(0)
kit.MIN_SPLIT_TOKENS = saved_min
from models.ltx2.ltx_core.model.transformer.attention import Attention, AttentionFunction  # noqa: E402
model_f = build_model(LTXRopeType.SPLIT)
for m in model_f.modules():
    if isinstance(m, Attention):
        m.attention_function = AttentionFunction.PYTORCH
ref = run(model_f, video, audio)
all_ok &= check("attention backend forced by the model (pay_attention route)", ref, run(model_f, video, audio, split=True))

# interleaved RoPE model
model_i = build_model(LTXRopeType.INTERLEAVED)
video, audio = inputs("per_token")
all_ok &= check("interleaved rope", run(model_i, video, audio), run(model_i, video, audio, split=True))


# --- the way mmgp hooks the transformer: block registry, load wrappers, and LoRA hooks (mmgp 4.x keeps the model of a
# LoRA layer in its __dict__; mmgp 3.7.6 registered it as a child module, which the copies also handle).
import functools  # noqa: E402
from mmgp import offload as mmgp_offload  # noqa: E402


def mmgp_like_hooks(model, rank=4, active=True):
    manager = mmgp_offload.offload.__new__(mmgp_offload.offload)
    manager.auto_preload_requested = manager.preload_plans = False
    manager.blocks_of_modules = {}
    for bi, blk in enumerate(model.transformer_blocks):
        manager.blocks_of_modules[f"transformer/transformer_blocks.{bi}"] = [
            (m, n, p, False, None) for m in blk.modules() for n, p in m.named_parameters(recurse=False)]
    for blk in model.transformer_blocks:
        for m in list(blk.modules()):
            previous = m.forward
            if isinstance(m, torch.nn.Linear):
                data = {}
                m._mm_lora_data = data
                m.__dict__["_mm_lora_model"] = model           # as mmgp 4.x hook_lora does
                m._mm_lora_old_forward = previous
                previous = functools.update_wrapper(functools.partial(mmgp_offload._mm_lora_linear_forward, m), previous)
                a = torch.randn(rank, m.in_features) * 0.3
                b = torch.randn(m.out_features, rank) * 0.3
                data["a"] = [a, b, None, None, 1.0, {}]
                data["a_GPU"] = [a, b, None, None, 1.0, {}]

            def check_load_into_GPU_needed_other(module, *args, _previous=previous, **kwargs):
                return _previous(*args, **kwargs)
            m._mm_manager = manager
            m._mm_forward = previous
            m.forward = functools.update_wrapper(functools.partial(check_load_into_GPU_needed_other, m), previous)
    model._loras_active_adapters = ["a"] if active else []
    model._loras_scaling = {"a": 1.0}
    model._loras_model_shortcuts = {}
    return model


model_m = mmgp_like_hooks(build_model(LTXRopeType.SPLIT))
video, audio = inputs("first_frame")
model_m._loras_active_adapters = []
no_lora = run(model_m, video, audio)
model_m._loras_active_adapters = ["a"]
ref = run(model_m, video, audio)
lora_effect = rel(ref[0], no_lora[0])
out = run(model_m, video, audio, split=True)
ok = check("mmgp-hooked model with an active LoRA", ref, out)
ok = ok and lora_effect > 1e-2
all_ok &= ok
print(f"     (the LoRA changes the one-GPU output by {lora_effect:.2e}, so the comparison is meaningful)")


# --- LTX-2.5 (and its INT8 ConvRot checkpoint): feed-forward without bias, learned keyframe embedding, INT8 ConvRot
# linears loaded the way WanGP loads them (mmgp's quanto router with WanGP's ConvRot handler), LoRA hooks on top
import threading  # noqa: E402
from shared.qtypes import int8_convrot  # noqa: E402
from mmgp import quant_router  # noqa: E402
from optimum.quanto import requantize  # noqa: E402


def build_model_25():
    model = LTXModel(model_type=LTXModelType.AudioVideo, num_attention_heads=4, attention_head_dim=32, in_channels=16,
                     out_channels=16, num_layers=3, cross_attention_dim=128, caption_channels=128,
                     audio_num_attention_heads=4, audio_attention_head_dim=16, audio_in_channels=16,
                     audio_out_channels=16, audio_cross_attention_dim=64, rope_type=LTXRopeType.SPLIT,
                     apply_gated_attention=True, cross_attention_adaln=True, ff_bias=False,
                     use_keyframes_abs_pos_embedding=True)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(torch.randn_like(p) * (0.05 if p.dim() > 1 else 0.1))
        model.keyframes_abs_pos_embedding.normal_(0, 0.5)
    return model.eval()


def convrot_int8(model):
    """Quantize the block linears to INT8 ConvRot as in WanGP's ltx-2.5 *_int8_convrot.safetensors checkpoints."""
    quant_router.register_handler(int8_convrot)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for name, module in model.named_modules():
        if not name.startswith("transformer_blocks.") or not isinstance(module, torch.nn.Linear):
            continue
        group = next(g for g in (256, 64, 16) if module.in_features % g == 0)
        h = int8_convrot._regular_hadamard(group, "cpu", torch.float32)
        w = module.weight.detach().float()
        w = (w.reshape(w.shape[0], -1, group) @ h).reshape(w.shape)
        scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127
        state[name + ".weight"] = torch.round(w / scale).clamp(-128, 127).to(torch.int8)
        state[name + ".weight_scale"] = scale
        config = '{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": %d}' % group
        state[name + ".comfy_quant"] = torch.tensor(list(config.encode()), dtype=torch.uint8)
    out = int8_convrot.convert_to_quanto(state, DT)
    requantize(model, out["state_dict"], out["quant_map"], device=torch.device("cpu"))
    return model


rotations = {}
_rotate = int8_convrot._rotate_activation


def _counting_rotate(x, group):
    name = threading.current_thread().name
    rotations[name] = rotations.get(name, 0) + 1
    return _rotate(x, group)


int8_convrot._rotate_activation = _counting_rotate
model_25 = convrot_int8(build_model_25())
convrot_layers = sum(getattr(m, "_convrot_group_size", 0) > 0 for m in model_25.transformer_blocks.modules())
video, audio = inputs("first_frame")
keyframes = torch.zeros(1, video.latent.shape[1], 1)
keyframes[:, :96] = 1.0   # the first latent frame carries a keyframe (start image)
video_kf = dataclasses.replace(video, keyframes_mask=keyframes)
plain = run(model_25, video, audio)
ref = run(model_25, video_kf, audio)
rotations.clear()
out = run(model_25, video_kf, audio, split=True)
secondary = sum(n for name, n in rotations.items() if name.startswith("wangp_multigpu_cuda-1_compute"))
ok = check("LTX-2.5: INT8 ConvRot weights, no feed-forward bias, keyframe embedding", ref, out)
ok = ok and convrot_layers > 0 and secondary > 0 and rel(ref[0], plain[0]) > 1e-3
all_ok &= ok
print(f"     {convrot_layers} INT8 ConvRot linears; ConvRot projections computed by the secondary side: {secondary}; "
      f"the keyframe embedding changes the output by {rel(ref[0], plain[0]):.1e}")
video_neg = dataclasses.replace(video_kf, context=torch.randn(1, 20, 128))
audio_neg = dataclasses.replace(audio, context=torch.randn(1, 20, 64))
ref = run(model_25, [video_kf, video_neg], [audio, audio_neg])
all_ok &= check("LTX-2.5 INT8 ConvRot joint CFG pass", ref, run(model_25, [video_kf, video_neg], [audio, audio_neg],
                                                                   split=True))
model_25m = mmgp_like_hooks(convrot_int8(build_model_25()))
ref = run(model_25m, video_kf, audio)
all_ok &= check("LTX-2.5 INT8 ConvRot, mmgp-hooked with an active LoRA", ref, run(model_25m, video_kf, audio, split=True))
int8_convrot._rotate_activation = _rotate


# --- v0.2: video-to-audio attention by partial results + exact merge ---------------------------------------------------
for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
    q = torch.randn(1, 43, 4, 16, dtype=dtype)
    k = torch.randn(1, 600, 4, 16, dtype=dtype) * 2
    v = torch.randn(1, 600, 4, 16, dtype=dtype)
    full = pay_attention([q.float(), k.float(), v.float()])
    worst = sabotaged = 0.0
    for cut in (1, 37, 300, 599):
        a = ds._partial_attention(q, k[:, :cut], v[:, :cut], score_bytes=40_000)   # small budget: several head chunks
        b = ds._partial_attention(q, k[:, cut:], v[:, cut:])
        worst = max(worst, rel(ds._merge_partial(a, b), full))
        sabotaged = max(sabotaged, rel(b[0], full))   # one part alone must be clearly wrong
    ok = worst < tol and sabotaged > 1e-2
    all_ok &= ok
    print(f"{'OK  ' if ok else 'FAIL'} partial attention + merge ({dtype}): max rel err {worst:.1e} "
          f"(one part alone: {sabotaged:.1e})")

# --- v0.2: block copies kept between the passes of a generation, dropped at its end ------------------------------------
model_c = mmgp_like_hooks(build_model(LTXRopeType.SPLIT))
video, audio = inputs("first_frame")
ds.drop_clones()
refs = []
for scale in (1.0, 0.3):   # LoRA strength changed between two steps: the kept copies read it live
    model_c._loras_scaling = {"a": scale}
    refs.append(run(model_c, video, audio))
model_c._loras_scaling = {"a": 1.0}
out1 = run(model_c, video, audio, split=True)
first = ds._state["clones"]
model_c._loras_scaling = {"a": 0.3}
out2 = run(model_c, video, audio, split=True)
second = ds._state["clones"]
ok = (first is not None and second is first and rel(out1[0], refs[0][0]) < 1e-5 and rel(out2[0], refs[1][0]) < 1e-5
      and rel(refs[0][0], refs[1][0]) > 1e-2)
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} block copies reused by the next step, LoRA strength read live "
      f"({rel(out1[0], refs[0][0]):.1e}, {rel(out2[0], refs[1][0]):.1e})")

ds.end_of_denoising()   # what the VAE decode does
ok = ds._state["clones"] is None
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} block copies dropped when the VAE decode starts")

# the watcher of LTX2.generate: copies dropped when a generation ends, even after an error
fake = types.ModuleType("models.ltx2.ltx2")


class LTX2:
    def generate(self, fail=False):
        run(model_c, video, audio, split=True)
        if fail:
            raise RuntimeError("error after the denoising")
        return ds._state["clones"] is not None


fake.LTX2 = LTX2
sys.modules["models.ltx2.ltx2"] = fake
ds._state["watching"] = False
run(model_c, video, audio, split=True)   # first split pass installs the watcher
during = LTX2().generate()
after_ok = ds._state["clones"] is None
try:
    LTX2().generate(fail=True)
except RuntimeError:
    pass
ok = during and after_ok and ds._state["clones"] is None
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} block copies kept during a generation, dropped at its end (also after an error)")

# failure on the secondary side: the pass is computed again on one GPU, the feature is disabled
video, audio = inputs("uniform")
ref = run(model, video, audio)
orig_worker = ds._Session._worker_job


def failing_worker(self):
    self.fail(RuntimeError("simulated failure on the secondary GPU"))
    return None


ds._Session._worker_job = failing_worker
settings.update(enabled=True)
with torch.inference_mode():
    out = model(video, audio, None)
ds._Session._worker_job = orig_worker
ok = rel(out[0], ref[0]) < 1e-6 and ds._state["broken"] is not None
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} fallback after a failure: rerun on one GPU, feature disabled ({ds._state['broken']})")

# interruption: the model reports it, the secondary side stops
ds._state["broken"] = None
model.interrupt_check = lambda: True
settings.update(enabled=True)
with torch.inference_mode():
    out = model(video, audio, None)
model.interrupt_check = None
ok = out == (None, None) and model.interrupted
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} interruption")

ok = not violations
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} no allocation without an explicit device inside WanGP's device context {violations[:3]}")

# off: identical call path
settings.update(enabled=False)
calls["n"] = 0
ref2 = run(model, video, audio)
ok = torch.equal(ref2[0], ref[0])
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} disabled plugin gives the exact one-GPU result")

print("ALL OK" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
