"""Hook-free copies of modules managed by WanGP's memory manager (MMGP).

The copies are the same module objects rebuilt without MMGP's hooks. Their tensors point to the CPU tensors that MMGP
keeps (in its pinned "Reserved RAM" when the profile pins the model), so nothing is duplicated in RAM; the plugin installs
copies of these tensors on the secondary GPU while a copy runs. LoRA linears keep MMGP's LoRA forward, with the adapter
weights of the secondary GPU.

Used for LTX-2 (written for mmgp 3.7.6, unchanged in mmgp 4.x, which keeps ``_mm_lora_model`` out of the child
modules). The per-module attributes used here are
``_mm_manager`` / ``_mm_forward`` (load hooks), ``_mm_lora_old_forward`` / ``_mm_lora_data`` / ``_mm_lora_model`` (LoRA
hooks) and ``offload.blocks_of_modules`` (the CPU tensors of every hooked module).
"""
import functools
import types

import torch
import torch.nn as nn

from .common import warn_once

_HOOK_DICTS = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called", "_forward_pre_hooks",
               "_forward_pre_hooks_with_kwargs", "_backward_hooks", "_backward_pre_hooks")


class CloneError(RuntimeError):
    pass


def _rebind(callable_, original, clone):
    """A callable bound to `original` (bound method or functools.partial with it as first argument), bound to `clone`."""
    if isinstance(callable_, types.MethodType) and callable_.__self__ is original:
        return types.MethodType(callable_.__func__, clone)
    if isinstance(callable_, functools.partial) and callable_.args and callable_.args[0] is original:
        return functools.partial(callable_.func, clone, *callable_.args[1:], **(callable_.keywords or {}))
    raise CloneError(f"unsupported forward replacement on module {type(original).__name__}")


def _is_mmgp_loader(callable_):
    func = callable_.func if isinstance(callable_, functools.partial) else callable_
    name = getattr(func, "__name__", "")
    return name.startswith("_mm_wrap_") or name.startswith("check_load_into_GPU_needed") or name == "check_change_module"


def _is_compiled(callable_):
    return callable_ is not None and (hasattr(callable_, "_torchdynamo_orig_callable")
                                      or type(callable_).__name__ in ("OptimizedModule", "_TorchDynamoContext"))


def _is_reference(name):
    """Children registered by MMGP that are references, not parts of the module: mmgp 3.7.6 sets
    ``linear._mm_lora_model = model`` on every LoRA-hooked layer, and nn.Module registers that model (the whole
    transformer) as a child. Walking into it from a block reaches every other block."""
    return name.startswith("_mm_")


def own_modules(root):
    """root.modules() without MMGP's reference children (see _is_reference)."""
    seen, stack, out = set(), [root], []
    while stack:
        module = stack.pop()
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        out.append(module)
        stack.extend(child for name, child in reversed(list(module._modules.items())) if not _is_reference(name))
    return out


def module_is_compiled(module):
    for m in own_modules(module):
        if _is_compiled(m.__dict__.get("forward")) or _is_compiled(m.__dict__.get("_mm_forward")):
            return True
    return False


def clone_module(module, memo, shared=frozenset()):
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
                          f"Secondary GPU copies ignore {len(values[key])} {key} of {type(module).__name__}.")
            values[key] = type(values[key])()
    values["_parameters"] = type(module._parameters)(module._parameters)
    values["_buffers"] = type(module._buffers)(module._buffers)
    values["_modules"] = type(module._modules)()
    values.pop("_mm_forward", None)
    forward = values.pop("forward", None)
    clone.__dict__.update(values)
    memo[id(module)] = clone
    for name, child in module._modules.items():
        if child is None or _is_reference(name):
            clone._modules[name] = child   # MMGP's reference to the model (LoRA scales, active adapters): kept as is
        else:
            clone._modules[name] = clone_module(child, memo, shared)

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
        if isinstance(inner, types.MethodType) and inner.__self__ is module \
                and inner.__func__ is getattr(type(module), "forward", None):
            pass  # the class forward: nothing to keep
        elif _is_mmgp_loader(inner):
            pass
        elif _is_compiled(inner):
            raise CloneError("the model is compiled (torch.compile)")
        else:
            clone.__dict__["forward"] = _rebind(inner, module, clone)
            # A forward replaced by another plugin may compute inputs on the primary GPU (for example from tensors it
            # keeps): the layers it calls take them on the device of their own weights.
            for child in clone._modules.values():
                if child is not None and "weight" in child._parameters:
                    _follow_weight_device(child)
    return clone


def _follow_weight_device(module):
    """The module takes its input on the device of its weights. An input left on the other GPU goes through RAM (no
    direct GPU-to-GPU copy on GeForce cards). The last input moved is kept: one transfer per pass for a tensor that
    another plugin keeps for a whole pass."""
    inner = module.__dict__.get("forward")
    if inner is None:
        inner = types.MethodType(type(module).forward, module)
    moved = {}

    def forward(x, *args, **kwargs):
        weight = module._parameters.get("weight")
        if torch.is_tensor(x) and weight is not None and x.device != weight.device:
            key = (id(x), x.data_ptr(), tuple(x.shape), x.dtype, x.device, weight.device)
            if moved.get("key") != key:
                if x.is_cuda:
                    torch.cuda.synchronize(x.device)
                host = x.detach().to("cpu")
                moved.clear()
                moved.update(key=key, source=x, value=host.to(weight.device, non_blocking=False))
            x = moved["value"]
        return inner(x, *args, **kwargs)
    module.__dict__["forward"] = forward


def mmgp_index(modules):
    """{id(module): [(module, name, cpu tensor, is_buffer, tied)]} from MMGP's registry, for the given modules."""
    wanted = {id(m) for m in modules}
    manager = None
    for m in modules:
        manager = m.__dict__.get("_mm_manager")
        if manager is not None:
            break
    index = {}
    if manager is None or getattr(manager, "blocks_of_modules", None) is None:
        return index
    for params in manager.blocks_of_modules.values():
        for entry in params:
            if id(entry[0]) in wanted:
                index.setdefault(id(entry[0]), []).append(entry)
    return index


def to_host(tensor):
    """A CPU copy of a tensor kept on a GPU (pinned, so that the secondary GPU reads it at full speed)."""
    data = tensor.data if isinstance(tensor, nn.Parameter) else tensor
    if not data.is_cuda:
        return data
    host = torch.empty(data.shape, dtype=data.dtype, device="cpu", pin_memory=True) if type(data) is torch.Tensor else None
    if host is None:   # quantized tensor subclass: let it move its inner tensors
        return data.to("cpu")
    host.copy_(data, non_blocking=True)
    torch.cuda.current_stream(data.device).synchronize()
    return host


def registry(root, memo, index):
    """(clone module, name, CPU tensor, is_buffer, (tied clone, name) or None) for every tensor under `root`. The CPU
    tensors are those MMGP keeps for these modules, so that nothing is copied twice in RAM."""
    modules = {id(m): m for m in own_modules(root) if id(m) in memo}
    entries, covered = [], set()
    for key in modules:
        for parent, name, tensor, is_buffer, tied in index.get(key, ()):
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
                entries.append((memo[id(module)], name, to_host(tensor), is_buffer, None))
    return entries


def install(module, name, tensor, is_buffer):
    if is_buffer:
        module._buffers[name] = tensor
    else:
        module._parameters[name] = tensor


def strip_mmgp(root):
    """Remove MMGP's per-module attributes from a persistent copy: they reference the memory manager, which holds the
    pinned RAM of every model (a copy kept across a model switch would keep that RAM from being freed)."""
    for module in own_modules(root):
        for key in [k for k in module._modules if _is_reference(k)]:
            del module._modules[key]
        for key in [k for k in module.__dict__ if k.startswith("_mm_") or k == "_hf_hook"]:
            del module.__dict__[key]
