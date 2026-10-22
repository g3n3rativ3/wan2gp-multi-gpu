"""Helpers shared by the multi-GPU features (MiniMax H3 and LTX-2: denoising, VAE and text encoder splits)."""
import dataclasses
from concurrent.futures import ThreadPoolExecutor

import torch

_executors = {}
_warned = set()


def warn_once(key, message):
    if key not in _warned:
        _warned.add(key)
        print(f"[Multi-GPU] {message}")


def gpu_list():
    """[(index, name, total GB)] of the GPUs visible to WanGP."""
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        out.append((i, props.name, props.total_memory / 1024 ** 3))
    return out


def gpu_name(index):
    for i, name, _ in gpu_list():
        if i == index:
            return name
    return "?"


def other_gpu(primary_index):
    """The secondary GPU: cuda:1 when WanGP runs on cuda:0, cuda:0 otherwise."""
    return 1 if primary_index == 0 else 0


def device_index(device):
    device = torch.device(device)
    return device.index if device.index is not None else torch.cuda.current_device()


def _bind_thread(index):
    if torch.cuda.is_available() and index is not None and index >= 0:
        torch.cuda.set_device(index)


def executor(index, role="compute"):
    """One worker thread per secondary GPU and role, bound to that GPU (its current device is set once, when the thread
    starts). Roles: "compute" (VAE, text encoder and denoising work) and "loader" (weights streamed for denoising).
    index -1 is the CPU stand-in used by the tests."""
    pool = _executors.get((index, role))
    if pool is None:
        pool = _executors[(index, role)] = ThreadPoolExecutor(max_workers=1,
                                                              thread_name_prefix=f"wangp_multigpu_cuda{index}_{role}",
                                                              initializer=_bind_thread, initargs=(index,))
    return pool


def clear_stale_cuda_error(index):
    """A failed CUDA call leaves its error pending in the calling thread until the next check. The threads of the
    secondary GPU are reused by every stage (text encoder, denoising, VAE), so a failure in one stage must not be
    reported by the first operation of the next one: a tiny kernel reads (and so clears) any pending error."""
    if not torch.cuda.is_available() or index is None or index < 0:
        return
    for _ in range(2):
        try:
            torch.empty(1, device=torch.device("cuda", index)).fill_(0)
            return
        except Exception:
            continue


def to_pinned_host(tensor):
    """Copy a GPU tensor to pinned RAM and wait for the copy. GPU-to-GPU copies always go through RAM: GeForce cards have
    no peer-to-peer access, and memory from WanGP's VMM allocator cannot be read directly by another GPU."""
    host = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu", pin_memory=True)
    host.copy_(tensor, non_blocking=True)
    torch.cuda.current_stream(tensor.device).synchronize()
    return host


def map_tensors(value, fn):
    """Apply fn to every tensor found in nested tuples, lists and dicts; anything else is returned unchanged."""
    if torch.is_tensor(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(v, fn) for v in value)
    if isinstance(value, list):
        return [map_tensors(v, fn) for v in value]
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    return value


def tensor_bytes(tensor):
    """Size in bytes of a tensor, including quantized tensors made of several inner tensors (quanto INT8: data + scale)."""
    data = tensor.data if isinstance(tensor, torch.nn.Parameter) else tensor
    flatten = getattr(data, "__tensor_flatten__", None)
    if flatten is not None:
        try:
            names, _ = flatten()
            return sum(tensor_bytes(getattr(data, name)) for name in names)
        except Exception:
            pass
    return data.numel() * data.element_size()


def module_bytes(module):
    return sum(tensor_bytes(t) for t in list(module.parameters()) + list(module.buffers()))


def match_float_dtypes(replica, primary):
    """WanGP's memory manager may convert leftover FP32 tensors of a model after loading: give the copy the same floating
    point dtypes, so both GPUs compute with the same precision. Tensors are matched by name."""
    primary_tensors = dict(primary.named_parameters())
    primary_tensors.update(dict(primary.named_buffers()))
    for name, tensor in list(replica.named_parameters()) + list(replica.named_buffers()):
        reference = primary_tensors.get(name)
        if (reference is not None and type(reference.data) is torch.Tensor and type(tensor.data) is torch.Tensor
                and reference.dtype.is_floating_point and tensor.dtype.is_floating_point and reference.dtype != tensor.dtype):
            tensor.data = tensor.data.to(reference.dtype)


def deep_map(value, fn, memo=None):
    """Apply fn to every tensor found in nested tuples, lists, dicts and dataclass instances (LTX-2's TransformerArgs,
    RopeCache, ...). A tensor referenced several times is mapped once. Anything else is returned unchanged."""
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        key = id(value)
        if key not in memo:
            memo[key] = (value, fn(value))   # keep the source alive so that its id is not reused
        return memo[key][1]
    if isinstance(value, tuple) and hasattr(value, "_fields"):   # NamedTuple
        return type(value)(*(deep_map(v, fn, memo) for v in value))
    if isinstance(value, tuple):
        return tuple(deep_map(v, fn, memo) for v in value)
    if isinstance(value, list):
        return [deep_map(v, fn, memo) for v in value]
    if isinstance(value, dict):
        return {k: deep_map(v, fn, memo) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {f.name: deep_map(getattr(value, f.name), fn, memo) for f in dataclasses.fields(value) if f.init}
        return dataclasses.replace(value, **changes)
    return value


def iter_tensors(value, seen=None):
    """Every distinct tensor found in nested containers and dataclasses."""
    if seen is None:
        seen = set()
    if torch.is_tensor(value):
        if id(value) not in seen:
            seen.add(id(value))
            yield value
        return
    if isinstance(value, (tuple, list)):
        for v in value:
            yield from iter_tensors(v, seen)
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_tensors(v, seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from iter_tensors(getattr(value, f.name), seen)


def cuda_graphs_on(module):
    """Whether WanGP's memory manager (mmgp 4.x) runs the blocks of the model owning `module` as CUDA graphs. A split pass
    must not: a graph would capture the exchange between the GPUs (host waits, events of the other GPU)."""
    try:
        from mmgp import cuda_graphs
        graphs = cuda_graphs._graphs(module)
    except Exception:
        return False
    return graphs is not None and getattr(graphs, "disabled", None) is None
