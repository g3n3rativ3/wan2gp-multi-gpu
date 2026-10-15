"""Helpers shared by the multi-GPU features (VAE split, text encoder split)."""
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
    if torch.cuda.is_available():
        torch.cuda.set_device(index)


def executor(index, role="compute"):
    """One worker thread per secondary GPU and role, bound to that GPU (its current device is set once, when the thread
    starts). Roles: "compute" (VAE, text encoder and denoising work) and "loader" (weights streamed for denoising)."""
    pool = _executors.get((index, role))
    if pool is None:
        pool = _executors[(index, role)] = ThreadPoolExecutor(max_workers=1,
                                                              thread_name_prefix=f"wangp_multigpu_cuda{index}_{role}",
                                                              initializer=_bind_thread, initargs=(index,))
    return pool


def to_pinned_host(tensor):
    """Copy a GPU tensor to pinned RAM and wait for the copy. GPU-to-GPU copies always go through RAM: GeForce cards have
    no peer-to-peer access, and memory from WanGP's VMM allocator cannot be read directly by another GPU."""
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
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
