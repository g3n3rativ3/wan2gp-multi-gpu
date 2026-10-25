"""WanGP's INT8 kernel backends on two GPUs at once.

WGPKernels
----------
wgpkernels 1.1.2 assumes one GPU per process, which breaks when the plugin runs INT8 layers on the secondary GPU
("INT8 GEMM launch failed (CUDA error 1)"):
- its native library raises the shared memory limit of each GEMM kernel once per process (a ``static bool configured``
  around ``cudaFuncSetAttribute``), but that limit is a per-GPU setting: on the second GPU the kernel is launched with
  64 KB of shared memory without the raised limit, and CUDA refuses the launch;
- its L2 flush buffer used while autotuning is one global allocation: two GPUs tuning at the same time free each
  other's buffer;
- the Python side queries workspace sizes into one shared ctypes array, which two threads can overwrite between the
  query and its use.

The fix, without modifying WGPKernels: each secondary GPU gets its own copy of the native library (a copy of the file,
loaded separately, so its "once per process" state is its own), chosen by the calling thread's current GPU; the size
array becomes one per thread; and the tuning cache is saved under a lock. Nothing changes for one-GPU use: the primary
GPU keeps the library WanGP loaded.

Triton
------
WanGP's Triton INT8 kernels (shared/kernels/quanto_int8_triton.py) time their candidate configurations the first time
a shape is seen, inside a CUDA graph captured in PyTorch's default "global" mode. While one thread captures in that
mode, CUDA refuses allocations and other "unsafe" calls from every other thread, so the GPU working at the same time
fails ("CUDA error: the operation cannot be performed in the present state"). The fix: these tuning sessions run one
at a time, and their captures use the "thread_local" mode, which only restricts the capturing thread. Nothing else
changes: other CUDA graphs keep their own mode, and the tuning results are the same.

Triton compilation (WanGP v18.81+)
----------------------------------
Triton keeps the kernels it compiles in a file cache (``~/.triton/cache``): it writes each kernel's files to a
temporary name, then renames them, and reads them back right after. When the two GPUs compile the same kernel at the
same moment (each GPU's thread meets it for the first time in the same pass, such as WanGP's ``token_mean`` kernels of
Sage attention in the H3 VAE), Windows refuses to open a file another thread is replacing: "PermissionError: [Errno 13]
Permission denied: ...\\.triton\\cache\\...\\_finish_kernel.json". The fix: Triton compilations of this process run one at a
time (the second thread then finds the kernel in the cache), and the launch tuner of WanGP's Triton kernels
(``shared.kernels.triton_tuner.LaunchTuner``, whose cache file both threads would also write) tunes one kernel at a
time. Compilation happens once per kernel and shape, so nothing changes once the kernels are compiled.
"""
import functools
import ctypes
import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import torch

_lock = threading.Lock()
_state = {"proxy": None, "failed": {}}


class _PerDeviceLibrary:
    """Stands in for ``wgpkernels.int8._lib``: every function is taken from the library of the current GPU."""

    def __init__(self, primary):
        self._primary = primary
        self._libs = {}

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        lib = self._libs.get(torch.cuda.current_device(), self._primary) if self._libs else self._primary
        return getattr(lib, name)


class _ThreadSizes:
    """Stands in for ``wgpkernels.int8._sizes`` (two int64 written by the library, then read): one array per thread."""

    def __init__(self):
        self._tls = threading.local()

    def _array(self):
        array = getattr(self._tls, "array", None)
        if array is None:
            array = self._tls.array = (ctypes.c_int64 * 2)()
        return array

    @property
    def _as_parameter_(self):
        return self._array()

    def __getitem__(self, index):
        return self._array()[index]

    def __setitem__(self, index, value):
        self._array()[index] = value

    def __len__(self):
        return 2


def _locked_save_cache(module):
    def save_cache():
        with _lock:
            snapshot = dict(module._tuned or {})
            try:
                module._TUNING_PATH.parent.mkdir(parents=True, exist_ok=True)
                temporary = module._TUNING_PATH.with_suffix(".tmp")
                temporary.write_text(json.dumps({"version": module._TUNING_VERSION, "configs": snapshot}, indent=1),
                                     encoding="utf-8")
                os.replace(temporary, module._TUNING_PATH)
            except OSError as error:
                print(f"[INT8] Could not save the INT8 GEMM tuning cache {module._TUNING_PATH}: {error}")
    save_cache._wangp_mgpu = True
    return save_cache


def _library_copy(original, index):
    """A copy of the native library file for GPU ``index``, next to the original (else in the temp folder)."""
    original = Path(original)
    name = f"{original.stem}_mgpu_cuda{index}{original.suffix}"
    for folder in (original.parent, Path(tempfile.gettempdir()) / "wangp_multigpu"):
        target = folder / name
        try:
            source_stat = original.stat()
            if target.exists():
                stat = target.stat()
                if stat.st_size == source_stat.st_size and int(stat.st_mtime) == int(source_stat.st_mtime):
                    return target
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            return target
        except OSError:
            continue
    raise OSError(f"cannot write a copy of {original.name} next to it or in the temp folder")


def _prepare_wgp(primary_index, secondary_index):
    """Make WGPKernels usable on ``secondary_index``. Returns None when it is ready (or not in use), else the reason why
    INT8 layers cannot run on that GPU with WGPKernels. Call it from the main thread, before any work on that GPU."""
    module = sys.modules.get("wgpkernels.int8")
    if module is None or getattr(module, "_lib", None) is None:
        return None
    if secondary_index in _state["failed"]:
        return _state["failed"][secondary_index]
    with _lock:
        proxy = module._lib if isinstance(module._lib, _PerDeviceLibrary) else None
        if proxy is not None and (secondary_index in proxy._libs or secondary_index == primary_index):
            return None
        try:
            if not isinstance(module._sizes, _ThreadSizes):
                module._sizes = _ThreadSizes()
            if not getattr(module._save_cache, "_wangp_mgpu", False):
                module._save_cache = _locked_save_cache(module)
            primary_lib = proxy._primary if proxy is not None else module._lib
            copy = _library_copy(primary_lib._name, secondary_index)
            lib = module.load_library(copy)  # sets the argument types of the copy (and makes it module._lib for now)
            if proxy is None:
                proxy = _PerDeviceLibrary(primary_lib)
            proxy._libs[primary_index] = primary_lib
            proxy._libs[secondary_index] = lib
            module._lib = proxy
            _state["proxy"] = proxy
        except Exception as exc:
            if proxy is not None:
                module._lib = proxy
            elif "primary_lib" in locals():
                module._lib = primary_lib
            reason = f"{type(exc).__name__}: {exc}"
            _state["failed"][secondary_index] = reason
            print(f"[Multi-GPU] WGPKernels cannot be set up for cuda:{secondary_index} ({reason}); the INT8 work stays "
                  f"on one GPU. Comfy Kitchen or Triton INT8 kernels work on both GPUs.")
            return reason
    print(f"[Multi-GPU] WGPKernels: cuda:{secondary_index} uses its own copy of the INT8 kernel library ({copy.name}).")
    return None


# ----------------------------------------------------------------------------------------------------------------------
# Triton INT8 autotuning
# ----------------------------------------------------------------------------------------------------------------------

_tuning = threading.local()
_tuning_lock = threading.RLock()
_TRITON_MODULE = "shared.kernels.quanto_int8_triton"
_TRITON_TUNERS = ("_autotune_config", "_ensure_compile_compatible_config")


def _serialized_tuner(function):
    @functools.wraps(function)
    def tuner(*args, **kwargs):
        with _tuning_lock:
            depth = getattr(_tuning, "depth", 0)
            _tuning.depth = depth + 1
            try:
                return function(*args, **kwargs)
            finally:
                _tuning.depth = depth
    tuner._wangp_mgpu = True
    return tuner


def _install_graph_mode():
    original = torch.cuda.graph
    if getattr(original, "_wangp_mgpu", False):
        return

    def graph(cuda_graph, pool=None, stream=None, capture_error_mode=None, **kwargs):
        if capture_error_mode is None:
            capture_error_mode = "thread_local" if getattr(_tuning, "depth", 0) else "global"
        return original(cuda_graph, pool=pool, stream=stream, capture_error_mode=capture_error_mode, **kwargs)
    graph._wangp_mgpu = True
    graph._wangp_mgpu_original = original
    torch.cuda.graph = graph


def _prepare_triton():
    module = sys.modules.get(_TRITON_MODULE)
    if module is None or getattr(module, "_wangp_mgpu_patched", False):
        return None
    with _lock:
        if getattr(module, "_wangp_mgpu_patched", False):
            return None
        try:
            for name in _TRITON_TUNERS:
                function = getattr(module, name, None)
                if function is not None and not getattr(function, "_wangp_mgpu", False):
                    setattr(module, name, _serialized_tuner(function))
            _install_graph_mode()
            module._wangp_mgpu_patched = True
        except Exception as exc:  # never fatal: the Triton kernels still work, only their first tuning is unprotected
            print(f"[Multi-GPU] Could not prepare the Triton INT8 kernels for two GPUs ({type(exc).__name__}: {exc}).")
    return None


_compile_lock = threading.RLock()


def _locked(function, lock):
    @functools.wraps(function)
    def locked(*args, **kwargs):
        with lock:
            return function(*args, **kwargs)
    locked._wangp_mgpu = True
    locked._wangp_mgpu_original = function
    return locked


def install_triton_locks():
    """One Triton compilation at a time in this process, and one tuning of WanGP's launch tuner at a time (see the
    module docstring). Installed at startup, whatever the plugin's switches: a lock taken once per new kernel costs
    nothing, and the secondary GPU's threads may compile kernels as soon as a split starts. Returns what was installed."""
    installed = []
    try:
        import triton.compiler as triton_compiler
        from triton.compiler import compiler as triton_compiler_module
        from triton.runtime import jit as triton_jit
    except Exception:
        return installed
    do_compile = getattr(triton_jit.JITFunction, "_do_compile", None)
    if do_compile is not None and not getattr(do_compile, "_wangp_mgpu", False):
        triton_jit.JITFunction._do_compile = _locked(do_compile, _compile_lock)
        installed.append("Triton compilation")
    for module in (triton_compiler_module, triton_compiler):   # kernels compiled outside JITFunction._do_compile
        function = getattr(module, "compile", None)
        if callable(function) and not getattr(function, "_wangp_mgpu", False):
            module.compile = _locked(function, _compile_lock)
            if "Triton compilation" not in installed:
                installed.append("Triton compilation")
    try:
        from shared.kernels import triton_tuner
        tuner = getattr(triton_tuner, "LaunchTuner", None)
        config = getattr(tuner, "config", None)
        if config is not None and not getattr(config, "_wangp_mgpu", False):
            tuner.config = _locked(config, _tuning_lock)
            installed.append("WanGP Triton launch tuning")
    except Exception:
        pass
    return installed


def prepare_secondary(primary_index, secondary_index):
    """Make WanGP's INT8 kernels usable on ``secondary_index`` while ``primary_index`` works. Returns None when ready (or
    nothing to do), else the reason why INT8 layers cannot run on that GPU. Call it from the main thread, before any
    work on that GPU."""
    _prepare_triton()
    return _prepare_wgp(primary_index, secondary_index)


# ----------------------------------------------------------------------------------------------------------------------
# WanGP fix: shared INT8 projection with WGPKernels on non-ConvRot layers (WanGP v18.21)
# ----------------------------------------------------------------------------------------------------------------------

def install_wangp_fixes():
    """WanGP v18.21's int8_backend.linear_multi (one input shared by several projections, e.g. LTX-2's k/v) reads
    ``modules[0]._convrot_group_size`` directly when WGPKernels is the INT8 backend, so a plain quanto INT8 checkpoint
    (no ConvRot, such as ltx-2.3-22b-distilled_..._quanto_int8) fails with "'QLinearQuantoRouter' object has no
    attribute '_convrot_group_size'", with or without this plugin. Only in that case, each projection now runs on its own
    (the path WanGP takes for projections it cannot share): same result. Installed whatever the plugin's switches, since
    the failure does not depend on them; nothing changes in any other case."""
    try:
        from shared.kernels import int8_backend
    except Exception:
        return False
    original = getattr(int8_backend, "linear_multi", None)
    if original is None or getattr(original, "_wangp_mgpu", False):
        return False
    try:   # fixed in WanGP (v18.81 reads the attribute with getattr): nothing to do
        import inspect
        if "modules[0]._convrot_group_size" not in inspect.getsource(original):
            return False
    except (OSError, TypeError):
        pass

    @functools.wraps(original)
    def linear_multi(modules, x):
        if getattr(int8_backend, "_backend", None) == "wgpkernels" and any(
                not hasattr(module, "_convrot_group_size") for module in modules):
            return [module(x) for module in modules]
        return original(modules, x)
    linear_multi._wangp_mgpu = True
    int8_backend.linear_multi = linear_multi
    return True
