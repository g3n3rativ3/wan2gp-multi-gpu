"""CPU checks of kernel_compat's process-wide fixes: Triton compilations and WanGP's Triton launch tuning run one at a
time across threads (two GPUs compiling the same kernel at once broke Triton's file cache on Windows), and the WanGP
v18.21 INT8 guard is only installed on a WanGP that needs it.

    python test_kernel_compat.py <WanGP folder> <plugins folder>
"""
import importlib
import os
import sys
import threading
import time

WANGP = sys.argv[1]
PLUGINS = sys.argv[2]
sys.path.insert(0, WANGP)
sys.path.insert(0, PLUGINS)
PLUGIN = os.environ.get("MULTIGPU_PLUGIN", "wan2gp-multi-gpu")

from triton.runtime import jit as triton_jit  # noqa: E402

state = {"inside": 0, "most": 0, "calls": 0}
guard = threading.Lock()


def slow(*args, **kwargs):
    with guard:
        state["inside"] += 1
        state["calls"] += 1
        state["most"] = max(state["most"], state["inside"])
    time.sleep(0.2)
    with guard:
        state["inside"] -= 1
    return "kernel"


triton_jit.JITFunction._do_compile = lambda self, *args, **kwargs: slow()   # stand-in for a real compilation (needs a GPU)
kc = importlib.import_module(f"{PLUGIN}.kernel_compat")
installed = kc.install_triton_locks()
again = kc.install_triton_locks()
all_ok = True


def both(call):
    state.update(inside=0, most=0, calls=0)
    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return state["most"], state["calls"]


most, calls = both(lambda: triton_jit.JITFunction._do_compile(object(), "key"))
ok = "Triton compilation" in installed and most == 1 and calls == 2 and not again
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} Triton compilations one at a time: {calls} calls, at most {most} at once ({installed})")

has_tuner = os.path.exists(os.path.join(WANGP, "shared", "kernels", "triton_tuner.py"))
if has_tuner:
    from shared.kernels import triton_tuner
    original = triton_tuner.LaunchTuner.config._wangp_mgpu_original

    def fake_config(self, *args, **kwargs):
        return slow()
    triton_tuner.LaunchTuner.config = kc._locked(fake_config, kc._tuning_lock)
    tuner = triton_tuner.LaunchTuner("test", 1, "test")
    most, calls = both(lambda: tuner.config("k", "id", None, [(1,)], None, True))
    ok = "WanGP Triton launch tuning" in installed and most == 1 and calls == 2 and original is not None
    all_ok &= ok
    print(f"{'OK  ' if ok else 'FAIL'} WanGP Triton launch tuning one at a time: {calls} calls, at most {most} at once")
else:
    print("     (no shared/kernels/triton_tuner.py in this WanGP: launch tuner check skipped)")

from shared.kernels import int8_backend  # noqa: E402
import inspect  # noqa: E402
buggy = "modules[0]._convrot_group_size" in inspect.getsource(int8_backend.linear_multi)
fixed = kc.install_wangp_fixes()
ok = fixed == buggy
all_ok &= ok
print(f"{'OK  ' if ok else 'FAIL'} WanGP INT8 guard installed only where needed (WanGP needs it: {buggy}, installed: {fixed})")
print("ALL OK" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
