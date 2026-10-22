"""A device context like WanGP's `with torch.device("cuda")`, which also reports any tensor allocation made by the
plugin's own code without an explicit device (under WanGP that allocation lands on the GPU: a "pinned" buffer then
fails with "Only dense CPU tensors can be pinned", the bug of v0.1.0)."""
import sys

from torch.utils._device import DeviceContext, _device_constructors

PLUGIN_MARK = "wan2gp-multi-gpu"
violations = []


class StrictDeviceContext(DeviceContext):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _device_constructors() and kwargs.get("device") is None:
            frame = sys._getframe(1)
            if PLUGIN_MARK in frame.f_code.co_filename and "tests" not in frame.f_code.co_filename:
                violations.append(f"{frame.f_code.co_filename}:{frame.f_lineno} {getattr(func, '__name__', func)}")
        return super().__torch_function__(func, types, args, kwargs)
