"""Persistent settings of the Multi-GPU plugin (settings.json next to this file)."""
import json
import os
import threading

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
_DEFAULTS = {
    "enabled": False,          # master switch
    "vae_enabled": True,       # H3 VAE decoding split
    "share_cuda0": 50,         # % of the VAE tiles decoded on cuda:0
    "te_enabled": False,       # H3 text encoder: last layers kept on the secondary GPU (skipped while denoising is split)
    "te_layers_cuda1": 20,     # number of text encoder layers (out of 50) kept on the secondary GPU
    "denoise_enabled": False,  # H3 denoising split (experimental)
    "denoise_share_cuda0": 50, # % of the token rows denoised on cuda:0
}
_lock = threading.Lock()
_values = dict(_DEFAULTS)


def _load():
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, default in _DEFAULTS.items():
            if key in data and isinstance(data[key], type(default)):
                _values[key] = data[key]
    except FileNotFoundError:
        pass
    except Exception as exc:  # a broken file must never prevent WanGP from starting
        print(f"[Multi-GPU] Could not read {_PATH}: {exc}. Using defaults.")


def get():
    with _lock:
        return dict(_values)


def update(**changes):
    with _lock:
        for key, value in changes.items():
            if key in _DEFAULTS:
                _values[key] = type(_DEFAULTS[key])(value)
        snapshot = dict(_values)
    try:
        with open(_PATH, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
    except Exception as exc:
        print(f"[Multi-GPU] Could not save {_PATH}: {exc}")
    return snapshot


_load()
