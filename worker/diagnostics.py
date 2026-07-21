"""Probe NC engine availability without running a full benchmark."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any


def _check_root(worker_root: str | None) -> dict[str, Any]:
    root = (worker_root or os.getenv("LIVEKIT_WORKER_ROOT") or "").strip()
    if not root:
        return {
            "ok": False,
            "path": "",
            "error": "LIVEKIT_WORKER_ROOT is empty",
        }
    path = Path(root).resolve()
    src = path / "src"
    if not src.is_dir():
        return {
            "ok": False,
            "path": str(path),
            "error": f"Missing src/ under {path}",
        }
    return {
        "ok": True,
        "path": str(path),
        "src": str(src),
        "hush_lib_darwin": (path / "sdk/hush/lib/libweya_nc.dylib").exists(),
        "hush_lib_linux": (path / "sdk/hush/lib/libweya_nc.so").exists(),
        "hush_model": any(
            [
                (
                    path
                    / "sdk/hush/models/onnx/advanced_dfnet16k_model_best_onnx.tar.gz"
                ).exists(),
                (
                    path / "sdk/hush/models/advanced_dfnet16k_model_best_onnx.tar.gz"
                ).exists(),
            ]
        ),
        "hecttor_wheels": sorted(
            p.name for p in (path / "wheels").glob("hecttor_sdk-*.whl")
        )
        if (path / "wheels").is_dir()
        else [],
    }


def _probe_build(engine: str, model: str | None, worker_root: str) -> dict[str, Any]:
    from worker.nc_engines import build_nc_processor

    try:
        processor = build_nc_processor(
            engine,
            model,
            strength=0.5,
            worker_root=worker_root,
        )
        return {
            "ok": True,
            "engine": engine,
            "model": model,
            "processor": type(processor).__name__ if processor else "none",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "engine": engine,
            "model": model,
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_tier_a(worker_root: str | None = None) -> dict[str, Any]:
    """Return a machine-readable Tier A readiness report."""
    root_info = _check_root(worker_root)
    python = {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    hecttor_import: dict[str, Any]
    try:
        import hecttor_sdk  # noqa: F401

        hecttor_import = {"ok": True, "module": getattr(hecttor_sdk, "__file__", None)}
    except Exception as exc:  # noqa: BLE001
        hecttor_import = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    api_key_set = bool(os.getenv("HECTTOR_API_KEY", "").strip())

    engines: dict[str, Any] = {}
    if root_info.get("ok"):
        engines["dtln"] = _probe_build("dtln", None, root_info["path"])
        engines["hush"] = _probe_build("hush", None, root_info["path"])
        engines["hecttor"] = _probe_build("hecttor", "coda-1.0", root_info["path"])
    else:
        for name in ("dtln", "hush", "hecttor"):
            engines[name] = {
                "ok": False,
                "engine": name,
                "error": root_info.get("error", "LIVEKIT_WORKER_ROOT invalid"),
            }

    failed = [
        name
        for name, result in engines.items()
        if not result.get("ok")
    ]
    hints: list[str] = []
    if not root_info.get("ok"):
        hints.append("Set LIVEKIT_WORKER_ROOT in NC_WER_Analysis/.env to the absolute worker path.")
    if not hecttor_import.get("ok"):
        hints.append(
            "Install hecttor into this venv: "
            "make install-nc-wheels  (needs Python 3.13 + wheel under livekit-agent-worker/wheels)."
        )
    if not api_key_set:
        hints.append("Set HECTTOR_API_KEY in NC_WER_Analysis/.env (not only in the worker .env).")
    hush = engines.get("hush") or {}
    if not hush.get("ok") and root_info.get("ok"):
        system = platform.system()
        if system == "Linux" and not root_info.get("hush_lib_linux"):
            hints.append(
                "Hush needs libweya_nc.so under livekit-agent-worker/sdk/hush/lib/ "
                "(repo currently ships macOS .dylib only)."
            )
        elif system == "Darwin" and not root_info.get("hush_lib_darwin"):
            hints.append("Missing sdk/hush/lib/libweya_nc.dylib in LIVEKIT_WORKER_ROOT.")
        elif not root_info.get("hush_model"):
            hints.append("Missing Hush ONNX model under sdk/hush/models/.")
        else:
            hints.append(f"Hush probe failed: {hush.get('error')}")

    ok = root_info.get("ok") and not failed
    message = (
        "Tier A engines ready"
        if ok
        else "Tier A not ready — " + "; ".join(failed) + " failed. " + " ".join(hints)
    )

    return {
        "ok": ok,
        "message": message,
        "python": python,
        "livekit_worker_root": root_info,
        "hecttor_sdk_import": hecttor_import,
        "hecttor_api_key_set": api_key_set,
        "engines": engines,
        "failed_engines": failed,
        "hints": hints,
    }
