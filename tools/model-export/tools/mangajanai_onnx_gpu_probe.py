#!/usr/bin/env python3
"""Measure seven-session ONNX Runtime CUDA load time, VRAM, and warm latency."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pynvml
import torch
import onnxruntime as ort


T = TypeVar("T")
MIB = 1024 * 1024


def as_text(value: str | bytes) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def parse_shape(value: str) -> tuple[int, int]:
    try:
        height, width = (int(part) for part in value.lower().split("x", maxsplit=1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("tile must be HEIGHTxWIDTH") from error
    if height < 4 or width < 4:
        raise argparse.ArgumentTypeError("tile dimensions must both be at least 4")
    return height, width


class MemorySampler:
    def __init__(self, handle: Any, interval_seconds: float = 0.01) -> None:
        self.handle = handle
        self.interval_seconds = interval_seconds

    def used_bytes(self) -> int:
        return int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)

    def measure(self, action: Callable[[], T]) -> tuple[T, float, int, int, int]:
        before = self.used_bytes()
        peak = before
        stop = threading.Event()

        def sample() -> None:
            nonlocal peak
            while not stop.wait(self.interval_seconds):
                peak = max(peak, self.used_bytes())

        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        started = time.perf_counter()
        try:
            result = action()
        finally:
            elapsed = time.perf_counter() - started
            stop.set()
            thread.join()
        after = self.used_bytes()
        peak = max(peak, after)
        return result, elapsed, before, after, peak


def memory_record(elapsed: float, before: int, after: int, peak: int) -> dict[str, Any]:
    return {
        "seconds": elapsed,
        "beforeMiB": before / MIB,
        "afterMiB": after / MIB,
        "peakMiB": peak / MIB,
        "residentDeltaMiB": (after - before) / MIB,
        "peakDeltaMiB": (peak - before) / MIB,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, help="Directory containing seven precision-qualified ONNX files")
    parser.add_argument("--output", type=Path, default=Path("gpu-probe.json"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--tile", type=parse_shape, default=(256, 256))
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--cudnn-algo", choices=("HEURISTIC", "EXHAUSTIVE", "DEFAULT"), default="HEURISTIC")
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--timed-runs", type=int, default=5)
    args = parser.parse_args()

    if args.warmup_runs < 0 or args.timed_runs < 1:
        raise SystemExit("--warmup-runs must be nonnegative and --timed-runs must be at least one")

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise SystemExit(f"CUDAExecutionProvider is unavailable: {ort.get_available_providers()}")
    paths = sorted(args.models.glob(f"2x_MangaJaNai_*_V1_ESRGAN_*.{args.precision}.onnx"))
    if len(paths) != 7:
        raise SystemExit(f"Expected seven {args.precision.upper()} models, found {len(paths)} under {args.models}")

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(args.device_id)
    sampler = MemorySampler(handle)
    initial_memory = sampler.used_bytes()
    provider_options = {
        "device_id": str(args.device_id),
        "cudnn_conv_algo_search": args.cudnn_algo,
        "do_copy_in_default_stream": "1",
        # Keep the FP32 probe numerically comparable with the strict reference.
        # FP16 tensor-core execution remains enabled because TF32 applies to FP32.
        "use_tf32": "0" if args.precision == "fp32" else "1",
    }
    providers = [("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 4
    sessions: list[tuple[Path, ort.InferenceSession]] = []
    loads: list[dict[str, Any]] = []
    try:
        for path in paths:
            session, elapsed, before, after, peak = sampler.measure(
                lambda path=path: ort.InferenceSession(str(path), sess_options=options, providers=providers)
            )
            if session.get_providers()[0] != "CUDAExecutionProvider":
                raise RuntimeError(f"{path.name} did not activate CUDAExecutionProvider: {session.get_providers()}")
            sessions.append((path, session))
            loads.append({"model": path.name, **memory_record(elapsed, before, after, peak)})
            print(json.dumps(loads[-1]), flush=True)

        after_session_load_memory = sampler.used_bytes()
        height, width = args.tile
        rng = np.random.default_rng(20260720)
        input_dtype = np.float16 if args.precision == "fp16" else np.float32
        input_tensor = rng.random((1, 3, height, width), dtype=np.float32).astype(input_dtype)
        runs: list[dict[str, Any]] = []
        for path, session in sessions:
            output_holder: list[np.ndarray] = []

            def infer() -> None:
                output_holder.extend(session.run(["output"], {"input": input_tensor}))

            _, elapsed, before, after, peak = sampler.measure(infer)
            output = output_holder[0]
            expected_shape = (1, 3, height * 2, width * 2)
            if output.shape != expected_shape or output.dtype != input_dtype or not np.isfinite(output).all():
                raise RuntimeError(
                    f"Invalid output from {path.name}: shape={output.shape}, dtype={output.dtype}, "
                    f"finite={np.isfinite(output).all()}"
                )

            for _ in range(args.warmup_runs):
                session.run(["output"], {"input": input_tensor})

            warm_runs: list[dict[str, Any]] = []
            for _ in range(args.timed_runs):
                _, warm_elapsed, warm_before, warm_after, warm_peak = sampler.measure(
                    lambda: session.run(["output"], {"input": input_tensor})
                )
                warm_runs.append(memory_record(warm_elapsed, warm_before, warm_after, warm_peak))
            warm_seconds = [item["seconds"] for item in warm_runs]
            runs.append(
                {
                    "model": path.name,
                    "inputShape": list(input_tensor.shape),
                    "outputShape": list(output.shape),
                    "coldShapeInitialization": memory_record(elapsed, before, after, peak),
                    "warmupRuns": args.warmup_runs,
                    "timedRuns": args.timed_runs,
                    "warmSeconds": warm_seconds,
                    "warmMeanSeconds": statistics.fmean(warm_seconds),
                    "warmMedianSeconds": statistics.median(warm_seconds),
                    "warmMinimumSeconds": min(warm_seconds),
                    "warmMaximumSeconds": max(warm_seconds),
                    "warmRuns": warm_runs,
                }
            )
            print(json.dumps(runs[-1]), flush=True)

        final_memory = sampler.used_bytes()
        result = {
            "schemaVersion": 1,
            "pid": os.getpid(),
            "environment": {
                "python": platform.python_version(),
                "onnxRuntime": ort.__version__,
                "torch": torch.__version__,
                "torchCuda": torch.version.cuda,
                "torchCudnn": torch.backends.cudnn.version(),
                "providers": ort.get_available_providers(),
                "gpu": as_text(pynvml.nvmlDeviceGetName(handle)),
                "driver": as_text(pynvml.nvmlSystemGetDriverVersion()),
                "deviceId": args.device_id,
                "precision": args.precision,
                "cudnnAlgorithmSearch": args.cudnn_algo,
                "tf32Allowed": args.precision != "fp32",
            },
            "initialUsedMiB": initial_memory / MIB,
            "afterSessionLoadUsedMiB": after_session_load_memory / MIB,
            "sessionLoadResidentDeltaMiB": (after_session_load_memory - initial_memory) / MIB,
            "afterAllSessionsUsedMiB": final_memory / MIB,
            "allSessionsResidentDeltaMiB": (final_memory - initial_memory) / MIB,
            "loads": loads,
            "serializedRuns": runs,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", flush=True)
    finally:
        pynvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
