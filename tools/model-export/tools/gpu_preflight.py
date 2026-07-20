#!/usr/bin/env python3
"""Fail fast unless PyTorch and ONNX Runtime can both use the selected CUDA GPU."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch cannot access CUDA. Check the NVIDIA driver and CUDA-enabled PyTorch install.")
    if args.device_id < 0 or args.device_id >= torch.cuda.device_count():
        raise SystemExit(f"Invalid CUDA device {args.device_id}; device count={torch.cuda.device_count()}")

    torch.cuda.set_device(args.device_id)
    # Force CUDA and cuDNN libraries to load before importing ONNX Runtime.
    sample = torch.rand(1, 3, 16, 16, device="cuda", dtype=torch.float16)
    weight = torch.rand(4, 3, 3, 3, device="cuda", dtype=torch.float16)
    torch.nn.functional.conv2d(sample, weight, padding=1)
    torch.cuda.synchronize()

    import onnxruntime as ort
    import onnx
    import onnxconverter_common
    import pynvml

    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise SystemExit(f"ONNX Runtime CUDAExecutionProvider is unavailable: {providers}")

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.device_id)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    finally:
        pynvml.nvmlShutdown()

    def text(value: str | bytes) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    result = {
        "passed": True,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchCuda": torch.version.cuda,
        "torchCudnn": torch.backends.cudnn.version(),
        "onnxRuntime": ort.__version__,
        "onnx": onnx.__version__,
        "onnxConverterCommon": version("onnxconverter-common"),
        "onnxRuntimeProviders": providers,
        "deviceId": args.device_id,
        "gpu": text(gpu_name),
        "computeCapability": list(torch.cuda.get_device_capability(args.device_id)),
        "driver": text(driver),
        "totalMemoryMiB": memory.total / (1024 * 1024),
        "freeMemoryMiBAtPreflight": memory.free / (1024 * 1024),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
