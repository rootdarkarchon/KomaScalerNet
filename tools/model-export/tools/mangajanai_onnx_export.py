#!/usr/bin/env python3
"""Export the seven official MangaJaNai V1 2x models to validated ONNX.

This is a one-time conversion/validation tool, not a production dependency.
It deliberately pins the reference architecture loader to Spandrel 0.4.1.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import onnxruntime as ort
from onnxconverter_common import float16
from spandrel import ModelLoader
from torch import Tensor, nn
from torch.nn import functional as F


OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAME = "output"
PARAMETER_COUNT = 16_703_171
# Slightly below the smallest positive IEEE-754 binary16 subnormal. This tells
# the converter not to clamp valid half-precision values while avoiding its
# strict less-than warning at exactly 2^-24.
FP16_MIN_POSITIVE_PRESERVE = 5.0e-08
FP16_MAX_FINITE = 65_504.0


@dataclass(frozen=True)
class OfficialModel:
    filename: str
    nominal_height: int
    minimum_height: int
    maximum_height: int | None
    source_sha256: str


OFFICIAL_MODELS = (
    OfficialModel(
        "2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth",
        1200,
        0,
        1250,
        "43b784f674bdbf89886a62a64cd5f8d8df92caf4d861bdf4d47dad249ede0267",
    ),
    OfficialModel(
        "2x_MangaJaNai_1300p_V1_ESRGAN_75k.pth",
        1300,
        1251,
        1350,
        "15ca3c0f75f97f7bf52065bf7c9b8d602de94ce9e3b078ac58793855eed18589",
    ),
    OfficialModel(
        "2x_MangaJaNai_1400p_V1_ESRGAN_70k.pth",
        1400,
        1351,
        1450,
        "a940ad8ebcf6bea5580f2f59df67deb009f054c9b87dbbc58c2e452722f34858",
    ),
    OfficialModel(
        "2x_MangaJaNai_1500p_V1_ESRGAN_90k.pth",
        1500,
        1451,
        1550,
        "d91f2d247fa61144c1634a2ba46926acd3956ae90d281a5bed6655f8364a5b2c",
    ),
    OfficialModel(
        "2x_MangaJaNai_1600p_V1_ESRGAN_90k.pth",
        1600,
        1551,
        1760,
        "6f5923f812dbc5d6aeed727635a21e74cacddce595afe6135cbd95078f6eee44",
    ),
    OfficialModel(
        "2x_MangaJaNai_1920p_V1_ESRGAN_70k.pth",
        1920,
        1761,
        1984,
        "1ad4aa6f64684baa430da1bb472489bff2a02473b14859015884a3852339c005",
    ),
    OfficialModel(
        "2x_MangaJaNai_2048p_V1_ESRGAN_95k.pth",
        2048,
        1985,
        None,
        "146cd009b9589203a8444fe0aa7195709bb5b9fdeaca3808b7fbbd5538f94c41",
    ),
)
OFFICIAL_BY_NAME = {item.filename: item for item in OFFICIAL_MODELS}


class ExactSpandrelWrapper(nn.Module):
    """Reproduce ImageModelDescriptor padding, RRDB inference, clamp, and crop.

    The exported contract accepts one RGB NCHW image with H,W >= 4. It pads
    right/bottom to a multiple of four using reflection, performs the model's
    factor-two pixel unshuffle and RRDB body, clamps to [0,1], and crops back
    to exactly 2H x 2W.
    """

    def __init__(self, rrdb_body: nn.Module) -> None:
        super().__init__()
        self.rrdb_body = rrdb_body

    def forward(self, x: Tensor) -> Tensor:
        height = x.shape[2]
        width = x.shape[3]
        pad_height = (4 - height % 4) % 4
        pad_width = (4 - width % 4) % 4
        x = F.pad(x, (0, pad_width, 0, pad_height), mode="reflect")
        x = F.pixel_unshuffle(x, 2)
        x = self.rrdb_body(x)
        x = torch.clamp(x, 0.0, 1.0)
        return x[:, :, : height * 2, : width * 2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_shape(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)[xX](\d+)", value)
    if not match:
        raise argparse.ArgumentTypeError("shape must be HEIGHTxWIDTH, for example 31x33")
    height, width = (int(part) for part in match.groups())
    if height < 4 or width < 4:
        raise argparse.ArgumentTypeError("height and width must both be at least 4")
    return height, width


def collect_models(input_path: Path, allow_unknown: bool) -> list[tuple[Path, OfficialModel]]:
    paths = [input_path] if input_path.is_file() else sorted(input_path.glob("2x_MangaJaNai_*_V1_ESRGAN_*.pth"))
    if not paths:
        raise SystemExit(f"No official 2x V1 .pth files found under {input_path}")

    results: list[tuple[Path, OfficialModel]] = []
    for path in paths:
        official = OFFICIAL_BY_NAME.get(path.name)
        if official is None:
            if not allow_unknown:
                raise SystemExit(f"Unknown model filename: {path.name}; pass --allow-unknown to override")
            match = re.search(r"_(\d+)p_", path.name)
            if not match:
                raise SystemExit(f"Cannot determine nominal height from {path.name}")
            height = int(match.group(1))
            official = OfficialModel(path.name, height, height, height, sha256_file(path))
        actual_sha256 = sha256_file(path)
        if actual_sha256 != official.source_sha256:
            raise SystemExit(
                f"SHA-256 mismatch for {path.name}: expected {official.source_sha256}, got {actual_sha256}"
            )
        results.append((path, official))
    return results


def load_reference(path: Path) -> Any:
    started = time.perf_counter()
    descriptor = ModelLoader(device="cpu").load_from_file(path)
    elapsed = time.perf_counter() - started

    parameters = sum(parameter.numel() for parameter in descriptor.model.parameters())
    requirements = descriptor.size_requirements
    if descriptor.scale != 2 or descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise RuntimeError(
            f"Unexpected descriptor for {path.name}: scale={descriptor.scale}, "
            f"input={descriptor.input_channels}, output={descriptor.output_channels}"
        )
    if parameters != PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected parameter count for {path.name}: {parameters}")
    if requirements.minimum != 4 or requirements.multiple_of != 4:
        raise RuntimeError(f"Unexpected size requirements for {path.name}: {requirements}")
    if descriptor.model.__class__.__name__ != "RRDBNet":
        raise RuntimeError(f"Unexpected architecture for {path.name}: {type(descriptor.model).__name__}")

    descriptor.model.eval().float()
    return descriptor, elapsed


def add_model_properties(model: onnx.ModelProto, *, precision: str, source: OfficialModel) -> None:
    onnx.helper.set_model_props(
        model,
        {
            "mangajanai.source_filename": source.filename,
            "mangajanai.source_sha256": source.source_sha256,
            "mangajanai.nominal_height": str(source.nominal_height),
            "mangajanai.architecture": "ESRGAN-RRDBNet-64nf-23nb-pixel-unshuffle-2",
            "mangajanai.scale": "2",
            "mangajanai.input_layout": "NCHW",
            "mangajanai.input_range": "0..1",
            "mangajanai.precision": precision,
            "mangajanai.wrapper": "reflect-pad-to-4;pixel-unshuffle-2;rrdb;clamp-0-1;crop-2x",
            "mangajanai.license": "CC-BY-NC-4.0",
        },
    )


def export_fp32(wrapper: nn.Module, sample: Tensor, target: Path, source: OfficialModel, opset: int) -> float:
    started = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            sample,
            target,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {2: "height", 3: "width"},
                OUTPUT_NAME: {2: "height_x2", 3: "width_x2"},
            },
        )

    model = onnx.load(target)
    add_model_properties(model, precision="fp32", source=source)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return time.perf_counter() - started


def convert_fp16(source_path: Path, target: Path, source: OfficialModel) -> float:
    started = time.perf_counter()
    model = onnx.load(source_path)
    model = float16.convert_float_to_float16(
        model,
        min_positive_val=FP16_MIN_POSITIVE_PRESERVE,
        max_finite_val=FP16_MAX_FINITE,
        keep_io_types=False,
        disable_shape_infer=False,
    )
    add_model_properties(model, precision="fp16", source=source)
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return time.perf_counter() - started


def inspect_onnx(path: Path) -> dict[str, Any]:
    model = onnx.load(path)
    operators = collections.Counter(node.op_type for node in model.graph.node)
    domains = sorted({node.domain for node in model.graph.node})
    if domains != [""]:
        raise RuntimeError(f"Non-standard ONNX domains in {path.name}: {domains}")
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "sizeBytes": path.stat().st_size,
        "irVersion": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "nodeCount": len(model.graph.node),
        "operators": dict(sorted(operators.items())),
        "input": {"name": model.graph.input[0].name, "elementType": model.graph.input[0].type.tensor_type.elem_type},
        "output": {
            "name": model.graph.output[0].name,
            "elementType": model.graph.output[0].type.tensor_type.elem_type,
        },
    }


def make_session(path: Path, provider: str) -> tuple[ort.InferenceSession, float, str]:
    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider is unavailable; providers={available}")
        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    elif provider == "auto" and "CUDAExecutionProvider" in available:
        providers = [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 4
    started = time.perf_counter()
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    elapsed = time.perf_counter() - started
    return session, elapsed, session.get_providers()[0]


def validate_model(
    descriptor: Any,
    onnx_path: Path,
    precision: str,
    shapes: list[tuple[int, int]],
    provider: str,
    seed_base: int,
) -> dict[str, Any]:
    session, session_load_seconds, actual_provider = make_session(onnx_path, provider)
    use_cuda_reference = precision == "fp16" and actual_provider == "CUDAExecutionProvider"
    if use_cuda_reference:
        descriptor.model.to("cuda").half().eval()

    results: list[dict[str, Any]] = []
    for index, (height, width) in enumerate(shapes):
        rng = np.random.default_rng(seed_base + index)
        input_fp32 = rng.random((1, 3, height, width), dtype=np.float32)
        reference_tensor = torch.from_numpy(input_fp32)
        if use_cuda_reference:
            reference_tensor = reference_tensor.to("cuda").half()

        with torch.inference_mode():
            reference = descriptor(reference_tensor).float().cpu().numpy()

        ort_input = input_fp32.astype(np.float16) if precision == "fp16" else input_fp32
        started = time.perf_counter()
        actual = session.run([OUTPUT_NAME], {INPUT_NAME: ort_input})[0].astype(np.float32)
        inference_seconds = time.perf_counter() - started
        difference = np.abs(reference - actual)
        result = {
            "inputShape": [1, 3, height, width],
            "outputShape": list(actual.shape),
            "maxAbsoluteError": float(difference.max()),
            "meanAbsoluteError": float(difference.mean()),
            "p999AbsoluteError": float(np.quantile(difference, 0.999)),
            "inferenceSeconds": inference_seconds,
        }
        expected_shape = (1, 3, height * 2, width * 2)
        if tuple(actual.shape) != expected_shape:
            raise RuntimeError(f"Unexpected output shape for {onnx_path.name}: {actual.shape} != {expected_shape}")
        if precision == "fp32":
            passed = result["maxAbsoluteError"] <= 1e-4 and result["meanAbsoluteError"] <= 1e-5
        else:
            passed = result["maxAbsoluteError"] <= 5e-3 and result["meanAbsoluteError"] <= 1e-3
        result["passed"] = passed
        if not passed:
            raise RuntimeError(f"Parity failed for {onnx_path.name} at {height}x{width}: {result}")
        results.append(result)

    if use_cuda_reference:
        descriptor.model.to("cpu").float()
        torch.cuda.empty_cache()
    return {
        "provider": actual_provider,
        "reference": "PyTorch FP16 CUDA" if use_cuda_reference else "PyTorch FP32 CPU",
        "sessionLoadSeconds": session_load_seconds,
        "cases": results,
    }


def export_one(
    source_path: Path,
    source: OfficialModel,
    output_dir: Path,
    precision: str,
    shapes: list[tuple[int, int]],
    provider: str,
    opset: int,
    skip_validation: bool,
) -> dict[str, Any]:
    print(f"Loading {source_path.name}", flush=True)
    descriptor, pytorch_load_seconds = load_reference(source_path)
    # RRDBNet.forward adds its own factor-two pixel-unshuffle wrapper. We use
    # the sequential RRDB body directly because ExactSpandrelWrapper performs
    # the descriptor's stricter pad-to-four behavior and pixel unshuffle.
    wrapper = ExactSpandrelWrapper(descriptor.model.model).eval().float()

    sample_height, sample_width = shapes[0]
    sample = torch.rand(1, 3, sample_height, sample_width, dtype=torch.float32)
    stem = source_path.stem
    fp32_path = output_dir / f"{stem}.fp32.onnx"
    fp16_path = output_dir / f"{stem}.fp16.onnx"

    print(f"Exporting {fp32_path.name}", flush=True)
    fp32_export_seconds = export_fp32(wrapper, sample, fp32_path, source, opset)
    artifacts: dict[str, Any] = {}
    if precision in {"fp32", "both"}:
        artifacts["fp32"] = inspect_onnx(fp32_path)
        if not skip_validation:
            artifacts["fp32"]["validation"] = validate_model(
                descriptor, fp32_path, "fp32", shapes, provider, source.nominal_height * 100
            )

    if precision in {"fp16", "both"}:
        print(f"Converting {fp16_path.name}", flush=True)
        fp16_conversion_seconds = convert_fp16(fp32_path, fp16_path, source)
        artifacts["fp16"] = inspect_onnx(fp16_path)
        artifacts["fp16"]["conversionSeconds"] = fp16_conversion_seconds
        if not skip_validation:
            artifacts["fp16"]["validation"] = validate_model(
                descriptor, fp16_path, "fp16", shapes, provider, source.nominal_height * 100
            )
        if precision == "fp16":
            # This exact path was created above solely as a conversion intermediate.
            fp32_path.unlink(missing_ok=True)

    return {
        "nominalHeight": source.nominal_height,
        "selection": {"minimumHeight": source.minimum_height, "maximumHeight": source.maximum_height},
        "source": {
            "file": source_path.name,
            "sha256": source.source_sha256,
            "sizeBytes": source_path.stat().st_size,
        },
        "pytorchLoadSeconds": pytorch_load_seconds,
        "fp32ExportSeconds": fp32_export_seconds,
        "artifacts": artifacts,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Official .pth file or directory containing all seven files")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--precision", choices=("fp32", "fp16", "both"), default="both")
    parser.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--opset", type=int, default=OPSET)
    parser.add_argument(
        "--validation-shape",
        action="append",
        type=parse_shape,
        dest="validation_shapes",
        help="Repeatable HEIGHTxWIDTH; defaults exercise aligned, odd, and non-multiple-of-four inputs",
    )
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--allow-unknown", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.opset < 17:
        raise SystemExit("Opset 17 or newer is required")
    shapes = args.validation_shapes or [(17, 19), (16, 20), (31, 33)]
    args.output.mkdir(parents=True, exist_ok=True)
    models = collect_models(args.input, args.allow_unknown)

    started = time.perf_counter()
    results = [
        export_one(
            source_path,
            source,
            args.output,
            args.precision,
            shapes,
            args.provider,
            args.opset,
            args.skip_validation,
        )
        for source_path, source in models
    ]

    metadata = {
        "schemaVersion": 1,
        "createdBy": "mangajanai_onnx_export.py",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxRuntime": ort.__version__,
            "spandrel": "0.4.1",
            "availableOnnxRuntimeProviders": ort.get_available_providers(),
        },
        "graphContract": {
            "opset": args.opset,
            "input": {"name": INPUT_NAME, "shape": [1, 3, "height", "width"], "range": [0.0, 1.0]},
            "output": {"name": OUTPUT_NAME, "shape": [1, 3, "height*2", "width*2"], "range": [0.0, 1.0]},
            "minimumHeight": 4,
            "minimumWidth": 4,
            "dynamicSpatialDimensions": True,
            "padding": "reflect right/bottom to multiple of 4 inside graph; crop to exact 2x output",
            "layout": "NCHW",
        },
        "license": {
            "model": "CC BY-NC 4.0",
            "source": "https://github.com/the-database/MangaJaNai",
            "modified": "Converted from official PyTorch state dictionaries to ONNX",
        },
        "models": results,
        "totalSeconds": time.perf_counter() - started,
    }
    metadata_path = args.output / "models.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
