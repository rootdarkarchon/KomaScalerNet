#!/usr/bin/env python3
"""Compare MangaJaNai PyTorch reference inference with ONNX Runtime.

This is an offline validation tool, not a production dependency. It generates
deterministic manga-oriented fixtures, optionally adds crops from the official
MangaJaNai demo, evaluates numerical and visual metrics, writes per-case PNGs,
and emits a machine-readable validation-results.json plus a contact sheet.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps
from skimage.metrics import structural_similarity
from torch.nn import functional as F

from mangajanai_onnx_export import (
    INPUT_NAME,
    OFFICIAL_MODELS,
    OUTPUT_NAME,
    load_reference,
    sha256_file,
)


SCHEMA_VERSION = 1
DEMO_SHA256 = "455169da3c0efef7d5ae12d19978db3cd7df34c9db76c1decb6bb1d551c0f4fc"
DEFAULT_CATEGORY_MODEL_HEIGHT = 1400


@dataclass(frozen=True)
class Thresholds:
    max_absolute_error: float
    mean_absolute_error: float
    minimum_psnr_db: float
    minimum_ssim: float
    maximum_level_region_mae: float
    maximum_halftone_energy_relative_error: float


def thresholds_for_precision(precision: str) -> Thresholds:
    if precision == "fp32":
        return Thresholds(3e-4, 1e-5, 90.0, 0.99999, 2e-5, 1e-4)
    return Thresholds(0.005, 0.001, 50.0, 0.999, 0.0015, 0.02)


@dataclass
class Fixture:
    name: str
    category: str
    image: Image.Image
    notes: str
    inspect_levels: bool = False
    inspect_halftone: bool = False


def _rgb(gray: np.ndarray) -> Image.Image:
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return Image.fromarray(np.repeat(gray[:, :, None], 3, axis=2), mode="RGB")


def fixture_line_art() -> Fixture:
    image = Image.new("L", (79, 71), 255)
    draw = ImageDraw.Draw(image)
    for offset, width in ((5, 1), (11, 2), (19, 3), (29, 1)):
        draw.line((3, offset, 75, 70 - offset // 2), fill=0, width=width)
    draw.ellipse((14, 16, 58, 61), outline=0, width=2)
    draw.arc((23, 25, 49, 54), 20, 155, fill=36, width=1)
    draw.rectangle((4, 52, 27, 67), outline=0, width=1)
    return Fixture("high-contrast-line-art", "line-art", image.convert("RGB"), "Pure black lines at several widths.", True)


def fixture_screentone() -> Fixture:
    height, width = 73, 81
    yy, xx = np.mgrid[:height, :width]
    result = np.full((height, width), 255, dtype=np.uint8)
    left = xx < width // 2
    dots_a = ((xx % 4 == 0) & (yy % 4 == 0)) | ((xx % 4 == 2) & (yy % 4 == 2))
    dots_b = ((xx % 3 == 0) & (yy % 3 == 0))
    result[left & dots_a] = 25
    result[~left & dots_b] = 65
    result[(yy > 31) & (yy < 37)] = ((xx[(yy > 31) & (yy < 37)] % 2) * 150).astype(np.uint8)
    return Fixture("screentone-heavy", "halftone", _rgb(result), "Two dot frequencies plus a dense transition band.", True, True)


def fixture_gradient() -> Fixture:
    height, width = 67, 83
    yy, xx = np.mgrid[:height, :width]
    ramp = 255.0 * (0.65 * xx / (width - 1) + 0.35 * yy / (height - 1))
    wave = 10.0 * np.sin(xx / 5.0) * np.cos(yy / 7.0)
    return Fixture("grayscale-gradient", "gradient", _rgb(ramp + wave), "Smooth ramp with low-amplitude texture.", True)


def fixture_noisy_jpeg() -> Fixture:
    height, width = 76, 84
    yy, xx = np.mgrid[:height, :width]
    base = 225 - 90 * (np.sin(xx / 4.0) * np.sin(yy / 6.0) > 0.55)
    rng = np.random.default_rng(20260720)
    base = np.clip(base + rng.normal(0, 17, base.shape), 0, 255).astype(np.uint8)
    image = _rgb(base)
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=24, subsampling=2)
    encoded.seek(0)
    decoded = Image.open(encoded).convert("RGB")
    return Fixture("noisy-jpeg-scan", "jpeg", decoded, "Deterministic noise with JPEG quality 24 artifacts.")


def fixture_visually_monochrome_rgb() -> Fixture:
    height, width = 71, 77
    yy, xx = np.mgrid[:height, :width]
    base = np.clip(238 - 160 * (((xx // 7 + yy // 9) % 3) == 0), 0, 255).astype(np.int16)
    rgb = np.stack((base + 2, base, base - 2), axis=2)
    return Fixture(
        "visually-monochrome-rgb",
        "rgb-monochrome",
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB"),
        "RGB input with a four-level red/blue channel spread; normalized to luminance once for both runtimes.",
    )


def fixture_landscape_spread() -> Fixture:
    height, width = 65, 117
    image = Image.new("L", (width, height), 250)
    draw = ImageDraw.Draw(image)
    draw.rectangle((56, 0, 60, height), fill=255)
    for x in range(4, 54, 7):
        draw.line((x, 5, x + 19, 59), fill=20 + (x % 3) * 35, width=1)
    for radius in range(4, 29, 4):
        draw.arc((65, 4, 65 + radius * 2, 4 + radius * 2), 15, 285, fill=15, width=1)
    draw.rectangle((2, 2, 114, 62), outline=0, width=1)
    return Fixture("landscape-double-page", "double-page", image.convert("RGB"), "Odd-width landscape spread with a bright center gutter.", True)


def fixture_odd_page() -> Fixture:
    height, width = 101, 69
    yy, xx = np.mgrid[:height, :width]
    page = np.full((height, width), 247.0)
    page -= 175 * ((xx > 5) & (xx < 8))
    page -= 135 * ((yy % 13 == 0) & (xx > 12))
    page -= 80 * (((xx - width / 2) ** 2 + (yy - height / 3) ** 2) < 18**2)
    page += 8 * np.sin(xx * 0.9 + yy * 0.35)
    return Fixture("odd-complete-page", "complete-page", _rgb(page), "Portrait page surrogate with odd dimensions and mixed structures.", True, True)


def fixture_small_page() -> Fixture:
    height, width = 17, 19
    yy, xx = np.mgrid[:height, :width]
    gray = np.where(((xx // 2 + yy // 3) % 2) == 0, 245, 22)
    return Fixture("small-odd-image", "small", _rgb(gray), "Near-minimum odd dimensions.", True, True)


def fixture_mixed_tile() -> Fixture:
    height, width = 49, 53
    yy, xx = np.mgrid[:height, :width]
    gray = 225.0 - 135.0 * (((xx // 5 + yy // 7) % 2) == 0)
    gray -= 75.0 * (((xx - 27) ** 2 + (yy - 24) ** 2) < 11**2)
    gray += 17.0 * np.sin(xx / 2.8) * np.cos(yy / 3.5)
    return Fixture("all-model-mixed-tile", "all-model-smoke", _rgb(gray), "Shared deterministic parity fixture for every model.", True, True)


def synthetic_fixtures() -> list[Fixture]:
    return [
        fixture_line_art(),
        fixture_screentone(),
        fixture_gradient(),
        fixture_noisy_jpeg(),
        fixture_visually_monochrome_rgb(),
        fixture_landscape_spread(),
        fixture_odd_page(),
        fixture_small_page(),
    ]


def official_demo_fixtures(path: Path) -> list[Fixture]:
    actual_hash = sha256_file(path)
    if actual_hash != DEMO_SHA256:
        raise RuntimeError(f"Official demo SHA-256 mismatch: expected {DEMO_SHA256}, got {actual_hash}")
    with Image.open(path) as source:
        source.seek(0)
        frame = source.convert("RGB")
    if frame.size != (1660, 1400):
        raise RuntimeError(f"Unexpected official demo size: {frame.size}")

    crops = [
        ("demo-architecture-halftone", (180, 520, 276, 616), "Architecture, rails, and dot patterns."),
        ("demo-face-lines", (1080, 535, 1176, 631), "Fine facial and hair line art."),
        ("demo-dense-screentone", (930, 1000, 1026, 1096), "Dense vertical and dot screentones."),
        ("demo-jpeg-texture", (1200, 650, 1296, 746), "Blurred texture and compression residue."),
    ]
    fixtures = [
        Fixture(name, "official-demo-crop", frame.crop(box), notes, True, True)
        for name, box, notes in crops
    ]
    page_region = frame.crop((0, 466, 1660, 1400)).resize((113, 127), Image.Resampling.LANCZOS)
    fixtures.append(
        Fixture(
            "demo-composite-page",
            "official-demo-page",
            page_region,
            "Entire disabled-upscaling demo collage, downsampled only to keep CPU validation bounded.",
            True,
            True,
        )
    )
    return fixtures


def corrected_luminance(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
    # Production decision from Step 2: correct RGB coefficients instead of
    # preserving the legacy wrapper's accidental red/blue swap.
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]) / 255.0


def prepare_model_input(image: Image.Image) -> np.ndarray:
    luminance = np.clip(corrected_luminance(image), 0.0, 1.0).astype(np.float32)
    return np.repeat(luminance[None, None, :, :], 3, axis=1)


def provider_list(requested: str, cudnn_algo: str) -> list[Any]:
    available = ort.get_available_providers()
    if requested == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; providers={available}")
        return [
            (
                "CUDAExecutionProvider",
                {
                    "cudnn_conv_algo_search": cudnn_algo,
                    "do_copy_in_default_stream": "1",
                    "use_tf32": "0",
                },
            ),
            "CPUExecutionProvider",
        ]
    if requested == "auto" and "CUDAExecutionProvider" in available:
        return provider_list("cuda", cudnn_algo)
    return ["CPUExecutionProvider"]


def make_session(path: Path, provider: str, cudnn_algo: str) -> tuple[ort.InferenceSession, float]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # cuDNN 9 may log rejected candidate plans as errors even when it finds a
    # supported fallback plan. Exceptions still surface normally.
    options.log_severity_level = 4
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=provider_list(provider, cudnn_algo)
    )
    return session, time.perf_counter() - started


def run_reference(descriptor: Any, model_input: np.ndarray, precision: str, device: str) -> np.ndarray:
    tensor = torch.from_numpy(model_input)
    if precision == "fp32":
        if device == "cuda":
            tensor = tensor.to("cuda")
        with torch.inference_mode():
            return descriptor(tensor).float().cpu().numpy()

    if device == "cuda":
        tensor = tensor.to("cuda").half()
        with torch.inference_mode():
            return descriptor(tensor).float().cpu().numpy()

    # torch 2.3's CPU reflection_pad2d has no Half kernel. Padding has no
    # arithmetic effect on existing pixels, so perform only that indexing step
    # in float32, cast back to Half, then run the exact FP16 RRDB path on CPU.
    # On CUDA the normal descriptor path can be used directly in Half.
    tensor = tensor.half()
    height, width = tensor.shape[2:]
    pad_height = (4 - height % 4) % 4
    pad_width = (4 - width % 4) % 4
    tensor = F.pad(tensor.float(), (0, pad_width, 0, pad_height), mode="reflect").half()
    tensor = F.pixel_unshuffle(tensor, 2)
    with torch.inference_mode():
        result = descriptor.model.model(tensor)
        result = result.clamp(0.0, 1.0)
        return result[:, :, : height * 2, : width * 2].float().cpu().numpy()


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean(np.square(reference.astype(np.float64) - candidate.astype(np.float64))))
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    # NCHW batch size one -> HWC, with standard local Gaussian-window SSIM.
    ref = np.moveaxis(reference[0], 0, 2)
    got = np.moveaxis(candidate[0], 0, 2)
    return float(
        structural_similarity(
            ref,
            got,
            data_range=1.0,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def high_frequency_energy(image: np.ndarray) -> float:
    gray = image[0, 0]
    horizontal = np.abs(np.diff(gray, axis=1)).mean()
    vertical = np.abs(np.diff(gray, axis=0)).mean()
    return float((horizontal + vertical) / 2.0)


def level_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = reference[0, 0]
    got = candidate[0, 0]
    black_mask = ref <= (5.0 / 255.0)
    white_mask = ref >= (250.0 / 255.0)

    def region(mask: np.ndarray) -> dict[str, float | int | None]:
        if not mask.any():
            return {"pixels": 0, "referenceMean": None, "candidateMean": None, "meanAbsoluteError": None}
        return {
            "pixels": int(mask.sum()),
            "referenceMean": float(ref[mask].mean()),
            "candidateMean": float(got[mask].mean()),
            "meanAbsoluteError": float(np.abs(ref[mask] - got[mask]).mean()),
        }

    return {"black": region(black_mask), "white": region(white_mask)}


def save_visuals(
    case_dir: Path,
    fixture: Fixture,
    model_input: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, str]:
    case_dir.mkdir(parents=True, exist_ok=True)
    input_gray = np.clip(model_input[0, 0] * 255.0, 0, 255).round().astype(np.uint8)
    ref_gray = np.clip(reference[0, 0] * 255.0, 0, 255).round().astype(np.uint8)
    got_gray = np.clip(candidate[0, 0] * 255.0, 0, 255).round().astype(np.uint8)
    difference = np.abs(reference[0, 0] - candidate[0, 0])
    diff_gray = np.clip(difference * 32.0 * 255.0, 0, 255).round().astype(np.uint8)

    files = {
        "input": "input.png",
        "reference": "reference-pytorch.png",
        "candidate": "candidate-onnx.png",
        "differenceX32": "difference-x32.png",
    }
    Image.fromarray(input_gray, mode="L").save(case_dir / files["input"])
    Image.fromarray(ref_gray, mode="L").save(case_dir / files["reference"])
    Image.fromarray(got_gray, mode="L").save(case_dir / files["candidate"])
    Image.fromarray(diff_gray, mode="L").save(case_dir / files["differenceX32"])
    return files


def evaluate_case(
    fixture: Fixture,
    descriptor: Any,
    session: ort.InferenceSession,
    output_root: Path,
    model_height: int,
    thresholds: Thresholds,
    precision: str,
    reference_device: str,
    save_images: bool,
) -> dict[str, Any]:
    model_input = prepare_model_input(fixture.image)
    input_height, input_width = model_input.shape[2:]
    reference_started = time.perf_counter()
    reference = run_reference(descriptor, model_input, precision, reference_device)
    reference_seconds = time.perf_counter() - reference_started

    candidate_started = time.perf_counter()
    candidate_input = model_input.astype(np.float16, copy=False) if precision == "fp16" else model_input
    candidate = session.run(
        [OUTPUT_NAME], {INPUT_NAME: candidate_input}
    )[0].astype(np.float32)
    candidate_seconds = time.perf_counter() - candidate_started

    expected_shape = (1, 3, input_height * 2, input_width * 2)
    difference = np.abs(reference - candidate)
    shape_passed = tuple(reference.shape) == expected_shape and tuple(candidate.shape) == expected_shape
    finite_passed = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    range_passed = bool(candidate.min() >= 0.0 and candidate.max() <= 1.0)
    mae = float(difference.mean())
    max_error = float(difference.max())
    psnr_db = psnr(reference, candidate)
    ssim_value = ssim(reference, candidate)
    levels = level_metrics(reference, candidate) if fixture.inspect_levels else None
    level_errors = [] if levels is None else [
        region["meanAbsoluteError"]
        for region in levels.values()
        if region["meanAbsoluteError"] is not None
    ]
    levels_passed = not level_errors or max(level_errors) <= thresholds.maximum_level_region_mae

    halftone = None
    halftone_passed = True
    if fixture.inspect_halftone:
        ref_energy = high_frequency_energy(reference)
        got_energy = high_frequency_energy(candidate)
        relative_error = abs(ref_energy - got_energy) / max(ref_energy, 1e-12)
        halftone = {
            "referenceHighFrequencyEnergy": ref_energy,
            "candidateHighFrequencyEnergy": got_energy,
            "relativeError": relative_error,
        }
        halftone_passed = relative_error <= thresholds.maximum_halftone_energy_relative_error

    numerical_passed = (
        max_error <= thresholds.max_absolute_error
        and mae <= thresholds.mean_absolute_error
        and psnr_db >= thresholds.minimum_psnr_db
        and ssim_value >= thresholds.minimum_ssim
    )
    passed = shape_passed and finite_passed and range_passed and numerical_passed and levels_passed and halftone_passed

    case_dir = output_root / f"{model_height}p" / fixture.name
    visuals = save_visuals(case_dir, fixture, model_input, reference, candidate) if save_images else None
    return {
        "fixture": fixture.name,
        "category": fixture.category,
        "notes": fixture.notes,
        "modelHeight": model_height,
        "inputShape": list(model_input.shape),
        "referenceOutputShape": list(reference.shape),
        "candidateOutputShape": list(candidate.shape),
        "referenceRange": [float(reference.min()), float(reference.max())],
        "candidateRange": [float(candidate.min()), float(candidate.max())],
        "meanAbsoluteError": mae,
        "maximumAbsoluteError": max_error,
        "p999AbsoluteError": float(np.quantile(difference, 0.999)),
        "fractionAboveOneCodeValue": float(np.mean(difference > (1.0 / 255.0))),
        "fractionAboveFiveCodeValues": float(np.mean(difference > (5.0 / 255.0))),
        "fractionAboveTwentyFiveCodeValues": float(np.mean(difference > (25.0 / 255.0))),
        "psnrDb": psnr_db,
        "ssim": ssim_value,
        "levels": levels,
        "halftone": halftone,
        "timingsSeconds": {"pytorchReference": reference_seconds, "onnxCandidate": candidate_seconds},
        "checks": {
            "shape": shape_passed,
            "finite": finite_passed,
            "range": range_passed,
            "numerical": numerical_passed,
            "levels": levels_passed,
            "halftone": halftone_passed,
        },
        "visuals": visuals,
        "passed": passed,
    }


def onnx_path_for(directory: Path, source_filename: str, precision: str) -> Path:
    candidate = directory / f"{Path(source_filename).stem}.{precision}.onnx"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def create_contact_sheet(output_root: Path, cases: Iterable[dict[str, Any]], target: Path) -> None:
    visual_cases = [case for case in cases if case.get("visuals")]
    if not visual_cases:
        return
    cell_width, cell_height = 150, 170
    columns = 4
    rows = math.ceil(len(visual_cases) / columns)
    canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, case in enumerate(visual_cases):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        case_dir = output_root / f"{case['modelHeight']}p" / case["fixture"]
        panels = []
        for key in ("input", "reference", "candidate", "differenceX32"):
            with Image.open(case_dir / case["visuals"][key]) as image:
                tile = image.convert("RGB")
                tile.thumbnail((68, 58), Image.Resampling.LANCZOS)
                panels.append(tile.copy())
        canvas.paste(panels[0], (x + 4, y + 20))
        canvas.paste(panels[1], (x + 77, y + 20))
        canvas.paste(panels[2], (x + 4, y + 88))
        canvas.paste(panels[3], (x + 77, y + 88))
        draw.text((x + 4, y + 3), f"{case['modelHeight']}p {case['fixture'][:19]}", fill="black")
        draw.text((x + 4, y + 150), f"max={case['maximumAbsoluteError']:.4g}  SSIM={case['ssim']:.6f}", fill="black")
    canvas.save(target)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("models", type=Path, help="Directory containing the seven official .pth files")
    result.add_argument("onnx", type=Path, help="Directory containing the seven precision-qualified ONNX files")
    result.add_argument("output", type=Path, help="Validation output directory")
    result.add_argument("--provider", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument(
        "--cudnn-algo",
        choices=("HEURISTIC", "EXHAUSTIVE", "DEFAULT"),
        default="HEURISTIC",
        help="ONNX Runtime CUDA convolution algorithm search mode",
    )
    result.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    result.add_argument("--official-demo", type=Path, help="Optional official mangajanaiv1demo.webp")
    result.add_argument("--category-model-height", type=int, default=DEFAULT_CATEGORY_MODEL_HEIGHT)
    result.add_argument("--all-models-all-fixtures", action="store_true")
    result.add_argument("--model-height", type=int, action="append", dest="model_heights", help="Repeat to limit models")
    result.add_argument("--fixture", action="append", dest="fixture_names", help="Repeat to limit fixture names")
    result.add_argument("--no-images", action="store_true", help="Skip per-case PNGs and contact sheet")
    return result


def main() -> int:
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    thresholds = thresholds_for_precision(args.precision)
    fixtures = synthetic_fixtures()
    demo_used = None
    if args.official_demo:
        fixtures.extend(official_demo_fixtures(args.official_demo))
        demo_used = {"path": args.official_demo.name, "sha256": sha256_file(args.official_demo)}

    official_by_height = {model.nominal_height: model for model in OFFICIAL_MODELS}
    if args.category_model_height not in official_by_height:
        raise SystemExit(f"Unknown category model height: {args.category_model_height}")

    if args.model_heights:
        unknown_heights = sorted(set(args.model_heights) - set(official_by_height))
        if unknown_heights:
            raise SystemExit(f"Unknown model heights: {unknown_heights}")
        models_to_run = [model for model in OFFICIAL_MODELS if model.nominal_height in set(args.model_heights)]
    else:
        models_to_run = list(OFFICIAL_MODELS)
    available_fixture_names = {fixture.name for fixture in fixtures} | {fixture_mixed_tile().name}
    if args.fixture_names:
        unknown_fixtures = sorted(set(args.fixture_names) - available_fixture_names)
        if unknown_fixtures:
            raise SystemExit(f"Unknown fixtures: {unknown_fixtures}")

    all_results: list[dict[str, Any]] = []
    model_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for model in models_to_run:
        source_path = args.models / model.filename
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if sha256_file(source_path) != model.source_sha256:
            raise RuntimeError(f"Source SHA-256 mismatch: {source_path}")
        onnx_path = onnx_path_for(args.onnx, model.filename, args.precision)

        print(f"Loading {model.nominal_height}p PyTorch and ONNX", flush=True)
        session, onnx_load_seconds = make_session(onnx_path, args.provider, args.cudnn_algo)
        descriptor, pytorch_load_seconds = load_reference(source_path)
        reference_device = "cuda" if session.get_providers()[0] == "CUDAExecutionProvider" else "cpu"
        if args.precision == "fp16":
            descriptor.model.to(reference_device).half().eval()
        elif reference_device == "cuda":
            descriptor.model.to(reference_device).float().eval()
        selected_fixtures = (
            fixtures
            if args.all_models_all_fixtures or model.nominal_height == args.category_model_height
            else [fixture_mixed_tile()]
        )
        if args.fixture_names:
            selected_fixtures = [fixture for fixture in selected_fixtures if fixture.name in set(args.fixture_names)]
        if not selected_fixtures:
            raise RuntimeError(f"No selected fixtures apply to {model.nominal_height}p")
        cases = []
        for fixture in selected_fixtures:
            print(f"  {fixture.name}", flush=True)
            case = evaluate_case(
                fixture,
                descriptor,
                session,
                args.output,
                model.nominal_height,
                thresholds,
                args.precision,
                reference_device,
                not args.no_images,
            )
            cases.append(case)
            all_results.append(case)
        model_results.append(
            {
                "nominalHeight": model.nominal_height,
                "source": {"file": source_path.name, "sha256": model.source_sha256},
                "onnx": {"file": onnx_path.name, "sha256": sha256_file(onnx_path)},
                "provider": session.get_providers()[0],
                "referenceDevice": reference_device,
                "sessionLoadSeconds": {"pytorch": pytorch_load_seconds, "onnx": onnx_load_seconds},
                "cases": cases,
                "passed": all(case["passed"] for case in cases),
            }
        )
        del session, descriptor
        gc.collect()

    thresholds_json = {
        "maximumAbsoluteError": thresholds.max_absolute_error,
        "meanAbsoluteError": thresholds.mean_absolute_error,
        "minimumPsnrDb": thresholds.minimum_psnr_db,
        "minimumSsim": thresholds.minimum_ssim,
        "maximumLevelRegionMae": thresholds.maximum_level_region_mae,
        "maximumHalftoneEnergyRelativeError": thresholds.maximum_halftone_energy_relative_error,
    }
    summary = {
        "caseCount": len(all_results),
        "passedCount": sum(case["passed"] for case in all_results),
        "failedCount": sum(not case["passed"] for case in all_results),
        "worstMaximumAbsoluteError": max(case["maximumAbsoluteError"] for case in all_results),
        "worstMeanAbsoluteError": max(case["meanAbsoluteError"] for case in all_results),
        "minimumPsnrDb": min(case["psnrDb"] for case in all_results),
        "minimumSsim": min(case["ssim"] for case in all_results),
        "maximumHalftoneEnergyRelativeError": max(
            (case["halftone"]["relativeError"] for case in all_results if case["halftone"]),
            default=None,
        ),
        "maximumLevelRegionMae": max(
            (
                region["meanAbsoluteError"]
                for case in all_results
                if case["levels"]
                for region in case["levels"].values()
                if region["meanAbsoluteError"] is not None
            ),
            default=None,
        ),
    }
    document = {
        "schemaVersion": SCHEMA_VERSION,
        "createdBy": Path(__file__).name,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchCuda": torch.version.cuda,
            "torchCudnn": torch.backends.cudnn.version(),
            "onnxRuntime": ort.__version__,
            "numpy": np.__version__,
            "availableProviders": ort.get_available_providers(),
            "cudnnAlgorithmSearch": args.cudnn_algo,
            "tf32Allowed": False,
        },
        "comparison": {
            "reference": (
                f"PyTorch FP16 {model_results[0]['referenceDevice'].upper()} exact Spandrel boundary"
                if args.precision == "fp16" and model_results
                else (
                    f"PyTorch FP32 {model_results[0]['referenceDevice'].upper()} exact Spandrel boundary"
                    if model_results
                    else "PyTorch FP32 Spandrel descriptor"
                )
            ),
            "candidate": f"ONNX Runtime {args.precision.upper()} input/weights/output",
            "sharedPreprocessing": "EXIF transpose; corrected sRGB luminance; [0,1]; repeat to RGB",
            "thresholds": thresholds_json,
        },
        "officialDemo": demo_used,
        "scope": {
            "precision": args.precision,
            "allModelsFixture": "all-model-mixed-tile",
            "categoryModelHeight": args.category_model_height,
            "allModelsAllFixtures": args.all_models_all_fixtures,
            "selectedModelHeights": [model.nominal_height for model in models_to_run],
            "selectedFixtures": args.fixture_names,
            "seamValidation": "deferred until the Step 7 tiled-inference candidate exists",
            "fullNominalHeightPages": "deferred to Step 7 tiled full-page validation",
        },
        "summary": summary,
        "models": model_results,
        "totalSeconds": time.perf_counter() - started,
        "validForProductionAcceptance": args.precision == "fp32" or all(
            model["provider"] == "CUDAExecutionProvider" for model in model_results
        ),
        "passed": summary["failedCount"] == 0,
    }
    results_path = args.output / "validation-results.json"
    results_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_images:
        create_contact_sheet(args.output, all_results, args.output / "visual-summary.png")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {results_path}", flush=True)
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
