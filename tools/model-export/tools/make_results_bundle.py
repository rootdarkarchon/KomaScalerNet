#!/usr/bin/env python3
"""Summarize a GPU validation run and create its single return ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile


MODEL_SUFFIXES = {".onnx", ".pth"}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def return_files(root: Path) -> list[Path]:
    """Collect diagnostics only; never follow or return model files."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and "resolved-pytorch-models" not in path.relative_to(root).parts
        and path.suffix.lower() not in MODEL_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--validation-exit", type=int, required=True)
    parser.add_argument("--probe-exit", type=int, required=True)
    parser.add_argument("--post-exit", type=int)
    args = parser.parse_args()

    root = args.result_dir.resolve()
    validation = load_json(root / "validation" / "validation-results.json")
    probe = load_json(root / "gpu-probe.json")
    post_exit = load_json(root / "gpu-post-exit.json")
    preflight = load_json(root / "environment" / "preflight.json")

    status = {
        "validationExitCode": args.validation_exit,
        "probeExitCode": args.probe_exit,
        "validationJsonPresent": validation is not None,
        "gpuProbeJsonPresent": probe is not None,
        "gpuPostExitJsonPresent": post_exit is not None,
        "postExitCode": args.post_exit,
    }
    (root / "exit-status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    lines = ["# MangaJaNai GPU Validation Return Summary", ""]
    if preflight:
        lines.extend(
            [
                f"- GPU: `{preflight.get('gpu')}`",
                f"- Driver: `{preflight.get('driver')}`",
                f"- PyTorch: `{preflight.get('torch')}`; CUDA `{preflight.get('torchCuda')}`; cuDNN `{preflight.get('torchCudnn')}`",
                f"- ONNX Runtime: `{preflight.get('onnxRuntime')}`",
            ]
        )
    if validation:
        summary = validation["summary"]
        precision = validation.get("scope", {}).get("precision", "unknown").upper()
        lines.extend(
            [
                "",
                f"## {precision} parity",
                "",
                f"- Passed: `{validation.get('passed')}`",
                f"- Valid for production acceptance: `{validation.get('validForProductionAcceptance')}`",
                f"- Cases: `{summary['passedCount']}/{summary['caseCount']}` passed",
                f"- Worst max error: `{summary['worstMaximumAbsoluteError']}`",
                f"- Worst MAE: `{summary['worstMeanAbsoluteError']}`",
                f"- Minimum PSNR: `{summary['minimumPsnrDb']} dB`",
                f"- Minimum SSIM: `{summary['minimumSsim']}`",
            ]
        )
    else:
        lines.extend(["", "## Model parity", "", "Validation JSON was not produced; see `validation.log`."])
    if probe:
        lines.extend(
            [
                "",
                "## Seven-session GPU probe",
                "",
                f"- Precision: `{probe.get('environment', {}).get('precision')}`",
                f"- Session-load VRAM delta: `{probe.get('sessionLoadResidentDeltaMiB')} MiB`",
                f"- Resident VRAM delta: `{probe.get('allSessionsResidentDeltaMiB')} MiB`",
                f"- Initial used VRAM: `{probe.get('initialUsedMiB')} MiB`",
                f"- Used VRAM after seven sessions: `{probe.get('afterAllSessionsUsedMiB')} MiB`",
            ]
        )
    else:
        lines.extend(["", "## Seven-session GPU probe", "", "Probe JSON was not produced; see `gpu-probe.log`."])
    if post_exit:
        lines.extend(
            [
                "",
                "## Worker-process VRAM release",
                "",
                f"- Released within tolerance: `{post_exit.get('releasedWithinTolerance')}`",
                f"- Minimum delta from pre-probe baseline: `{post_exit.get('minimumDeltaFromBaselineMiB')} MiB`",
                f"- Tolerance: `{post_exit.get('toleranceMiB')} MiB`",
            ]
        )
    lines.extend(["", "The complete metrics, images, environment, hashes, and logs are included in this archive.", ""])
    (root / "RETURN-SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    files = [path for path in return_files(root) if path.name != "result-sha256s.txt"]
    manifest = "\n".join(f"{sha256(path)}  {path.relative_to(root)}" for path in files) + "\n"
    (root / "result-sha256s.txt").write_text(manifest, encoding="utf-8")
    files = return_files(root)

    archive = root.parent / f"{root.name}-RETURN.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=6) as output:
        for path in files:
            output.write(path, arcname=f"{root.name}/{path.relative_to(root)}")
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
