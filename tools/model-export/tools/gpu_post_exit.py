#!/usr/bin/env python3
"""Confirm that the completed GPU probe process released its CUDA allocations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pynvml


MIB = 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--tolerance-mib", type=float, default=128.0)
    args = parser.parse_args()

    if args.samples < 1 or args.interval < 0 or args.tolerance_mib < 0:
        raise SystemExit("samples must be positive; interval and tolerance must be nonnegative")
    probe = json.loads(args.probe.read_text(encoding="utf-8"))
    baseline = float(probe["initialUsedMiB"])

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.device_id)
        samples: list[float] = []
        for index in range(args.samples):
            samples.append(pynvml.nvmlDeviceGetMemoryInfo(handle).used / MIB)
            if index + 1 < args.samples:
                time.sleep(args.interval)
    finally:
        pynvml.nvmlShutdown()

    minimum = min(samples)
    result = {
        "schemaVersion": 1,
        "deviceId": args.device_id,
        "probeInitialUsedMiB": baseline,
        "samplesMiB": samples,
        "minimumAfterExitUsedMiB": minimum,
        "finalAfterExitUsedMiB": samples[-1],
        "minimumDeltaFromBaselineMiB": minimum - baseline,
        "toleranceMiB": args.tolerance_mib,
        "releasedWithinTolerance": minimum <= baseline + args.tolerance_mib,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["releasedWithinTolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
