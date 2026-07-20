#!/usr/bin/env python3
"""Safely inspect tensor metadata in a PyTorch zip checkpoint without importing torch.

The restricted unpickler accepts only the small set of constructors used by
PyTorch state-dict serialization and never materializes tensor storage bytes.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import pickle
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageInfo:
    dtype: str
    key: str
    location: str
    elements: int


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    storage_key: str
    location: str
    storage_elements: int
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool


class StorageType:
    def __init__(self, name: str):
        self.name = name


def rebuild_tensor_v2(storage, offset, shape, stride, requires_grad, _hooks):
    if not isinstance(storage, StorageInfo):
        raise pickle.UnpicklingError("Unexpected tensor storage descriptor")
    return TensorInfo(
        dtype=storage.dtype,
        storage_key=storage.key,
        location=storage.location,
        storage_elements=storage.elements,
        offset=offset,
        shape=tuple(shape),
        stride=tuple(stride),
        requires_grad=requires_grad,
    )


class RestrictedTorchUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name.removesuffix("Storage"))
        raise pickle.UnpicklingError(f"Blocked global: {module}.{name}")

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != "storage":
            raise pickle.UnpicklingError(f"Unexpected persistent id: {pid!r}")
        _, storage_type, key, location, elements, *_ = pid
        if not isinstance(storage_type, StorageType):
            raise pickle.UnpicklingError("Unexpected storage type")
        return StorageInfo(storage_type.name, str(key), str(location), int(elements))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        pickle_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(pickle_names) != 1:
            raise ValueError(f"Expected one data.pkl, found {pickle_names}")
        root = RestrictedTorchUnpickler(io.BytesIO(archive.read(pickle_names[0]))).load()

    if not isinstance(root, (dict, collections.OrderedDict)):
        raise ValueError(f"Unexpected top-level object: {type(root).__name__}")

    tensors = []
    non_tensors = []
    for key, value in root.items():
        if isinstance(value, TensorInfo):
            tensors.append({"key": key, **asdict(value)})
        else:
            non_tensors.append({"key": key, "type": type(value).__name__, "repr": repr(value)})

    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "serialization": "pytorch-zip",
        "top_level_type": type(root).__name__,
        "tensor_count": len(tensors),
        "non_tensor_entries": non_tensors,
        "dtypes": sorted({tensor["dtype"] for tensor in tensors}),
        "devices_recorded": sorted({tensor["location"] for tensor in tensors}),
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    results = [inspect(path) for path in args.checkpoints]
    if args.summary:
        results = [
            {
                key: value
                for key, value in result.items()
                if key not in {"tensors", "non_tensor_entries"}
            }
            | {
                "first_tensor": result["tensors"][0],
                "last_tensor": result["tensors"][-1],
                "key_shape_signature": hashlib.sha256(
                    json.dumps(
                        [(tensor["key"], tensor["shape"], tensor["dtype"]) for tensor in result["tensors"]],
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            }
            for result in results
        ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
