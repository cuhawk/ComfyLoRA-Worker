"""Inspect safetensors without materializing tensors or importing a GPU framework."""

import hashlib
import hmac
from pathlib import Path
import re

from safetensors import SafetensorError, safe_open

# Common paired suffixes used by ComfyUI's LoRA adapter.
PAIRS = (
    (".lora_up.weight", ".lora_down.weight"),
    (".lora_B.weight", ".lora_A.weight"),
    (".lora_B", ".lora_A"),
    (".lora_B.default.weight", ".lora_A.default.weight"),
    (".lora.up.weight", ".lora.down.weight"),
    ("_lora.up.weight", "_lora.down.weight"),
    (".lora_linear_layer.up.weight", ".lora_linear_layer.down.weight"),
)


def inspect_lora(path: Path, expected_sha256: str | None = None) -> dict:
    if path.suffix != ".safetensors":
        raise ValueError("expected a .safetensors file")
    if expected_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
        raise ValueError("expected SHA-256 must be 64 hexadecimal characters")
    try:
        # The Rust reader validates the header and maps the file. get_slice reads metadata.
        with path.open("rb") as stream, safe_open(str(path), framework="np") as reader:
            metadata = reader.metadata() or {}
            tensors = {}
            for key in sorted(reader.keys()):
                tensor = reader.get_slice(key)
                tensors[key] = {"shape": tensor.get_shape(), "dtype": tensor.get_dtype()}
            sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    except SafetensorError as exc:
        raise ValueError(f"invalid safetensors file: {exc}") from exc
    if expected_sha256 is not None and not hmac.compare_digest(sha256, expected_sha256.lower()):
        raise ValueError("SHA-256 mismatch")

    paired_keys = set()
    targets = set()
    for up_suffix, down_suffix in PAIRS:
        for up in tensors:
            if not up.endswith(up_suffix):
                continue
            target = up.removesuffix(up_suffix)
            down = target + down_suffix
            if not target or down not in tensors:
                continue
            up_shape = tensors[up]["shape"]
            down_shape = tensors[down]["shape"]
            if len(up_shape) >= 2 and len(down_shape) >= 2 and up_shape[1] == down_shape[0] > 0:
                targets.add(target)
                paired_keys.update((up, down))
                if target + ".alpha" in tensors:
                    paired_keys.add(target + ".alpha")

    return {
        "sha256": sha256,
        "metadata": metadata,
        "tensors": tensors,
        "compatible": bool(targets),
        "paired_targets": sorted(targets),
        "unverified_keys": sorted(tensors.keys() - paired_keys),
    }
