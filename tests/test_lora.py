import hashlib
import json
import math

import pytest

from comfy_lora_worker.cli import main
from comfy_lora_worker.lora import inspect_lora


def write_fixture(path, up=".lora_up.weight", down=".lora_down.weight", rank=2):
    target = "lora_unet_test"
    header = {"__metadata__": {"sku": "mug"}}
    offset = 0
    for key, shape in {
        target + up: [3, rank], target + down: [2, 4], target + ".alpha": [], "extra": [1],
    }.items():
        end = offset + math.prod(shape) * 4
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}
        offset = end
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(offset))
    return path


@pytest.mark.parametrize(("up", "down"), [
    (".lora_up.weight", ".lora_down.weight"),
    (".lora_B.weight", ".lora_A.weight"),
    (".lora_B", ".lora_A"),
])
def test_metadata_hash_and_pairs(tmp_path, up, down):
    path = write_fixture(tmp_path / "mug.safetensors", up, down)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    report = inspect_lora(path, digest.upper())
    assert report["sha256"] == digest
    assert report["metadata"] == {"sku": "mug"}
    assert report["paired_targets"] == ["lora_unet_test"]
    assert report["tensors"]["lora_unet_test" + up] == {"dtype": "F32", "shape": [3, 2]}
    assert report["compatible"] is True
    assert report["unverified_keys"] == ["extra"]


def test_invalid_pair_rank(tmp_path):
    path = write_fixture(tmp_path / "bad.safetensors", rank=3)
    assert not inspect_lora(path)["compatible"]
    assert main(["inspect-lora", str(path)]) == 1


def test_bad_header(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="invalid safetensors"):
        inspect_lora(path)


def test_bad_hash(tmp_path):
    path = write_fixture(tmp_path / "mug.safetensors")
    with pytest.raises(ValueError, match="mismatch"):
        inspect_lora(path, "0" * 64)
    with pytest.raises(ValueError, match="hexadecimal"):
        inspect_lora(path, "not-hex")
