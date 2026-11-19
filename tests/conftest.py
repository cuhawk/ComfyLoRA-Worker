import json
from pathlib import Path

import pytest


@pytest.fixture
def template():
    path = Path(__file__).resolve().parents[1] / "examples/product_sku_workflow.json"
    return json.loads(path.read_text())


@pytest.fixture
def params():
    return {
        "lora": "catalog/mug.safetensors", "strength_model": 0.8,
        "strength_clip": 0.4, "seed": 42, "width": 768, "height": 512,
        "sampler_name": "euler", "steps": 24,
    }
