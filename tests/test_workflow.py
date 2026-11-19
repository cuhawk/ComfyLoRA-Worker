from copy import deepcopy
import json

from pydantic import ValidationError
import pytest

from comfy_lora_worker.cli import main
from comfy_lora_worker.workflow import Nodes, Parameters, render_workflow


def test_determinism_and_injection(template, params):
    original = deepcopy(template)
    p = Parameters(**params, positive="SKU 42", negative="", scheduler="karras")
    first = render_workflow(template, p)
    assert first == render_workflow(template, p)
    assert template == original
    assert first["10"]["inputs"] == {
        "model": ["4", 0], "clip": ["4", 1], "lora_name": params["lora"],
        "strength_model": 0.8, "strength_clip": 0.4,
    }
    assert first["3"]["inputs"] == {
        **original["3"]["inputs"], "model": ["10", 0], "seed": 42,
        "steps": 24, "scheduler": "karras",
    }
    assert first["5"]["inputs"] == {"width": 768, "height": 512, "batch_size": 1}
    assert first["6"]["inputs"] == {"clip": ["10", 1], "text": "SKU 42"}
    assert first["7"]["inputs"] == {"clip": ["10", 1], "text": ""}


def test_updates_existing_loader(template, params):
    first = render_workflow(template, Parameters(**params))
    second = render_workflow(first, Parameters(**(params | {"lora": "new.safetensors"})))
    assert first.keys() == second.keys()
    assert second["10"]["inputs"]["lora_name"] == "new.safetensors"


def test_custom_node_ids(template, params):
    assert "sku-loader" in render_workflow(template, Parameters(**params), Nodes(lora="sku-loader"))


@pytest.mark.parametrize(("key", "value"), [
    ("lora", "../file.safetensors"), ("lora", "C:\\file.safetensors"),
    ("lora", "/file.safetensors"), ("lora", "file.pt"), ("lora", "a//file.safetensors"),
    ("seed", -1), ("seed", 2**64), ("seed", True), ("width", 513), ("height", 9000),
    ("steps", 0), ("sampler_name", ""), ("strength_model", float("nan")),
    ("strength_clip", 100),
])
def test_rejects_bad_parameters(params, key, value):
    with pytest.raises(ValidationError):
        Parameters(**(params | {key: value}))


@pytest.mark.parametrize(("node_id", "node", "message"), [
    ("5", {"class_type": "Unknown", "inputs": {}}, "EmptyLatentImage"),
    ("10", {"class_type": "SaveImage", "inputs": {}}, "LoraLoader"),
    ("50", {"class_type": "Custom", "inputs": {"missing": ["61", 0]}}, "invalid link"),
    ("50", {"class_type": "Custom", "inputs": {"self": ["50", 0]}}, "cycle"),
    ("50", {"inputs": {}}, "invalid API-format node"),
])
def test_invalid_graph(template, params, node_id, node, message):
    template[node_id] = node
    with pytest.raises(ValueError, match=message):
        render_workflow(template, Parameters(**params))


def test_cli_renders_workflow(tmp_path, template, params):
    workflow, parameters, output = (tmp_path / name for name in ("workflow", "params", "out"))
    workflow.write_text(json.dumps(template))
    parameters.write_text(json.dumps(params))
    assert main(["render-workflow", str(workflow), str(parameters), "-o", str(output)]) == 0
    assert json.loads(output.read_text()) == render_workflow(template, Parameters(**params))
    parameters.write_text("{}")
    assert main(["render-workflow", str(workflow), str(parameters)]) == 2
