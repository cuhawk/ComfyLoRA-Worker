"""Parameterize a trusted ComfyUI API-format graph."""

from copy import deepcopy
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

Graph = dict[str, dict[str, Any]]
Name = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Dimension = Annotated[int, Field(strict=True, ge=64, le=8192, multiple_of=8)]


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)

    lora: Annotated[str, StringConstraints(min_length=1, max_length=255)]
    strength_model: float = Field(default=1.0, ge=-20, le=20)
    strength_clip: float = Field(default=1.0, ge=-20, le=20)
    seed: int = Field(strict=True, ge=0, le=2**64 - 1)
    width: Dimension
    height: Dimension
    sampler_name: Name
    steps: int = Field(strict=True, ge=1, le=1000)
    scheduler: Name | None = None
    positive: str | None = Field(default=None, max_length=16000)
    negative: str | None = Field(default=None, max_length=16000)

    @field_validator("lora")
    @classmethod
    def validate_lora(cls, name: str) -> str:
        if (
            not name.endswith(".safetensors")
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or any(c in name for c in "\\:")
            or any(ord(c) < 32 for c in name)
        ):
            raise ValueError("expected a relative .safetensors path under models/loras")
        return name


class Nodes(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sampler: Name = "3"
    checkpoint: Name = "4"
    latent: Name = "5"
    positive: Name = "6"
    negative: Name = "7"
    lora: Name = "10"


def validate_graph(graph: Graph) -> None:
    if not isinstance(graph, dict) or not graph:
        raise ValueError("expected a nonempty ComfyUI API-format graph")
    for key, node in graph.items():
        if (
            not isinstance(key, str) or not key or not isinstance(node, dict)
            or not isinstance(node.get("class_type"), str) or not node["class_type"]
            or not isinstance(node.get("inputs"), dict)
        ):
            raise ValueError(f"invalid API-format node {key!r}")
    dependencies = {}
    for key, node in graph.items():
        dependencies[key] = set()
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                source, slot = value
                if source not in graph or type(slot) is not int or slot < 0:
                    raise ValueError(f"invalid link on node {key!r}: {value!r}")
                dependencies[key].add(source)
    try:
        TopologicalSorter(dependencies).prepare()
    except CycleError as exc:
        raise ValueError("workflow contains a dependency cycle") from exc


def _inputs(graph: Graph, node_id: str, class_type: str, required: set[str]) -> dict:
    node = graph.get(node_id, {})
    if node.get("class_type") != class_type or not required <= node.get("inputs", {}).keys():
        raise ValueError(f"node {node_id!r} must be a {class_type} with inputs {sorted(required)}")
    return node["inputs"]


def render_workflow(template: Graph, parameters: Parameters, nodes: Nodes = Nodes()) -> Graph:
    validate_graph(template)
    if len(set(nodes.model_dump().values())) != len(type(nodes).model_fields):
        raise ValueError("workflow roles must have distinct node IDs")

    graph = deepcopy(template)
    p = parameters
    _inputs(graph, nodes.checkpoint, "CheckpointLoaderSimple", {"ckpt_name"})
    sampler = _inputs(graph, nodes.sampler, "KSampler", {
        "model", "seed", "steps", "sampler_name", "scheduler", "positive", "negative",
        "latent_image", "cfg", "denoise",
    })
    latent = _inputs(graph, nodes.latent, "EmptyLatentImage", {"width", "height", "batch_size"})
    positive = _inputs(graph, nodes.positive, "CLIPTextEncode", {"clip", "text"})
    negative = _inputs(graph, nodes.negative, "CLIPTextEncode", {"clip", "text"})
    if nodes.lora in graph:
        _inputs(graph, nodes.lora, "LoraLoader", {
            "model", "clip", "lora_name", "strength_model", "strength_clip",
        })
    graph[nodes.lora] = {
        **graph.get(nodes.lora, {}),
        "class_type": "LoraLoader",
        "inputs": {
            "model": [nodes.checkpoint, 0],
            "clip": [nodes.checkpoint, 1],
            "lora_name": p.lora,
            "strength_model": p.strength_model,
            "strength_clip": p.strength_clip,
        },
    }
    sampler.update(
        model=[nodes.lora, 0], seed=p.seed, steps=p.steps, sampler_name=p.sampler_name,
        positive=[nodes.positive, 0], negative=[nodes.negative, 0], latent_image=[nodes.latent, 0],
    )
    if p.scheduler is not None:
        sampler["scheduler"] = p.scheduler
    latent.update(width=p.width, height=p.height)
    positive["clip"] = negative["clip"] = [nodes.lora, 1]
    if p.positive is not None:
        positive["text"] = p.positive
    if p.negative is not None:
        negative["text"] = p.negative
    validate_graph(graph)
    return graph
