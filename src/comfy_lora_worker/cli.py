"""Command-line entry point for serving, inspecting and rendering."""

import argparse
import json
from pathlib import Path
import sys

import uvicorn

from .api import Settings, create_app
from .lora import inspect_lora
from .workflow import Nodes, Parameters, render_workflow


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="comfylora")
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the FastAPI worker")
    serve.add_argument("--workflow", required=True, type=Path)
    serve.add_argument("--nodes", type=Path, help="JSON file mapping workflow roles to node IDs")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", default=8000, type=int)

    inspect = commands.add_parser("inspect-lora", help="inspect a safetensors LoRA")
    inspect.add_argument("path", type=Path)
    inspect.add_argument("--sha256", help="verify against an expected SHA-256")

    render = commands.add_parser("render-workflow", help="write parameterized API-format JSON")
    render.add_argument("workflow", type=Path)
    render.add_argument("parameters", type=Path)
    render.add_argument("--nodes", type=Path)
    render.add_argument("-o", "--output", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-lora":
            report = inspect_lora(args.path, args.sha256)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["compatible"] else 1

        template = read_json(args.workflow)
        nodes = Nodes.model_validate(read_json(args.nodes)) if args.nodes else Nodes()
        if args.command == "serve":
            uvicorn.run(create_app(template, Settings.from_env(), nodes), host=args.host, port=args.port)
        else:
            graph = render_workflow(template, Parameters.model_validate(read_json(args.parameters)), nodes)
            output = json.dumps(graph, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            if args.output:
                args.output.write_text(output, encoding="utf-8")
            else:
                sys.stdout.write(output)
    except (OSError, ValueError) as exc:
        print(f"comfylora: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
