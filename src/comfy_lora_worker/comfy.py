"""Async ComfyUI API client with WebSocket hints and history reconciliation."""

import asyncio
from contextlib import AsyncExitStack
import json
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field, ValidationError, field_validator
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from .workflow import Graph


class ComfyError(RuntimeError):
    pass


class Image(BaseModel):
    filename: str = Field(min_length=1)
    subfolder: str = ""
    type: str = "output"

    @field_validator("filename", "subfolder")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if any(part in {".", ".."} for part in value.split("/")) or (
            value.startswith("/") or any(c in value for c in "\\:\x00")
        ):
            raise ValueError("invalid ComfyUI image path")
        return value

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        if value not in {"output", "temp"}:
            raise ValueError("unsupported ComfyUI image type")
        return value


class ComfyClient:
    def __init__(self, http: httpx.AsyncClient, *, timeout: float = 300, poll_interval: float = 1):
        self.http = http
        self.timeout = timeout
        self.poll_interval = poll_interval

    async def _json(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = await self.http.request(method, path, **kwargs)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ComfyError(f"ComfyUI {method} {path} failed: {exc}") from exc
        if not isinstance(data, dict):
            raise ComfyError(f"ComfyUI {path} returned an invalid response")
        return data

    async def stats(self) -> dict:
        return await self._json("GET", "system_stats")

    async def render(self, workflow: Graph) -> list[Image]:
        try:
            async with asyncio.timeout(self.timeout):
                return await self._render(workflow)
        except TimeoutError as exc:
            raise ComfyError("ComfyUI render timed out") from exc

    async def _render(self, workflow: Graph) -> list[Image]:
        client_id = uuid4().hex
        base = self.http.base_url
        ws_url = base.join("ws").copy_with(
            scheme="wss" if base.scheme == "https" else "ws",
            params={"clientId": client_id},
        )
        async with AsyncExitStack() as stack:
            try:
                ws = await stack.enter_async_context(connect(
                    str(ws_url), open_timeout=min(3, self.timeout / 2), close_timeout=1,
                    max_size=8 * 1024 * 1024,
                ))
            except (TimeoutError, OSError, WebSocketException):
                ws = None

            result = await self._json("POST", "prompt", json={
                "prompt": workflow, "client_id": client_id,
            })
            prompt_id = result.get("prompt_id")
            if not isinstance(prompt_id, str) or not prompt_id or result.get("node_errors"):
                raise ComfyError("ComfyUI rejected the workflow")
            history_path = f"history/{quote(prompt_id, safe='')}"
            while True:
                history = (await self._json("GET", history_path)).get(prompt_id)
                if history is not None:
                    return self._outputs(history)
                if ws is None:
                    await asyncio.sleep(self.poll_interval)
                    continue
                deadline = asyncio.get_running_loop().time() + self.poll_interval
                try:
                    async with asyncio.timeout_at(deadline):
                        while True:
                            frame = await ws.recv()
                            if isinstance(frame, bytes):
                                continue
                            try:
                                event = json.loads(frame)
                            except ValueError:
                                continue
                            if not isinstance(event, dict):
                                continue
                            data = event.get("data")
                            if not isinstance(data, dict) or data.get("prompt_id") != prompt_id:
                                continue
                            kind = event.get("type")
                            if kind in {"execution_error", "execution_interrupted"}:
                                raise ComfyError(f"ComfyUI reported {kind}")
                            if kind == "execution_success" or (
                                kind == "executing" and data.get("node") is None
                            ):
                                break
                except TimeoutError:
                    pass
                except (OSError, WebSocketException):
                    ws = None
                    await asyncio.sleep(self.poll_interval)

    @staticmethod
    def _outputs(history: dict) -> list[Image]:
        try:
            status = history.get("status", {})
            if status.get("status_str") == "error" or status.get("completed") is False:
                raise ComfyError("ComfyUI history reports an unsuccessful execution")
            images = [
                Image.model_validate(image)
                for output in history["outputs"].values()
                for image in output.get("images", [])
            ]
        except (AttributeError, KeyError, TypeError, ValidationError) as exc:
            raise ComfyError("ComfyUI returned malformed history") from exc
        if not images:
            raise ComfyError("ComfyUI returned no images")
        return images

    def image_stream(self, image: Image):
        return self.http.stream("GET", "view", params=image.model_dump())
