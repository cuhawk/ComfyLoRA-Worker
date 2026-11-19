from contextlib import asynccontextmanager
import json

import httpx
import pytest

from comfy_lora_worker import comfy
from comfy_lora_worker.comfy import ComfyClient, ComfyError


class Socket:
    def __init__(self, events):
        self.events = iter(events)

    async def recv(self):
        event = next(self.events)
        if isinstance(event, Exception):
            raise event
        return event


def event(kind, prompt="p1", node=None):
    return json.dumps({"type": kind, "data": {"prompt_id": prompt, "node": node}})


def test_history_requires_valid_images():
    with pytest.raises(ComfyError, match="malformed history"):
        ComfyClient._outputs({"outputs": {"9": {"images": [{"filename": "../secret"}]}}})
    with pytest.raises(ComfyError, match="no images"):
        ComfyClient._outputs({"outputs": {}})


@pytest.mark.parametrize("mode", ["success", "disconnect", "unavailable", "execution_error", "interrupted"])
async def test_http_and_websocket(template, monkeypatch, mode):
    client_ids = []
    histories = 0
    events = [b"preview", "garbage", event("execution_error", "other")]
    events.append({
        "success": event("executing"), "disconnect": OSError("closed"),
        "unavailable": None, "execution_error": event("execution_error"),
        "interrupted": event("execution_interrupted"),
    }[mode])

    @asynccontextmanager
    async def connect(url, **kwargs):
        client_ids.append(httpx.URL(url).params["clientId"])
        if mode == "unavailable":
            raise OSError("offline")
        yield Socket(events)

    def dispatch(request):
        nonlocal histories
        assert request.url.path.startswith("/comfy/")
        path = request.url.path.removeprefix("/comfy")
        if path == "/prompt":
            assert request.method == "POST"
            assert json.loads(request.content) == {"client_id": client_ids[0], "prompt": template}
            return httpx.Response(200, json={"prompt_id": "p1"})
        if path == "/history/p1":
            histories += 1
            return httpx.Response(200, json={} if histories == 1 else {
                "p1": {"status": {"completed": True}, "outputs": {
                    "9": {"images": [{"filename": "mug.png", "subfolder": "sku", "type": "output"}]},
                }},
            })
        assert path == "/view"
        assert dict(request.url.params) == {"filename": "mug.png", "subfolder": "sku", "type": "output"}
        return httpx.Response(200, content=b"image data")

    monkeypatch.setattr(comfy, "connect", connect)
    async with httpx.AsyncClient(
        base_url="https://comfy.example/comfy", transport=httpx.MockTransport(dispatch),
    ) as http:
        client = ComfyClient(http, poll_interval=0.001)
        if mode in {"execution_error", "interrupted"}:
            with pytest.raises(ComfyError, match="ComfyUI reported execution_"):
                await client.render(template)
        else:
            images = await client.render(template)
            assert histories == 2
            async with client.image_stream(images[0]) as response:
                assert await response.aread() == b"image data"


@pytest.mark.parametrize(("response", "message"), [
    (httpx.Response(400, json={"error": "bad graph"}), "POST prompt failed"),
    (httpx.Response(200, json={"prompt_id": "p", "node_errors": {"3": "bad seed"}}), "rejected"),
    (httpx.Response(200, json=[]), "invalid response"),
])
async def test_prompt_failure(monkeypatch, template, response, message):
    @asynccontextmanager
    async def unavailable(*args, **kwargs):
        raise OSError("offline")
        yield

    monkeypatch.setattr(comfy, "connect", unavailable)
    async with httpx.AsyncClient(
        base_url="http://comfy", transport=httpx.MockTransport(lambda _: response),
    ) as http:
        with pytest.raises(ComfyError, match=message):
            await ComfyClient(http).render(template)


async def test_render_timeout(monkeypatch, template):
    @asynccontextmanager
    async def unavailable(*args, **kwargs):
        raise OSError("offline")
        yield

    monkeypatch.setattr(comfy, "connect", unavailable)
    async with httpx.AsyncClient(
        base_url="http://comfy",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"prompt_id": "never"})),
    ) as http:
        with pytest.raises(ComfyError, match="timed out"):
            await ComfyClient(http, timeout=0.02, poll_interval=0.001).render(template)
