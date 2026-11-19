import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
from unittest.mock import AsyncMock

import httpx
from pydantic import ValidationError
import pytest

from comfy_lora_worker import comfy
from comfy_lora_worker.api import Settings, create_app
from comfy_lora_worker.comfy import ComfyError, Image
from comfy_lora_worker.jobs import Job, JobQueue, Webhook


async def test_queue_fifo_backpressure_failure_and_retention(template):
    release = asyncio.Event()
    started = asyncio.Event()
    calls = []

    async def render(graph):
        calls.append(graph["id"])
        if graph["id"] == 1:
            started.set()
            await release.wait()
            raise ComfyError("boom")
        return [Image(filename="ok.png")]

    client = AsyncMock()
    client.render.side_effect = render
    queue = JobQueue(client, capacity=1, retention=1)
    first = queue.submit({"id": 1})
    assert first.status == "queued"
    task = asyncio.create_task(queue.run())
    try:
        await asyncio.wait_for(started.wait(), 1)
        assert first.status == "running"
        second = queue.submit({"id": 2})
        with pytest.raises(asyncio.QueueFull):
            queue.submit(template)
        release.set()
        await asyncio.wait_for(queue.pending.join(), 1)
        assert calls == [1, 2]
        assert first.status == "failed"
        assert first.started_at <= first.finished_at
        assert queue.get(first.id) is None
        assert queue.get(second.id).status == "succeeded"
        assert second.images == [Image(filename="ok.png")]
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.parametrize(("codes", "success"), [
    ([202], True), ([503, 429, 204], True), ([503, 503, 503], False),
    ([302], False), ([400], False),
])
async def test_signed_webhook_retry_policy(codes, success):
    received = []

    def dispatch(request):
        body = request.content
        signature = hmac.new(b"secret", body, hashlib.sha256).hexdigest()
        assert request.headers["X-ComfyLoRA-Signature"] == f"sha256={signature}"
        payload = json.loads(body)
        assert request.headers["X-ComfyLoRA-Event-ID"] == payload["id"]
        assert payload["status"] == "failed"
        received.append(body)
        return httpx.Response(codes[len(received) - 1], headers={"Location": "/other"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(dispatch)) as http:
        webhook = Webhook(http, "https://hook.example/callback", "secret", retry_delay=0)
        assert await webhook.deliver(Job(status="failed")) is success
    assert len(received) == len(codes)
    assert len(set(received)) == 1


def test_webhook_configuration_requires_pair():
    with pytest.raises(ValidationError, match="configure both"):
        Settings(webhook_url="https://hook.example")


async def test_api_lifecycle_and_signed_webhook(template, params, monkeypatch):
    @asynccontextmanager
    async def socket(*args, **kwargs):
        yield None

    monkeypatch.setattr(comfy, "connect", socket)
    callbacks = []

    def dispatch(request):
        path = request.url.path
        if path == "/system_stats":
            return httpx.Response(200, json={"devices": [{"name": "mock GPU", "vram_free": 4096}]})
        if path == "/prompt":
            graph = json.loads(request.content)["prompt"]
            assert graph["10"]["inputs"]["lora_name"] == params["lora"]
            return httpx.Response(200, json={"prompt_id": "mock"})
        if path == "/history/mock":
            return httpx.Response(200, json={
                "mock": {"outputs": {"9": {"images": [{"filename": "mug.png"}]}}},
            })
        assert path == "/view"
        return httpx.Response(200, content=b"mug png")

    def callback(request):
        callbacks.append(json.loads(request.content))
        return httpx.Response(204)

    settings = Settings(
        api_token="test-token", webhook_url="https://hook.example/callback", webhook_secret="secret",
    )
    app = create_app(
        template, settings, comfy_transport=httpx.MockTransport(dispatch),
        webhook_transport=httpx.MockTransport(callback),
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client,
    ):
        assert (await client.get("/health")).json() == {"status": "ok"}
        assert (await client.post("/jobs", json=params)).status_code == 401
        client.headers["Authorization"] = "Bearer test-token"
        assert (await client.get("/ready")).json()["devices"][0]["vram_free"] == 4096
        assert (await client.post("/jobs", json=params | {"width": 47})).status_code == 422
        assert (await client.get("/jobs/missing")).status_code == 404
        response = await client.post("/jobs", json=params)
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        job_id = response.json()["id"]
        await asyncio.wait_for(app.state.jobs.pending.join(), 1)
        status = (await client.get(f"/jobs/{job_id}")).json()
        assert status["status"] == "succeeded"
        assert status["webhook_status"] == "delivered"
        assert callbacks[0]["id"] == job_id
        assert (await client.get(f"/jobs/{job_id}/images/0")).content == b"mug png"
        assert (await client.get(f"/jobs/{job_id}/images/-1")).status_code == 404


async def test_api_readiness_failure(template):
    transport = httpx.MockTransport(lambda _: httpx.Response(503))
    app = create_app(template, comfy_transport=transport)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker") as client,
    ):
        assert (await client.get("/health")).status_code == 200
        assert (await client.get("/ready")).status_code == 503
