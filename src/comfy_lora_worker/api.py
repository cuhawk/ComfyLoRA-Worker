"""FastAPI application and operator configuration."""

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import os
from pathlib import Path
from typing import Annotated
from secrets import compare_digest

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx
from pydantic import AnyHttpUrl, BaseModel, Field, SecretStr, model_validator
from starlette.background import BackgroundTask

from .comfy import ComfyClient, ComfyError
from .jobs import Job, JobQueue, Webhook
from .workflow import Graph, Nodes, Parameters, render_workflow, validate_graph


class Settings(BaseModel):
    comfy_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8188")
    queue_capacity: int = Field(default=32, ge=1, le=10000)
    retention: int = Field(default=256, ge=1, le=100000)
    render_timeout: float = Field(default=300, gt=0, le=86400)
    webhook_url: AnyHttpUrl | None = None
    webhook_secret: SecretStr | None = None
    api_token: SecretStr | None = None

    @model_validator(mode="after")
    def validate_configuration(self):
        if bool(self.webhook_url) != bool(self.webhook_secret and self.webhook_secret.get_secret_value()):
            raise ValueError("configure both webhook URL and nonempty signing secret")
        for url in (self.comfy_url, self.webhook_url):
            if url and (url.username or url.password or url.query or url.fragment):
                raise ValueError("service URLs must not contain credentials, queries or fragments")
        if self.api_token is not None and not self.api_token.get_secret_value():
            raise ValueError("API token must be nonempty")
        return self

    @classmethod
    def from_env(cls):
        return cls.model_validate({
            name: os.environ[f"COMFYLORA_{name.upper()}"]
            for name in cls.model_fields if f"COMFYLORA_{name.upper()}" in os.environ
        })


def create_app(
    template: Graph, settings: Settings | None = None, nodes: Nodes = Nodes(), *,
    comfy_transport: httpx.AsyncBaseTransport | None = None,
    webhook_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings()
    validate_graph(template)
    render_workflow(template, Parameters(
        lora="startup-check.safetensors", width=512, height=512,
        seed=0, sampler_name="euler", steps=20,
    ), nodes)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with (
            httpx.AsyncClient(base_url=str(settings.comfy_url), timeout=10, transport=comfy_transport) as http,
            httpx.AsyncClient(timeout=5, transport=webhook_transport) as callbacks,
        ):
            client = ComfyClient(http, timeout=settings.render_timeout)
            webhook = None
            if settings.webhook_url and settings.webhook_secret:
                webhook = Webhook(callbacks, str(settings.webhook_url), settings.webhook_secret.get_secret_value())
            jobs = JobQueue(client, capacity=settings.queue_capacity, retention=settings.retention, webhook=webhook)
            app.state.jobs = jobs
            task = asyncio.create_task(jobs.run(), name="comfylora-worker")
            try:
                yield
            finally:
                try:
                    await asyncio.wait_for(jobs.pending.join(), timeout=10)
                except TimeoutError:
                    pass
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="ComfyLoRA Worker", version="0.1.0", lifespan=lifespan)

    def authorize(authorization: Annotated[str | None, Header()] = None) -> None:
        token = settings.api_token
        if token and not compare_digest(
            (authorization or "").encode(), f"Bearer {token.get_secret_value()}".encode(),
        ):
            raise HTTPException(401, "invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    protected = [Depends(authorize)]

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready", dependencies=protected)
    async def ready():
        jobs: JobQueue = app.state.jobs
        try:
            stats = await jobs.client.stats()
        except ComfyError:
            return JSONResponse({"status": "unavailable"}, status_code=503)
        return {
            "status": "ready",
            "queue_depth": jobs.pending.qsize(),
            "queue_capacity": jobs.pending.maxsize,
            "running": sum(job.status == "running" for job in jobs.active.values()),
            "devices": [
                {field: device[field] for field in ("name", "type", "vram_total", "vram_free",
                                                     "torch_vram_total", "torch_vram_free") if field in device}
                for device in stats.get("devices", []) if isinstance(device, dict)
            ] if isinstance(stats.get("devices"), list) else [],
        }

    @app.post("/jobs", status_code=202, dependencies=protected, response_model=Job)
    async def submit(parameters: Parameters):
        graph = render_workflow(template, parameters, nodes)
        try:
            return app.state.jobs.submit(graph)
        except asyncio.QueueFull as exc:
            raise HTTPException(429, "job queue full", headers={"Retry-After": "1"}) from exc

    def find_job(job_id: str) -> Job:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/jobs/{job_id}", dependencies=protected, response_model=Job)
    async def status(job_id: str):
        return find_job(job_id)

    @app.get("/jobs/{job_id}/images/{index}", dependencies=protected)
    async def image(job_id: str, index: int):
        job = find_job(job_id)
        if not 0 <= index < len(job.images):
            raise HTTPException(404, "image not found")
        stream = app.state.jobs.client.image_stream(job.images[index])
        try:
            response = await stream.__aenter__()
        except httpx.HTTPError as exc:
            raise HTTPException(502, "ComfyUI image unavailable") from exc
        if not response.is_success:
            await stream.__aexit__(None, None, None)
            raise HTTPException(502, "ComfyUI image unavailable")
        return StreamingResponse(
            response.aiter_bytes(), media_type="application/octet-stream",
            headers={"Content-Disposition": "attachment", "X-Content-Type-Options": "nosniff"},
            background=BackgroundTask(stream.__aexit__, None, None, None),
        )

    return app


def from_env() -> FastAPI:
    path = Path(os.environ.get("COMFYLORA_WORKFLOW", "examples/product_sku_workflow.json"))
    return create_app(json.loads(path.read_text(encoding="utf-8")), Settings.from_env())
