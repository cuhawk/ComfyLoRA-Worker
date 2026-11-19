"""Bounded process-local FIFO and terminal webhook delivery."""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import hmac
import logging
from typing import Literal
from uuid import uuid4

import httpx
from pydantic import BaseModel, Field

from .comfy import ComfyClient, Image
from .workflow import Graph

logger = logging.getLogger(__name__)
Status = Literal["queued", "running", "succeeded", "failed"]
WebhookStatus = Literal["disabled", "pending", "delivered", "failed"]


class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    status: Status = "queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    images: list[Image] = Field(default_factory=list)
    error: str | None = None
    webhook_status: WebhookStatus = "disabled"


@dataclass(frozen=True)
class Webhook:
    http: httpx.AsyncClient
    url: str
    secret: str
    attempts: int = 3
    retry_delay: float = 0.5

    async def deliver(self, job: Job) -> bool:
        payload = job.model_dump_json(exclude={"webhook_status"}).encode()
        signature = hmac.new(self.secret.encode(), payload, hashlib.sha256).hexdigest()
        for attempt in range(self.attempts):
            try:
                response = await self.http.post(self.url, content=payload, headers={
                    "Content-Type": "application/json",
                    "X-ComfyLoRA-Event-ID": job.id,
                    "X-ComfyLoRA-Signature": f"sha256={signature}",
                }, follow_redirects=False)
            except httpx.RequestError:
                pass
            else:
                if response.is_success:
                    return True
                if response.status_code not in {408, 429} and response.status_code < 500:
                    return False
            if attempt + 1 < self.attempts:
                await asyncio.sleep(self.retry_delay * 2**attempt)
        return False


class JobQueue:
    def __init__(
        self, client: ComfyClient, *, capacity: int = 32, retention: int = 256,
        webhook: Webhook | None = None,
    ):
        self.client = client
        self.webhook = webhook
        self.pending: asyncio.Queue[tuple[Job, Graph]] = asyncio.Queue(maxsize=capacity)
        self.active: dict[str, Job] = {}
        self.completed: OrderedDict[str, Job] = OrderedDict()
        self.retention = retention

    def submit(self, graph: Graph) -> Job:
        job = Job(webhook_status="pending" if self.webhook else "disabled")
        self.pending.put_nowait((job, graph))
        self.active[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self.active.get(job_id) or self.completed.get(job_id)

    async def run(self) -> None:
        while True:
            job, graph = await self.pending.get()
            try:
                await self._process(job, graph)
            finally:
                self.pending.task_done()

    async def _process(self, job: Job, graph: Graph) -> None:
        job.status = "running"
        job.started_at = datetime.now(UTC)
        try:
            job.images = await self.client.render(graph)
            job.status = "succeeded"
        except asyncio.CancelledError:
            job.status = "failed"
            job.error = "worker stopped during render"
            if self.webhook:
                job.webhook_status = "failed"
            raise
        except Exception:
            logger.exception("Render failed for job %s", job.id)
            job.status = "failed"
            job.error = "ComfyUI render failed; see worker logs"
        finally:
            job.finished_at = datetime.now(UTC)
            del self.active[job.id]
            self.completed[job.id] = job
            if len(self.completed) > self.retention:
                self.completed.popitem(last=False)

        if self.webhook is not None:
            try:
                job.webhook_status = "delivered" if await self.webhook.deliver(job) else "failed"
            finally:
                if job.webhook_status == "pending":
                    job.webhook_status = "failed"
