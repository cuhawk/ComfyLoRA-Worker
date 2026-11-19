# ComfyLoRA-Worker

Headless FastAPI + async WebSocket queue worker for production **ComfyUI** pipelines, parameterized `.safetensors` LoRA workflow injection (`LoraLoader`), safetensors metadata verification, VRAM telemetry, and verified Colab GPU LoRA training benchmarks.

## Quick start

Install with Python 3.12 or newer. Run ComfyUI separately, with the checkpoint
referenced by the workflow under `models/checkpoints` and the requested LoRA
under `models/loras`.

```sh
python -m pip install '.[test]'
python -m pytest
export COMFYLORA_COMFY_URL=http://127.0.0.1:8188
comfylora serve --workflow examples/product_sku_workflow.json
```

Save the following as `params.json`.

```json
{
  "lora": "catalog/mug.safetensors",
  "strength_model": 0.8,
  "strength_clip": 0.6,
  "seed": 42,
  "width": 768,
  "height": 512,
  "sampler_name": "euler",
  "steps": 24,
  "positive": "studio product photo of a blue ceramic mug, SKU MUG-042"
}
```

```sh
comfylora render-workflow examples/product_sku_workflow.json params.json -o rendered.json
comfylora inspect-lora /path/to/mug.safetensors
curl --json @params.json http://127.0.0.1:8000/jobs
```

The graph must be a ComfyUI API-format export. The built-in roles are KSampler
`3`, CheckpointLoaderSimple `4`, EmptyLatentImage `5`, positive and negative
CLIPTextEncode `6` and `7`, and LoraLoader `10`. Both `serve` and
`render-workflow` accept `--nodes` pointing to a JSON object with optional
`sampler`, `checkpoint`, `latent`, `positive`, `negative`, and `lora` ID
overrides. The selected sampler, latent and prompt encoders are wired to the
configured roles. The loader uses the checkpoint's model and CLIP outputs.
Omit prompt or scheduler overrides to keep the template values. Valid sampler
and scheduler names depend on the upstream ComfyUI installation.

The inspector validates the safetensors structure and reports metadata, tensor
names, shapes, dtypes and the full-file SHA-256. Pass `--sha256 DIGEST` to
verify an expected hash. A compatible report indicates at least one
rank-compatible pair using a key convention recognized by ComfyUI's LoRA
adapter. Actual target mapping and model compatibility depend on the selected
checkpoint. The command exits with 1 for a file with no recognized pairs and
2 for read, format or hash errors.

## HTTP API

| Endpoint | Behavior |
| --- | --- |
| `POST /jobs` | Validates parameters and enqueues a render. Returns 202 with a job ID, 422 for invalid parameters or 429 when full. |
| `GET /jobs/{id}` | Returns queued, running, succeeded or failed status with timestamps, image references and webhook delivery status. |
| `GET /jobs/{id}/images/{index}` | Streams a zero-based output image from ComfyUI as a download. |
| `GET /health` | Worker liveness. |
| `GET /ready` | ComfyUI connectivity, queue utilization and per-device VRAM in bytes when reported by `/system_stats`. Returns 503 if ComfyUI cannot be reached. |

If a token is configured, include `Authorization: Bearer TOKEN` on readiness
and job requests. OpenAPI is available at `/docs`. ComfyUI receives the
graph at `/prompt`, WebSocket completion events wake history checks, and
`/history/{prompt_id}` supplies output references. The worker also polls
history when no completion event arrives.

| Environment variable | Default |
| --- | --- |
| `COMFYLORA_COMFY_URL` | `http://127.0.0.1:8188` |
| `COMFYLORA_API_TOKEN` | Unset. Set before exposing the worker to other hosts. |
| `COMFYLORA_QUEUE_CAPACITY` | 32 pending jobs, plus one job being processed |
| `COMFYLORA_RETENTION` | 256 terminal jobs |
| `COMFYLORA_RENDER_TIMEOUT` | 300 seconds from start of processing |
| `COMFYLORA_WEBHOOK_URL` | Unset. Operator-configured callback destination. |
| `COMFYLORA_WEBHOOK_SECRET` | Unset. Required with webhook URL. |

Each terminal render sends a JSON callback to the configured URL when enabled.
Verify `X-ComfyLoRA-Signature` as `sha256=` followed by the hex HMAC-SHA256
of the raw body using the signing secret. Deduplicate retries using
`X-ComfyLoRA-Event-ID`. Transport failures, HTTP 408, 429 and server errors
are retried up to three total attempts. A callback failure is recorded
independently of the render status.

The queue and job history are process-local and clear on restart. Run one
worker process per dedicated ComfyUI instance for serialized renders. Image
downloads depend on upstream file retention. A render timeout ends tracking in
this worker and the upstream prompt can still finish. The worker waits up to
10 seconds for outstanding work on shutdown.

## Container

```sh
docker build -t comfy-lora-worker .
docker run --rm -p 127.0.0.1:8000:8000 --network comfy \
  -e COMFYLORA_COMFY_URL=http://comfyui:8188 comfy-lora-worker
```

Use the Docker network and hostname of your ComfyUI deployment in the example.
The image uses the bundled workflow by default and runs as a non-root user.

## Verified Google Colab GPU Benchmark

Executed cleanly on **NVIDIA L4 (22.03 GB VRAM, `torch.bfloat16`)** via Google Colab Pro+ (telemetry stored in [`benchmarks/colab_gpu_metrics.json`](benchmarks/colab_gpu_metrics.json)).

| Metric | Value |
| :--- | :--- |
| **Backbone Model** | `stabilityai/sd-turbo` |
| **LoRA Config & Params** | `r=16`, `alpha=32`, `3,319,808` trainable UNet attention params |
| **40-Step LoRA Loss** | `0.12345` -> `0.03738` (`7.46s` training time) |
| **Exported `.safetensors`** | `256` `lora_unet_*` FP16 tensors (`6,678,680` bytes, ComfyUI `LoraLoader` verified: `True`) |
| **Inference Latency & VRAM** | **`240.1 ms/image`** (`2`-step inference) @ `3655.9 MB` peak VRAM |
| **Adapter SHA-256** | `77b781de001b39e7c746ae2ece4397f3833ccfb14c558fb4d39f29a8e1eba30f` |
