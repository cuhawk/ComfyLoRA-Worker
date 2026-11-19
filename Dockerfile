FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir . && useradd --system --uid 10001 worker
COPY examples/ examples/
USER 10001
EXPOSE 8000
ENTRYPOINT ["comfylora"]
CMD ["serve", "--host", "0.0.0.0", "--workflow", "examples/product_sku_workflow.json"]
