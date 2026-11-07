FROM python:3.12-slim
WORKDIR /opt/project
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY agent agent
COPY api api
COPY training training
COPY scripts scripts
COPY params.yaml ./
RUN useradd --uid 10001 --create-home app
USER 10001
EXPOSE 8000 8001
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

