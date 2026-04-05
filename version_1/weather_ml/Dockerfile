FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir influxdb-client

# Copy source code
COPY src/ src/
COPY models/ models/
COPY data/ data/

# Create output dirs
RUN mkdir -p models logs

ENV PYTHONUNBUFFERED=1

WORKDIR /app/src

CMD ["python", "scheduler.py"]
