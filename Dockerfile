FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY ml_pipline/ ./ml_pipline/
COPY makefile ./makefile

RUN mkdir -p /tmp && chmod 777 /tmp
EXPOSE 8000
EXPOSE 5000