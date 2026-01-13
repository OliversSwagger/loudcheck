<<<<<<< HEAD
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
=======
# Stage 1: Build dependencies
FROM python:3.14.2 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install --upgrade pip
RUN .venv/bin/pip install -r requirements.txt

# Stage 2: Final image
FROM python:3.14.2-slim
WORKDIR /app

COPY --from=builder /app/.venv .venv/
COPY . .

# Make uploads dir
RUN mkdir -p uploads

# Use Uvicorn to start FastAPI on 0.0.0.0:$PORT
ENV PORT=8000
CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> 54cf6f5 (Add deployment files)
