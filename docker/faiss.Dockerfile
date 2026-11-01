# Build context is ./retrieval (see docker-compose.yaml), so faiss_service.py
# is at the context root. Only the three libraries the service needs.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir faiss-cpu numpy flask

COPY faiss_service.py .

CMD ["python", "faiss_service.py"]
