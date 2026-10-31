FROM python:3.9

WORKDIR /app

# Only the tiny libraries needed for FAISS search
RUN pip install --no-cache-dir faiss-cpu numpy flask

COPY faiss_service.py .

CMD ["python", "faiss_service.py"]
