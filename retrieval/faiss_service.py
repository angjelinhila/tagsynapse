"""Minimal FAISS search service. Runs inside the faiss-service container
(docker/faiss.Dockerfile) with no dependency on the rest of the repo, so the
index path is read straight from the environment rather than from
tagsynapse.config - keep the default here and in config.py in sync.

    POST /search  {"vector": [...], "k": 5}  ->  {"indices": [[...]], "distances": [[...]]}
"""
import faiss
import numpy as np
import os
import time
from flask import Flask, request, jsonify

# Inside the container ./faiss/index is mounted read-only at /mnt/faiss_index.
FAISS_INDEX_PATH = os.getenv(
    "TAGSYNAPSE_FAISS_INDEX_PATH",
    "/mnt/faiss_index/artificial_filtered_output_faiss.index",
)


app = Flask(__name__)

# Wait briefly for the index file to appear (in case the volume mount is slow)
while not os.path.exists(FAISS_INDEX_PATH):
    print(f"Waiting for FAISS index at {FAISS_INDEX_PATH} ...")
    time.sleep(2)

print(f"Loading FAISS index from {FAISS_INDEX_PATH} ...")
index = faiss.read_index(FAISS_INDEX_PATH)
DIM = index.d
print(f"FAISS index loaded. Dimension = {DIM}, ntotal = {index.ntotal}")

@app.route("/search", methods=["POST"])
def search():
    payload = request.get_json(force=True) or {}
    vec = payload.get("vector")
    k = int(payload.get("k", 5))

    if vec is None:
        return jsonify({"error": "Missing 'vector' in JSON body"}), 400

    q = np.array(vec, dtype="float32").reshape(1, -1)

    if q.shape[1] != DIM:
        return jsonify({
            "error": "Query vector has wrong dimension",
            "expected": DIM,
            "got": int(q.shape[1]),
        }), 400

    distances, indices = index.search(q, k)
    return jsonify({
        "indices": indices.tolist(),
        "distances": distances.tolist(),
    })

if __name__ == "__main__":
    # Expose on all interfaces so the host / other containers can call it
    app.run(host="0.0.0.0", port=5000)
