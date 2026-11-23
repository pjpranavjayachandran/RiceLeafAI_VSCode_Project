"""FastAPI backend and same-origin browser interface."""
from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
from PIL import Image

from vision import features, load_rgb

ROOT = Path(__file__).resolve().parent
MODEL = Path(os.getenv("RICE_MODEL_PATH", ROOT / "artifacts" / "rice_ann.joblib"))
MAX_BYTES = 8 * 1024 * 1024
app = FastAPI(title="RiceLeaf AI")
app.mount("/assets", StaticFiles(directory=ROOT / "web"), name="assets")


@lru_cache(maxsize=1)
def model():
    return joblib.load(MODEL)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/status")
def status():
    return {"ready": MODEL.is_file()}


@app.post("/api/predict")
async def predict(request: Request):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "Choose an image smaller than 8 MB.")
        data.extend(chunk)
    try:
        image = load_rgb(BytesIO(data))
    except (OSError, ValueError, Image.DecompressionBombError) as error:
        raise HTTPException(422, "Choose a readable JPEG, PNG or WebP under 24 megapixels.") from error
    if not MODEL.is_file():
        raise HTTPException(503, "Train the ANN before analyzing an image.")
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(score_image, image)


def score_image(image):
    bundle = model()
    network = bundle["network"]
    vector = bundle["scaler"].transform(features(image).reshape(1, -1))
    scores = network.predict_proba(vector)[0]
    ranking = sorted(zip(network.classes_, scores), key=lambda pair: -pair[1])
    return {"prediction": ranking[0][0], "low_score": bool(ranking[0][1] < 0.60),
            "scores": [{"label": label, "score": float(score)} for label, score in ranking]}
