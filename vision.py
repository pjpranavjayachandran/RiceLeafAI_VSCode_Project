"""Shared image processing for training and prediction."""
import numpy as np
from PIL import Image, ImageOps

FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_PIXELS = 24_000_000


def load_rgb(source):
    with Image.open(source) as image:
        if image.format not in FORMATS or image.width * image.height > MAX_PIXELS:
            raise ValueError("Use a JPEG, PNG or WebP image under 24 megapixels.")
        return ImageOps.exif_transpose(image).convert("RGB")


def features(image):
    rgb = np.asarray(image.resize((64, 64), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    gray = rgb.mean(axis=2)
    dy, dx = np.gradient(gray)
    magnitude = np.hypot(dy, dx)
    direction = (np.arctan2(dy, dx) + np.pi) % np.pi
    vector = []
    # Keep local colour and edge patterns from a 4-by-4 grid.
    for row in range(0, 64, 16):
        for col in range(0, 64, 16):
            tile = np.s_[row:row + 16, col:col + 16]
            pixels = rgb[tile]
            edges, _ = np.histogram(direction[tile], bins=8, range=(0, np.pi),
                                     weights=magnitude[tile])
            vector.extend(pixels.mean(axis=(0, 1)))
            vector.extend(pixels.std(axis=(0, 1)))
            vector.extend(edges / (edges.sum() + 1e-6))
    return np.asarray(vector, dtype=np.float32)
