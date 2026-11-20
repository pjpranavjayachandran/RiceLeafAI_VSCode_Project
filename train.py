"""Train a dense ANN and Random Forest with shared, stratified splits."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from vision import features, load_rgb

CLASSES = {
    "Rice___Brown_Spot": "Brown Spot", "Rice___Healthy": "Healthy",
    "Rice___Leaf_Blast": "Leaf Blast", "Rice___Neck_Blast": "Neck Blast",
}
SEED = 42


def read_dataset(root):
    vectors, labels, paths = [], [], []
    seen = {}
    skipped = duplicates = 0
    for folder, label in CLASSES.items():
        directory = root / folder
        if not directory.is_dir():
            raise ValueError(f"Missing class folder: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            try:
                image = load_rgb(path)
            except (OSError, ValueError, Image.DecompressionBombError):
                skipped += 1
                continue
            digest = sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            if digest in seen:
                if seen[digest] != label:
                    raise ValueError(f"Identical images have conflicting labels: {path}")
                duplicates += 1
                continue
            seen[digest] = label
            vectors.append(features(image))
            labels.append(label)
            paths.append(path.relative_to(root).as_posix())
    counts = Counter(labels)
    if any(counts[label] < 10 for label in CLASSES.values()):
        raise ValueError(f"Use at least 10 unique images per class. Found {dict(counts)}")
    return np.stack(vectors), np.asarray(labels), paths, skipped, duplicates


def fit_ann(x_train, y_train, x_val, y_val, epochs):
    from copy import deepcopy

    network = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                            solver="adam", random_state=SEED)
    best_score = -1
    history = []
    for epoch in range(1, epochs + 1):
        network.partial_fit(x_train, y_train, classes=np.unique(y_train))
        score = f1_score(y_val, network.predict(x_val), average="macro", zero_division=0)
        history.append({"epoch": epoch, "training_loss": float(network.loss_),
                        "validation_macro_f1": float(score)})
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_network = deepcopy(network)
        if epoch - best_epoch >= 12:
            break
    return best_network, best_epoch, history


def train(data, output, epochs):
    x, y, paths, skipped, duplicates = read_dataset(data)
    training, remainder = train_test_split(np.arange(len(y)), test_size=0.4,
                                           stratify=y, random_state=SEED)
    validation, test = train_test_split(remainder, test_size=0.5,
                                       stratify=y[remainder], random_state=SEED)
    scaler = StandardScaler().fit(x[training])
    x = scaler.transform(x)
    ann, best_epoch, history = fit_ann(x[training], y[training], x[validation], y[validation], epochs)
    forest = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                    random_state=SEED, n_jobs=-1)
    forest.fit(x[training], y[training])
    report = {
        "seed": SEED, "class_counts": dict(Counter(y)), "invalid_images": skipped,
        "exact_duplicates_removed": duplicates, "best_ann_epoch": best_epoch,
        "splits": {name: dict(Counter(y[indices])) for name, indices in
                   (("train", training), ("validation", validation), ("test", test))},
        "models": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        staging = Path(temp)
        for name, estimator in (("ann", ann), ("random_forest", forest)):
            on_val = estimator.predict(x[validation])
            prediction = estimator.predict(x[test])
            report["models"][name] = {
                "validation_macro_f1": f1_score(y[validation], on_val, average="macro", zero_division=0),
                "test_accuracy": accuracy_score(y[test], prediction),
                "test_macro_f1": f1_score(y[test], prediction, average="macro", zero_division=0),
                "test_classification_report": classification_report(
                    y[test], prediction, output_dict=True, zero_division=0),
            }
            display = ConfusionMatrixDisplay.from_predictions(
                y[test], prediction, xticks_rotation=25, cmap="Greens")
            display.ax_.set_title(f"{name.replace('_', ' ').upper()} · Held-out test")
            display.figure_.tight_layout()
            display.figure_.savefig(staging / f"{name}_confusion.png", dpi=150)
            plt.close(display.figure_)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for axis, key, title in zip(axes, ("training_loss", "validation_macro_f1"),
                                    ("Training loss", "Validation macro F1")):
            axis.plot([item["epoch"] for item in history], [item[key] for item in history])
            axis.axvline(best_epoch, color="green", linestyle="--")
            axis.set(title=title, xlabel="Epoch")
        fig.tight_layout()
        fig.savefig(staging / "ann_learning_curves.png", dpi=150)
        plt.close(fig)
        manifest = {name: [paths[i] for i in indices] for name, indices in
                    (("train", training), ("validation", validation), ("test", test))}
        (staging / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (staging / "splits.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (staging / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        joblib.dump({"network": ann, "scaler": scaler}, staging / "rice_ann.joblib")
        for path in sorted(staging.iterdir(), key=lambda file: file.suffix == ".joblib"):
            path.replace(output / path.name)
    print(json.dumps(report["models"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    try:
        train(args.data, args.output, args.epochs)
    except ValueError as error:
        parser.exit(2, f"{error}\n")
