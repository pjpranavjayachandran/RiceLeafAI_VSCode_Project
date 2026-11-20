import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app import ADVANCED, BASELINE, load_model
from train import CLASSES
from train_advanced import predict_scores
from vision import features, load_rgb


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
WEB = ROOT / "web"
PAGE = WEB / "index.html"
LABELS = list(CLASSES.values())
SPLITS = ("train", "validation", "test")


def get_predictor():
    if ADVANCED.is_file():
        bundle = load_model(ADVANCED)
        name = bundle["winner"]
        classes = np.asarray(bundle["labels"])

        def predict(batch):
            return predict_scores(name, bundle["models"], batch)

        return name, classes, predict

    if BASELINE.is_file():
        bundle = load_model(BASELINE)
        network = bundle["network"]

        def predict(batch):
            scaled = bundle["scaler"].transform(batch)
            return network.predict_proba(scaled)

        return "Dense ANN", network.classes_, predict

    raise SystemExit("No trained model found in artifacts. Run train.py first.")


def split_metrics(rows):
    if not rows:
        return None

    actual = [row["expected"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, labels=LABELS, average="macro", zero_division=0
    )
    return {
        "count": len(rows),
        "correct": sum(a == b for a, b in zip(actual, predicted)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classes": classification_report(
            actual, predicted, labels=LABELS, output_dict=True, zero_division=0
        ),
        "matrix": confusion_matrix(actual, predicted, labels=LABELS).tolist(),
    }


def audit():
    manifest = json.loads((ART / "splits.json").read_text(encoding="utf-8"))
    membership = {}
    for split in SPLITS:
        for filename in manifest[split]:
            key = Path(filename).as_posix()
            if key in membership:
                raise SystemExit(f"Duplicate split assignment: {key}")
            membership[key] = split

    name, classes, predict = get_predictor()
    rows, pending, vectors = [], [], []
    found = set()

    def flush():
        if not vectors:
            return
        scores = predict(np.stack(vectors))
        for row, probabilities in zip(pending, scores):
            best = int(np.argmax(probabilities))
            row["predicted"] = str(classes[best])
            row["top_score"] = round(float(probabilities[best]), 6)
            row["outcome"] = (
                "correct" if row["expected"] == row["predicted"] else "incorrect"
            )
        vectors.clear()
        pending.clear()

    for folder, label in CLASSES.items():
        directory = DATA / folder
        if not directory.is_dir():
            raise SystemExit(f"Missing class folder: {directory}")

        print(f"Scanning {label}...", flush=True)
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            key = path.relative_to(DATA).as_posix()
            found.add(key)
            row = {
                "file": key, "split": membership.get(key, "outside manifest"),
                "expected": label, "predicted": "", "top_score": "", "outcome": "",
            }
            rows.append(row)
            try:
                vector = features(load_rgb(path))
            except (OSError, ValueError, Image.DecompressionBombError):
                row["outcome"] = "unreadable or unsupported"
                continue
            vectors.append(vector)
            pending.append(row)
            if len(vectors) >= 64:
                flush()
    flush()

    missing = sorted(set(membership) - found)
    completed = [r for r in rows if r["predicted"]]
    if not completed:
        raise SystemExit("No readable photos found.")

    report = {
        "model": name,
        "files_found": len(rows),
        "predicted": len(completed),
        "skipped": len(rows) - len(completed),
        "outside": sum(r["split"] == "outside manifest" for r in completed),
        "missing": len(missing),
        "manifest_counts": {s: len(manifest[s]) for s in SPLITS},
        "labels": LABELS,
        "splits": {
            s: split_metrics([r for r in completed if r["split"] == s])
            for s in SPLITS
        },
        "photo_counts": dict(Counter(r["expected"] for r in completed)),
    }

    csv_path = ART / "all_photo_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (ART / "dataset_audit.json").write_text(
        json.dumps({**report, "missing_files": missing}, indent=2),
        encoding="utf-8",
    )
    return report


STYLE = """
<style id="full-dashboard-style">
  .wrap { width: 100%; max-width: none; margin: 0; }
  header, nav, main, footer {
    padding-inline: clamp(18px, 4vw, 76px);
  }
  header { min-height: 68px; background: #fafbf4; }
  .hero {
    box-sizing: border-box;
    width: 100%;
    min-height: calc(100svh - 68px);
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: flex-start;
    padding: clamp(28px, 8vw, 140px);
    border-radius: 0;
    background:
      radial-gradient(ellipse at 95% 15%, #67916a 0%, transparent 42%),
      repeating-linear-gradient(154deg, transparent 0 46px,
        rgba(205,230,152,.055) 47px 49px, transparent 50px 93px),
      linear-gradient(130deg, #092e24, #194637 62%, #315844);
  }
  .hero h1 { font-size: clamp(3.3rem, 8vw, 8rem); max-width: 12ch; }
  .hero p { font-size: clamp(1rem, 1.5vw, 1.35rem); max-width: 46ch; }
  .hero a { padding: 13px 22px; }
  .hero .hero-actions { display: flex; flex-wrap: wrap; gap: 10px; }
  .hero .atlas-link {
    color: white; background: transparent; border: 1px solid #b8ceba;
  }
  nav { margin: 0; padding-block: 14px; }
  .card { padding: clamp(20px, 3vw, 44px); }
  #map { height: clamp(320px, 55vh, 620px); }
  .audit-stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(145px,1fr));
    gap: 10px; margin-block: 16px;
  }
  .audit-stat { background: #edf3e5; padding: 16px; border-radius: 12px; }
  .audit-stat strong { display: block; font-size: 1.8rem; color: #184432; }
  #dataset-audit h3 { margin-top: 22px; }
  #dataset-audit select { padding: 7px; font: inherit; }
  #dataset-audit td, #dataset-audit th { padding: 10px; }
</style>
"""

AUDIT_SECTION = """
<section id="dataset-audit" class="card">
  <h2>Dataset and model performance</h2>
  <p id="audit-summary"></p>
  <div class="audit-stats" id="audit-stats"></div>
  <p id="audit-coverage"></p>

  <h3>How well does the model perform?</h3>
  <p>Training measures fit to learned examples. Validation guides model
     selection. The held-out test split is the main performance estimate.</p>
  <div class="scroll" id="audit-table"></div>
  <details>
    <summary>What do the metrics mean?</summary>
    <p><b>Accuracy</b> counts how many predictions were correct overall.
       <b>Precision</b> asks how often a predicted disease label was correct.
       <b>Recall</b> asks how many actual cases of a disease were detected.
       <b>F1</b> combines precision and recall. Macro averages give each class
       equal weight.</p>
  </details>

  <h3>Disease-by-disease results</h3>
  <label>Choose a split <select id="audit-split">
    <option value="test">Held-out test</option>
    <option value="validation">Validation</option>
    <option value="train">Training</option>
  </select></label>
  <div class="scroll" id="audit-classes"></div>
  <h3>Which diseases were confused?</h3>
  <div class="scroll" id="audit-matrix"></div>
</section>
"""

AUDIT_JS = """
<script>
(() => {
  const report = JSON.parse(document.getElementById("audit-data").textContent);
  const get = id => document.getElementById(id);
  const percent = n => (100 * n).toFixed(2) + "%";
  const names = {train:"Training", validation:"Validation", test:"Held-out test"};

  function table(id, headings, rows) {
    const element = document.createElement("table");
    const heading = element.createTHead().insertRow();
    for (const title of headings) {
      const cell = document.createElement("th");
      cell.textContent = title;
      heading.append(cell);
    }
    const body = element.createTBody();
    for (const values of rows) {
      const row = body.insertRow();
      for (const value of values) row.insertCell().textContent = value;
    }
    get(id).replaceChildren(element);
  }

  get("audit-summary").textContent =
    "Saved full-dataset scan using " + report.model + ".";
  const test = report.splits.test;
  const stats = [
    ["Files found", report.files_found],
    ["Photos analyzed", report.predicted],
    ["Test accuracy", test ? percent(test.accuracy) : "Unavailable"],
    ["Test macro precision", test ? percent(test.precision) : "Unavailable"]
  ];
  for (const [label, value] of stats) {
    const box = document.createElement("div");
    box.className = "audit-stat";
    const strong = document.createElement("strong");
    strong.textContent = value;
    box.append(strong, document.createTextNode(label));
    get("audit-stats").append(box);
  }
  get("audit-coverage").textContent =
    "Unreadable or unsupported files: " + report.skipped +
    ". Analyzed photos outside the original split manifest: " + report.outside +
    ". Missing manifest files: " + report.missing + ".";

  const rows = [];
  for (const key of ["train","validation","test"]) {
    const m = report.splits[key];
    if (m) rows.push([names[key], m.count + " / " + report.manifest_counts[key],
      m.correct, percent(m.accuracy), percent(m.precision),
      percent(m.recall), percent(m.f1)]);
  }
  table("audit-table",
    ["Split","Scanned / assigned","Correct","Accuracy","Macro precision","Macro recall","Macro F1"],
    rows);

  function drawClassResults() {
    const m = report.splits[get("audit-split").value];
    if (!m) {
      get("audit-classes").textContent = "No readable images in this split.";
      get("audit-matrix").replaceChildren();
      return;
    }
    table("audit-classes", ["Class","Photos","Precision","Recall","F1"],
      report.labels.map(label => {
        const entry = m.classes[label];
        return [label, entry.support, percent(entry.precision),
          percent(entry.recall), percent(entry["f1-score"])];
      }));
    table("audit-matrix", ["Actual ↓ / Predicted →", ...report.labels],
      m.matrix.map((row, i) => [report.labels[i], ...row]));
    const cells = get("audit-matrix").querySelector("tbody").rows;
    m.matrix.forEach((row, i) => {
      const total = row.reduce((a,b) => a+b, 0);
      row.forEach((value, j) => {
        if (!value) return;
        const color = i === j ? "99,166,102" : "224,130,82";
        cells[i].cells[j+1].style.background =
          "rgba(" + color + "," + (0.12 + 0.55*value/(total || 1)) + ")";
      });
    });
  }
  get("audit-split").onchange = drawClassResults;
  drawClassResults();
})();
</script>
"""


def update_page(report):
    html = PAGE.read_text(encoding="utf-8-sig")

    # Remove our earlier generated blocks so repeating the command is safe.
    start = "<!-- RICE-AUDIT-START -->"
    end = "<!-- RICE-AUDIT-END -->"
    while start in html:
        before, rest = html.split(start, 1)
        if end not in rest:
            raise SystemExit("Incomplete dashboard block in index.html. Restore its backup.")
        html = before + rest.split(end, 1)[1]

    def block(content):
        return start + content + end

    html = html.replace("</head>", block(STYLE) + "\n</head>", 1)
    html = html.replace(
        '<section id="evaluation"',
        block(AUDIT_SECTION) + '\n<section id="evaluation"', 1
    )
    html = html.replace(
        '<a href="#evaluation">Evaluation</a>',
        '<a href="#dataset-audit">Dataset analysis</a>'
        '<a href="#evaluation">Compare models</a>', 1
    )
    html = html.replace(
        '<a href="#scanner">Scan a photo ↓</a>',
        '<div class="hero-actions"><a href="#scanner">Scan a photo ↓</a>'
        '<a class="atlas-link" href="#atlas">Explore the rice atlas</a></div>', 1
    )

    encoded = json.dumps(report, ensure_ascii=True).replace("<", "\\u003c")
    script = '<script type="application/json" id="audit-data">' + encoded + "</script>"
    html = html.replace("</body>", block(script + AUDIT_JS) + "\n</body>", 1)

    backup = WEB / "index.before-dashboard.html"
    if not backup.exists():
        shutil.copy2(PAGE, backup)
    temp = WEB / "index.dashboard.tmp"
    temp.write_text(html, encoding="utf-8")
    temp.replace(PAGE)


if __name__ == "__main__":
    if not PAGE.exists():
        raise SystemExit("Missing web/index.html. Run this from the RiceLeafAI project.")

    current = PAGE.read_text(encoding="utf-8-sig")
    required = ['class="hero"', 'id="scanner"', '<section id="evaluation"', "</head>", "</body>"]
    if not all(item in current for item in required):
        raise SystemExit(
            "This installer expects the complete 'Read the rice leaf.' index.html "
            "from the previous message. Your current page has a different structure."
        )

    report = audit()
    update_page(report)

    print("\nDashboard updated.")
    print(f"Active model: {report['model']}")
    print(f"Files found: {report['files_found']}")
    print(f"Photos analyzed: {report['predicted']}")
    print(f"Skipped: {report['skipped']}")
    print("Per-photo results: artifacts/all_photo_predictions.csv")
    print("Full analysis: artifacts/dataset_audit.json")
    print("Original page backup: web/index.before-dashboard.html")