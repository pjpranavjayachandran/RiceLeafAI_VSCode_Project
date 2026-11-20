# RiceLeaf AI

Advanced Machine Learning project by Pranav Jayachandran. The app previews an uploaded rice image and returns the top class and scores from a trained dense artificial neural network.

## Run in VS Code on Windows

Use 64-bit Python 3.12. Extract the ZIP, open the `RiceLeafAI` folder in VS Code and run these commands in the PowerShell terminal. Calling the virtual environment's Python directly avoids the need to activate a PowerShell script.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy the four class folders from your multi-crop dataset into `data`. Use JPEG, PNG or WebP files. If your four classes are already together inside another folder, pass that folder directly using `--data "D:\path to\dataset"`.

```text
data/
  Rice___Brown_Spot/
  Rice___Healthy/
  Rice___Leaf_Blast/
  Rice___Neck_Blast/
```

Train and evaluate, then start the app from the same terminal.

```powershell
.\.venv\Scripts\python.exe train.py --data data --epochs 60
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and upload a photo. The app works locally after dependencies are installed and the model is trained. Restart the server after retraining to load the new model. The API expects raw image bytes in a POST to `/api/predict` and returns JSON. Only load model files you produced or trust.

## ANN and baseline

The image is resized to 64 × 64 pixels and divided into 16 tiles. Each tile contributes six colour statistics and eight edge-orientation values. This produces 224 inputs for both models.

- **Dense ANN** uses 224 inputs, two hidden layers (128 and 64 ReLU neurons), and four softmax outputs. Cross-entropy measures classification loss, backpropagation computes gradients, and Adam updates the weights. The network returns one score per class. The 60% top-score warning is a UI heuristic. Scores are uncalibrated and an unrelated image can still receive a high score.
- **Random Forest** trains on the same features and split as a classical ML baseline. The app uses the ANN to demonstrate neural network inference.

## Training and evaluation

The script removes duplicates with identical decoded pixels and uses a reproducible stratified 60/20/20 train/validation/test split. The scaler learns from training data. Validation macro F1 selects the ANN checkpoint, then the script reports held-out test metrics for both models. Macro F1 weights each class equally.

Prepare a flat class-folder dataset for this script. If your dataset includes official split assignments, or multiple photos from the same plant, preserve those assignments or use a plant-group split before reporting results. This pipeline randomly splits individual images and does not detect near-duplicates. Use a separate field-image test set to assess real-world performance.

The `artifacts` directory contains the trained ANN and scaler, `metrics.json` with per-class precision, recall and F1, two confusion matrices, ANN learning curves, the training history and a split manifest. Capture a screenshot with an unseen image for the demo. Record the dataset source, class counts and actual test metrics in your submission.

The ZIP contains source code. The labeled images and trained model are supplied by your training run. Until then the app shows “Training required.”
