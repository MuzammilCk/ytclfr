# YTCLFR — Model Training Guide

This guide documents how to collect labeled training data and train the YTCLFR text and frame classifiers.

---

## Step 1: Label Training Data

After the system has processed videos, samples are automatically saved to `backend/training_data/` with a `human_label: null` field.

Label them using the built-in admin labeling tool:

1. Log in as an admin user at the frontend
2. Click the **🏷️ Label** button in the top navigation bar
3. For each sample, review the video title, AI prediction, transcript preview, and OCR text
4. Click the correct category button (listicle, music, educational, comedy, shopping, recipe, documentary, tutorial, news, other)
5. Use **Skip →** to skip uncertain samples

---

## Step 2: Export Labeled Data

Export as CSV via the API:

```bash
curl -H "Authorization: Bearer <your_admin_token>" \
  http://localhost:8000/api/v1/admin/training-data/export \
  -o training_data.csv
```

Or click **⬇ Export CSV** in the Labeling Tool UI.

The CSV has columns: `sample_id`, `video_title`, `predicted_category`, `confidence`, `human_label`, `transcript_preview`, `ocr_preview`.

---

## Step 3: Train the Text Classifier

Requires: Python 3.11, PyTorch, ~4GB RAM (CPU), ~1GB GPU VRAM (GPU)

```bash
# From project root
cd backend
python scripts/train_text_classifier.py \
  --data training_data.csv \
  --epochs 5 \
  --output checkpoints/best_text_model.pth
```

**GPU (recommended, ~30 min):**
```bash
TORCH_DEVICE=cuda python scripts/train_text_classifier.py --data training_data.csv --epochs 10 --output checkpoints/best_text_model.pth
```

**CPU fallback (~4 hours):**
```bash
python scripts/train_text_classifier.py --data training_data.csv --epochs 5 --output checkpoints/best_text_model.pth
```

Alternatively, use [Google Colab](https://colab.research.google.com) with an A100 GPU for free training in ~10 minutes.

---

## Step 4: Train the Frame Classifier (Optional)

The frame classifier uses EfficientNet fine-tuned on frame screenshots.

> ⚠️ **GPU required.** Recommended: Colab A100 or local RTX 3070+.

```bash
python scripts/train_frame_classifier.py \
  --frames_dir /tmp/ytclassifier/frames \
  --labels training_data.csv \
  --output checkpoints/best_frame_model.pth \
  --epochs 10
```

---

## Step 5: Deploy New Checkpoints

1. Copy the trained checkpoint to `backend/checkpoints/`:
   ```bash
   cp checkpoints/best_text_model.pth backend/checkpoints/
   ```

2. Restart the Celery workers to load new weights:
   ```bash
   # Local dev
   pkill -f "celery worker" && celery -A services.pipeline.celery_app worker --loglevel=info
   
   # Docker
   docker-compose restart worker
   ```

3. Verify the model loaded without error in the worker logs:
   ```
   [worker] Loaded text classifier from checkpoints/best_text_model.pth
   ```

---

## Accuracy Measurement

To measure classification accuracy before and after retraining, run:

```bash
pytest tests/integration/test_classifier_accuracy.py -v
```

This runs 20 known videos through the pipeline and checks that category predictions match expected labels. The baseline accuracy threshold is 80%.
