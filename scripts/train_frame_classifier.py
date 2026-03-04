"""
scripts/train_frame_classifier.py

Fine-tune EfficientNet-B0 on video frame data for the 8-class category classification.

Usage
─────
# 1. Prepare a dataset CSV:
#    columns: frame_path, label
#    label must be one of: comedy|listicle|music|educational|news|review|gaming|vlog

# 2. Run training:
#    python scripts/train_frame_classifier.py \
#        --data data/frames_dataset.csv \
#        --epochs 20 \
#        --batch 32 \
#        --lr 1e-4 \
#        --output models/frame_classifier.pt

Architecture
────────────
- EfficientNet-B0 pretrained on ImageNet
- Replace classifier head: Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.2) → Linear(256→8)
- Freeze backbone for 5 epochs (warm-up), then unfreeze all layers
- Use label smoothing cross-entropy + cosine annealing LR scheduler
- Mixed precision training (torch.amp) for faster training on GPU
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

CATEGORIES = ["comedy", "listicle", "music", "educational", "news", "review", "gaming", "vlog", "shopping", "recipe"]
N_CLASSES = len(CATEGORIES)


# ── Dataset ────────────────────────────────────────────────────────────────────

class FrameDataset(Dataset):
    """Loads (frame_path, label) pairs from a CSV file."""

    TRAIN_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    EVAL_TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, csv_path: str, train: bool = True):
        self.df = pd.read_csv(csv_path)
        self.transform = self.TRAIN_TRANSFORM if train else self.EVAL_TRANSFORM
        self.label_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}

        # Filter out missing/invalid rows
        valid = []
        for _, row in self.df.iterrows():
            if row["label"] in self.label_to_idx and Path(row["frame_path"]).exists():
                valid.append(row)

        self.samples = valid
        print(f"Dataset: {len(self.samples)} valid samples")

        # Class distribution for weighted sampler
        counts = [0] * N_CLASSES
        for row in self.samples:
            counts[self.label_to_idx[row["label"]]] += 1
        self.class_weights = torch.tensor(
            [max(counts) / max(c, 1) for c in counts], dtype=torch.float
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        try:
            img = Image.open(row["frame_path"]).convert("RGB")
            img = self.transform(img)
        except (UnidentifiedImageError, FileNotFoundError, OSError):
            # Return a black image on bad files to avoid crashing the dataloader
            img = torch.zeros(3, 224, 224)
        label = self.label_to_idx[row["label"]]
        return img, label


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(n_classes: int = N_CLASSES) -> nn.Module:
    """EfficientNet-B0 with custom classification head."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, n_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device.type, enabled=device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with autocast(device.type, enabled=device.type == "cuda"):
            logits = model(imgs)
            loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        n += imgs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / n, correct / n, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 frame classifier")
    parser.add_argument("--data",    required=True, help="Path to CSV with frame_path,label columns")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--warmup",  type=int, default=5, help="Epochs with frozen backbone")
    parser.add_argument("--val",     type=float, default=0.15, help="Validation split fraction")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output",  default="models/frame_classifier.pt")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Training on: {device}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = FrameDataset(args.data, train=True)
    n_val = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    val_ds.dataset.transform = FrameDataset.EVAL_TRANSFORM

    # Weighted sampler to handle class imbalance
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=[dataset.class_weights[dataset.label_to_idx.get(dataset.samples[i]["label"], 0)]
                 for i in range(len(dataset))],
        num_samples=len(dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch,
        sampler=sampler,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch * 2,
        num_workers=args.workers, pin_memory=True,
    )

    # ── Model + optimiser ─────────────────────────────────────────────────────
    model = build_model().to(device)
    freeze_backbone(model)   # warm-up: train head only

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1,
        weight=dataset.class_weights.to(device),
    )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after warm-up
        if epoch == args.warmup + 1:
            print(f"\n[Epoch {epoch}] Unfreezing backbone for full fine-tuning")
            unfreeze_all(model)
            optimizer = AdamW(model.parameters(), lr=args.lr / 10, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=1e-7)

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:2d}/{args.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(output_path))
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.3f})")

    # ── Final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(torch.load(str(output_path), map_location=device))
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion, device)

    print(f"\n{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(final_labels, final_preds, target_names=CATEGORIES))
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(final_labels, final_preds))
    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    main()
