"""
scripts/train_text_classifier.py

Fine-tune BERT (bert-base-uncased) on video transcript + title data
for the 8-class category classification task.

Usage
─────
python scripts/train_text_classifier.py \
    --data data/transcripts_dataset.csv \
    --epochs 5 \
    --batch 16 \
    --lr 2e-5 \
    --output models/text_classifier.pt

CSV format
──────────
columns: text, label
  text  = concatenation of video title + " [SEP] " + transcript excerpt
  label = one of: comedy|listicle|music|educational|news|review|gaming|vlog
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertModel, BertTokenizer
from sklearn.metrics import classification_report

CATEGORIES = ["comedy", "listicle", "music", "educational", "news", "review", "gaming", "vlog"]
N_CLASSES = len(CATEGORIES)
MAX_LEN = 512


# ── Dataset ────────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["text", "label"])
        df = df[df["label"].isin(CATEGORIES)]
        self.texts = df["text"].tolist()
        self.labels = [CATEGORIES.index(l) for l in df["label"]]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        print(f"TextDataset: {len(self.texts)} samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids", torch.zeros(MAX_LEN, dtype=torch.long)).squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class BERTClassifier(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return self.classifier(out.pooler_output)


# ── Train / eval ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, criterion, scaler, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += labels.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels  = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"].to(device)
        with autocast(enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        n          += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / n, correct / n, all_preds, all_labels


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT text classifier")
    parser.add_argument("--data",    required=True)
    parser.add_argument("--epochs",  type=int, default=5)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--lr",      type=float, default=2e-5)
    parser.add_argument("--val",     type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output",  default="models/text_classifier.pt")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Training on: {device}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Data
    dataset = TextDataset(args.data)
    n_val   = int(len(dataset) * args.val)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=args.workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch * 2, shuffle=False, num_workers=args.workers)

    # Model
    model     = BERTClassifier().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = total_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device)
        val_loss, val_acc, preds, labels = eval_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch}/{args.epochs}  "
            f"train={train_loss:.4f}/{train_acc:.3f}  "
            f"val={val_loss:.4f}/{val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(output_path))
            print(f"  ✓ Saved best model (val_acc={val_acc:.4f})")

    # Final report
    model.load_state_dict(torch.load(str(output_path), map_location=device))
    _, final_acc, final_preds, final_labels = eval_epoch(model, val_loader, criterion, device)
    print(f"\nBest val acc: {best_val_acc:.4f}")
    print(classification_report(final_labels, final_preds, target_names=CATEGORIES))
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
