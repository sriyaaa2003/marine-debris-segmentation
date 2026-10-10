"""Loss, metrics and the training loop used for every model in the comparison."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .data import NUM_CLASSES


class InvFreqWCE(nn.Module):
    """Cross-entropy with inverse-frequency class weights (debris is 0.45% of labelled pixels); -1 is ignored."""

    def __init__(self, class_counts: dict, ignore_index: int = -1):
        super().__init__()
        total = sum(class_counts.values())
        w = torch.tensor([total / (2.0 * max(class_counts.get(i, 1), 1)) for i in range(NUM_CLASSES)], dtype=torch.float32)
        w = w / w.sum() * NUM_CLASSES
        self.ce = nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)

    def forward(self, logits, targets):
        return self.ce(logits.float(), targets)


def compute_metrics(preds: np.ndarray, targets: np.ndarray, ignore_index: int = -1) -> dict:
    valid = targets != ignore_index
    preds, targets = preds[valid], targets[valid]
    if len(preds) == 0:
        return {"debris_f1": 0.0, "debris_iou": 0.0, "macro_f1": 0.0, "accuracy": 0.0}
    labels = list(range(NUM_CLASSES))
    debris_f1 = float(f1_score(targets, preds, labels=[0], average="macro", zero_division=0))
    return {
        "debris_f1": debris_f1,
        "debris_iou": debris_f1 / (2.0 - debris_f1) if debris_f1 < 2.0 else 0.0,   # IoU = F1 / (2 - F1) for one class
        "macro_f1": float(f1_score(targets, preds, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(targets, preds)),
    }


def warmup_cosine(epoch: int, epochs: int, warmup: int = 5) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(epochs - warmup, 1)))


def run_epoch(model, loader, criterion, device, optimizer=None, clip_grad=0.5) -> tuple[float, dict]:
    train = optimizer is not None
    model.train(train)
    total, n, preds, targets = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            if (labels != -1).sum() == 0:
                continue
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                continue
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            total, n = total + loss.item(), n + 1
            preds.append(logits.argmax(1).cpu().numpy().ravel())
            targets.append(labels.cpu().numpy().ravel())
    metrics = compute_metrics(np.concatenate(preds), np.concatenate(targets)) if preds else compute_metrics(np.array([]), np.array([]))
    return total / max(n, 1), metrics


def fit(model, train_loader, val_loader, criterion, device, epochs=50, lr=5e-4, weight_decay=1e-5, ckpt_path=None):
    """Adam with 5 warm-up epochs and cosine decay. Keeps the checkpoint with the best validation debris F1."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: warmup_cosine(e, epochs))
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        tr_loss, tr = run_epoch(model, train_loader, criterion, device, opt)
        va_loss, va = run_epoch(model, val_loader, criterion, device)
        sched.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, **{f"val_{k}": v for k, v in va.items()}})
        if va["debris_f1"] > best:
            best = va["debris_f1"]
            if ckpt_path:
                torch.save({"epoch": epoch, "model": model.state_dict(), "metrics": va}, ckpt_path)
    return history
