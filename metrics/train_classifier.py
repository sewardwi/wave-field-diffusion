"""
Train the feature-extractor classifiers used for Frechet-distance metrics.

These classifiers are NOT used during diffusion training — they are evaluated
once on real data, saved to disk, then loaded read-only by metrics/evaluate.py
and train_audio.py to produce embeddings for FMD / FSD computation.

Usage:
    python -m metrics.train_classifier --task mnist
    python -m metrics.train_classifier --task sc09

Output:
    metrics/weights/{task}_classifier.pt   # state_dict
    metrics/weights/{task}_classifier.json # test accuracy + config
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms

from metrics.classifier import MNISTClassifier, SC09Classifier, param_count


WEIGHTS_DIR = Path(__file__).parent / "weights"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# MNIST
# ---------------------------------------------------------------------------

def train_mnist(epochs: int, batch_size: int, lr: float, device):
    # Match diffusion preprocessing: scale into [-1, 1]
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    model = MNISTClassifier().to(device)
    print(f"MNIST classifier params: {param_count(model):,}")
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, losssum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losssum += loss.item() * x.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += x.size(0)
        train_acc = correct / total

        # Eval
        model.eval()
        ec, et = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                ec += (model(x).argmax(-1) == y).sum().item()
                et += x.size(0)
        test_acc = ec / et
        print(f"  epoch {epoch:2d}  train_loss={losssum/total:.4f}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")

    return model, {"test_acc": test_acc, "epochs": epochs, "params": param_count(model)}


# ---------------------------------------------------------------------------
# SC09
# ---------------------------------------------------------------------------

def train_sc09(epochs: int, batch_size: int, lr: float, device):
    from datasets.sc09 import SC09

    train_ds = SC09(root="./data", subset="training")
    val_ds   = SC09(root="./data", subset="validation")
    print(f"SC09 train clips: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=min(4, os.cpu_count()), pin_memory=(device.type != "cpu"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=min(4, os.cpu_count()), pin_memory=(device.type != "cpu"),
    )

    model = SC09Classifier().to(device)
    print(f"SC09 classifier params: {param_count(model):,}")
    opt = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, losssum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losssum += loss.item() * x.size(0)
            correct += (logits.argmax(-1) == y).sum().item()
            total += x.size(0)
        train_acc = correct / total

        model.eval()
        ec, et = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                ec += (model(x).argmax(-1) == y).sum().item()
                et += x.size(0)
        val_acc = ec / et
        print(f"  epoch {epoch:2d}  train_loss={losssum/total:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")

    return model, {"val_acc": val_acc, "epochs": epochs, "params": param_count(model)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["mnist", "sc09"])
    p.add_argument("--epochs", type=int, default=None,
                   help="Default: 5 for MNIST, 20 for SC09")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.task == "mnist":
        epochs = args.epochs or 5
        model, stats = train_mnist(epochs, args.batch_size, args.lr, device)
    else:
        epochs = args.epochs or 20
        model, stats = train_sc09(epochs, args.batch_size, args.lr, device)

    wpath = WEIGHTS_DIR / f"{args.task}_classifier.pt"
    jpath = WEIGHTS_DIR / f"{args.task}_classifier.json"
    torch.save(model.state_dict(), wpath)
    stats["batch_size"] = args.batch_size
    stats["lr"] = args.lr
    with open(jpath, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved → {wpath}")
    print(f"Stats → {jpath}")


if __name__ == "__main__":
    main()
