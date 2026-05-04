"""
Week 1 — Single-device training baseline.

CIFAR-10 + ResNet18 in pure PyTorch (no Lightning, no Trainer).
Goal: read this file end-to-end and understand every line.

Run:
    python train_single.py --epochs 2                       # cuda if available
    python train_single.py --device cpu --subset 500        # local Mac smoke test
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18


def get_loaders(batch_size: int, subset: int | None = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test = datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    if subset:
        train = Subset(train, range(subset))
        test = Subset(test, range(subset))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader


def build_model(num_classes: int = 10) -> nn.Module:
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # The 5 lines that ARE the training loop:
        optimizer.zero_grad()           # 1) clear stale gradients from previous step
        logits = model(x)               # 2) forward pass
        loss = criterion(logits, y)     # 3) scalar loss
        loss.backward()                 # 4) backward — fills .grad on every parameter
        optimizer.step()                # 5) apply update rule (SGD/Adam) using .grad

        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=None,
                        help="use a small subset for smoke testing")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device={device}  batch_size={args.batch_size}  epochs={args.epochs}")

    train_loader, test_loader = get_loaders(args.batch_size, args.subset)
    model = build_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        dt = time.time() - t0
        print(
            f"epoch {epoch+1}/{args.epochs}  "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
            f"test_loss={test_loss:.3f} test_acc={test_acc:.3f}  "
            f"({dt:.1f}s)"
        )


if __name__ == "__main__":
    main()
