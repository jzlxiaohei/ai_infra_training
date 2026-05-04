"""
Week 1 — DDP version of train_single.py.

Differences from the single-GPU version are marked with `# DDP:`.

Launch with torchrun (it spawns one process per GPU and sets env vars):
    torchrun --nproc_per_node=2 train_ddp.py --epochs 2

torchrun sets these env vars in each child process:
    RANK         — global rank across all nodes (0 .. WORLD_SIZE-1)
    LOCAL_RANK   — rank within this node (0 .. nproc_per_node-1)
    WORLD_SIZE   — total processes
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet18


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def setup() -> int:
    # DDP: join the process group. NCCL = NVIDIA's GPU collective comm library.
    # Blocks until every rank has called this.
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def get_loaders(batch_size: int, subset: int | None = None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # DDP: only rank 0 downloads to avoid races on disk; others wait at the barrier.
    if is_main_process():
        datasets.CIFAR10("./data", train=True, download=True)
        datasets.CIFAR10("./data", train=False, download=True)
    dist.barrier()

    train = datasets.CIFAR10("./data", train=True, download=False, transform=transform)
    test = datasets.CIFAR10("./data", train=False, download=False, transform=transform)

    if subset:
        train = Subset(train, range(subset))
        test = Subset(test, range(subset))

    # DDP: sampler partitions indices across ranks so each rank sees a different shard.
    train_sampler = DistributedSampler(train, shuffle=True)
    test_sampler = DistributedSampler(test, shuffle=False)

    train_loader = DataLoader(train, batch_size=batch_size, sampler=train_sampler,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, sampler=test_sampler,
                             num_workers=2, pin_memory=True)
    return train_loader, test_loader, train_sampler


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
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        # DDP: backward triggers NCCL all-reduce of gradients across ranks.
        # After this line every rank has the SAME averaged .grad on every parameter.
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return reduce_metrics(running_loss, correct, total, device)


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
    return reduce_metrics(running_loss, correct, total, device)


def reduce_metrics(running_loss, correct, total, device):
    # DDP: each rank covers only its shard. SUM-reduce the three counters across ranks
    # so every rank ends up with the same global view.
    t = torch.tensor([running_loss, correct, total], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    loss, c, n = t.tolist()
    return loss / n, c / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="per-rank batch size; global batch = batch_size * world_size")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--subset", type=int, default=None)
    args = parser.parse_args()

    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process():
        print(f"world_size={dist.get_world_size()}  per-rank batch={args.batch_size}  "
              f"global batch={args.batch_size * dist.get_world_size()}")

    train_loader, test_loader, train_sampler = get_loaders(args.batch_size, args.subset)

    model = build_model().to(device)
    # DDP: wrap the model. This registers gradient hooks that all-reduce on backward.
    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        # DDP: required so each epoch's shuffle is different but identical across ranks.
        train_sampler.set_epoch(epoch)
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        dt = time.time() - t0
        if is_main_process():
            print(
                f"epoch {epoch+1}/{args.epochs}  "
                f"train_loss={train_loss:.3f} train_acc={train_acc:.3f}  "
                f"test_loss={test_loss:.3f} test_acc={test_acc:.3f}  "
                f"({dt:.1f}s)"
            )

    cleanup()


if __name__ == "__main__":
    main()
