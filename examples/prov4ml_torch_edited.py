#!/usr/bin/env python
import argparse, random, sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# local prov4ml
sys.path.append("../yProvML")
import prov4ml
from prov4ml.datamodel.metric_type import MetricsType

# --- helpers to tolerate enum name differences across prov4ml versions ---
def _ctx_training():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).TRAINING
def _ctx_validation():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).VALIDATION
def _ctx_models():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).MODELS

class MNISTModel(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, out_classes),
        )
    def forward(self, x):
        return self.net(x)

def _dataset(root: str, train: bool, tform):
    """
    MNIST with guarded download to avoid parallel corruption.
    """
    mnist_marker = os.path.join(root, "MNIST", "raw")
    download = not os.path.exists(mnist_marker)
    return MNIST(root, train=train, download=download, transform=tform)

def main():
    p = argparse.ArgumentParser(description="MNIST MLP training with Prov4ML logging")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_id", type=int, default=0)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--experiment_name", type=str, default="mnist_experiment")
    p.add_argument("--save_dir", type=str, default="examples/prov")
    p.add_argument("--unify_experiments", type=lambda s: s.lower() in {"1","true","yes"}, default=True)
    args = p.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Start provenance
    prov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name=args.experiment_name,
        provenance_save_dir=args.save_dir,
        metrics_file_type=MetricsType.NETCDF,
        save_after_n_logs=1000,
        collect_all_processes=True,
        disable_codecarbon=True,
        unify_experiments=args.unify_experiments,
    )

    print(f"[run {args.run_id}] device={args.device} bs={args.batch_size} lr={args.lr} epochs={args.epochs}")

    # Data / transforms
    tform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    os.makedirs(args.data_dir, exist_ok=True)

    train_ds = _dataset(args.data_dir, train=True,  tform=tform)
    # keep runtime short + deterministic size across bs
    train_ds = Subset(train_ds, range(args.batch_size * 200))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    prov4ml.log_dataset("train_loader", train_loader)

    test_ds = _dataset(args.data_dir, train=False, tform=transforms.ToTensor())
    test_ds = Subset(test_ds, range(args.batch_size * 100))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    prov4ml.log_dataset("test_loader", test_loader)

    # Model / optim / loss
    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model = MNISTModel().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss().to(device)

    # --- Log static run parameters as metrics (once per run) ---
    prov4ml.log_metric("param_batch_size", float(args.batch_size), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_lr", float(args.lr), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_epochs", float(args.epochs), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_seed", float(args.seed), context=_ctx_training(), step=0)

    # Train
    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            y_onehot = F.one_hot(y, 10).float()
            loss = loss_fn(logits, y_onehot)
            loss.backward()
            optim.step()

        # log once per epoch
        prov4ml.log_metric("MSE_train", float(loss.item()), context=_ctx_training(), step=epoch)

        # only attempt GPU system metrics when CUDA is around
        if torch.cuda.is_available():
            try:
                prov4ml.log_system_metrics(_ctx_training(), step=epoch)
            except ZeroDivisionError:
                pass

        prov4ml.save_model_version("mnist_model_version", model, _ctx_models(), epoch)

    # Eval
    model.eval()
    val_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            y_onehot = F.one_hot(y, 10).float()
            loss = loss_fn(logits, y_onehot)
            val_sum += loss.item()

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = val_sum / max(1, len(test_loader))
    acc = correct / max(1, total)

    prov4ml.log_metric("MSE_val", float(avg_loss), _ctx_validation(), step=args.epochs - 1)
    prov4ml.log_metric("ACC_val", float(acc), _ctx_validation(), step=args.epochs - 1)
    prov4ml.log_model("mnist_model_final", model, log_model_layers=True, is_input=False)

    print(f"[run {args.run_id}] Val loss: {avg_loss:.4f} | Val acc: {acc:.4f}")

    prov4ml.end_run(create_graph=True, create_svg=True)

if __name__ == "__main__":
    main()
