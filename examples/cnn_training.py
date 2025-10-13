# examples/cnn_training.py
import argparse, random, sys, time, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from prov4ml.constants import PROV4ML_DATA
from prov4ml.datamodel.metric_type import get_file_type
import inspect, prov4ml.datamodel.metric_data as md

# local prov4ml
sys.path.append("../yProvML")
import prov4ml
from prov4ml.datamodel.metric_type import MetricsType  # <-- important
from prov4ml.constants import PROV4ML_DATA


# --- tiny helper to tolerate different enum names between versions ---
def _ctx_training():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).TRAINING
def _ctx_validation():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).VALIDATION
def _ctx_models():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).MODELS


class CNN(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 64, 5),
            nn.ReLU(),
        )
        self.mlp = nn.Linear(36864, out_features=out_classes)

    def forward(self, x):
        x = self.sequential(x)
        x = x.reshape(x.shape[0], -1)
        return self.mlp(x)


# ---------- dataset helper: download only if missing ----------
def _dataset(root: str, train: bool, tform):
    """
    Uses torchvision's CIFAR10 but sets download=True only if the dataset
    folder doesn't exist yet, to avoid concurrent download corruption.
    """
    cifar_root = os.path.join(root, "cifar-10-batches-py")
    download = not os.path.exists(cifar_root)
    return CIFAR10(root, train=train, download=download, transform=tform)


def main():
    parser = argparse.ArgumentParser(description="CIFAR10 CNN training with Prov4ML logging")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Start provenance run
    prov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name="experiment_name",
        provenance_save_dir="examples/prov",
        metrics_file_type=MetricsType.NETCDF,  # <-- NetCDF
        save_after_n_logs=1000,                # <-- flush every log so .nc files appear immediately
        collect_all_processes=True,
        disable_codecarbon=True,
        unify_experiments=True
    )

    print(f"[run {args.run_id}] device={args.device} bs={args.batch_size} lr={args.lr} epochs={args.epochs}")

    # Data
    tform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    # use guarded dataset creation (avoids parallel download races)
    os.makedirs(args.data_dir, exist_ok=True)
    train_ds = _dataset(args.data_dir, train=True,  tform=tform)
    train_ds = Subset(train_ds, range(args.batch_size * 200))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    prov4ml.log_dataset("train_loader", train_loader)

    test_ds = _dataset(args.data_dir, train=False, tform=tform)
    test_ds = Subset(test_ds, range(args.batch_size * 100))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    prov4ml.log_dataset("test_loader", test_loader)

    # Model/optim/loss
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss().to(device)  # keep your original choice

    # Train
    losses = []
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
            losses.append(loss.item())

        # log once per epoch
        prov4ml.log_metric("MSE_train", float(loss.item()), context=_ctx_training(), step=epoch)

        # Only log GPU system metrics if CUDA is available to avoid ZeroDivisionError on CPU
        if torch.cuda.is_available():
            try:
                prov4ml.log_system_metrics(_ctx_training(), step=epoch)
            except ZeroDivisionError:
                # Some environments report reserved==0 early; ignore and continue
                pass

        # incremental model snapshot
        prov4ml.save_model_version("cifar_model_version", model, _ctx_models(), epoch)

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

            # ✅ Accuracy from logits, not from one-hot labels
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    avg_loss = val_sum / max(1, len(test_loader))
    acc = correct / max(1, total)

    # Log final metrics/model
    prov4ml.log_metric("MSE_val", float(avg_loss), _ctx_validation(), step=args.epochs - 1)
    prov4ml.log_metric("ACC_val", float(acc), _ctx_validation(), step=args.epochs - 1)
    prov4ml.log_model("cifar_model_final", model, log_model_layers=True, is_input=False)
    # Log static run parameters as metrics (once per run)
    prov4ml.log_metric("param_batch_size", float(args.batch_size), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_lr", float(args.lr), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_epochs", float(args.epochs), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_seed", float(args.seed), context=_ctx_training(), step=0)


    print(f"[run {args.run_id}] Val loss: {avg_loss:.4f} | Val acc: {acc:.4f}")

    # End run + graphs
    prov4ml.end_run(create_graph=True, create_svg=True)


if __name__ == "__main__":
    main()
