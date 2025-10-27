<<<<<<< Updated upstream
=======
# examples/cnn_training.py
import argparse, random, sys, time, os, csv
from pathlib import Path

import numpy as np
>>>>>>> Stashed changes
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
from prov4ml.datamodel.metric_type import MetricsType
from prov4ml.constants import PROV4ML_DATA

# --- helpers to tolerate enum naming across versions ---
def _ctx_training():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).TRAINING
def _ctx_validation():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).VALIDATION
def _ctx_models():
    return getattr(prov4ml, "Contexts", getattr(prov4ml, "Context", None)).MODELS
>>>>>>> Stashed changes

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = "cpu"

class CNN(nn.Module): 
        
    def __init__(self, out_classes=10):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )
        self.mlp = nn.Linear(36864, out_features=out_classes)

    def forward(self, x): 
        x = self.sequential(x)
        x = x.reshape(x.shape[0], -1)
        x = self.mlp(x)
        return x


<<<<<<< Updated upstream
model = CNN()


tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

train_ds = CIFAR10(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(BATCH_SIZE*200))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
yprov4ml.log_dataset("train_loader", train_loader)


test_ds = CIFAR10(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*100))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
yprov4ml.log_dataset("test_loader", test_loader)
=======
# ---------- dataset helper: download only if missing ----------
def _dataset(root: str, train: bool, tform):
    cifar_root = os.path.join(root, "cifar-10-batches-py")
    download = not os.path.exists(cifar_root)
    return CIFAR10(root, train=train, download=download, transform=tform)


# ---------- LR scheduler factory + logger ----------
def make_scheduler(optimizer, args, steps_per_epoch=None):
    """Return (scheduler, mode) where mode in {'epoch','batch','plateau'}."""
    if args.scheduler == "none":
        return None, "epoch"
    if args.scheduler == "step":
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
        return sch, "epoch"
    if args.scheduler == "exp":
        sch = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.lr_gamma
        )
        return sch, "epoch"
    if args.scheduler == "cosine":
        T = args.t_max or args.epochs
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)
        return sch, "epoch"
    if args.scheduler == "onecycle":
        assert steps_per_epoch is not None, "OneCycleLR requires steps_per_epoch"
        max_lr = args.max_lr if args.max_lr is not None else args.lr * 10.0
        sch = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            pct_start=args.pct_start
        )
        return sch, "batch"
    if args.scheduler == "plateau":
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_gamma, patience=args.plateau_patience
        )
        return sch, "plateau"
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


class LRLogger:
    """Append-only CSV logger for learning rate trace."""
    def __init__(self, out_dir: str | None, exp_id: str | None):
        self.enabled = out_dir is not None
        if not self.enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{(exp_id or 'exp')}.csv")
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["step_type", "index", "lr"])

    def log(self, step_type: str, index: int, optimizer):
        if not self.enabled:
            return
        lr = optimizer.param_groups[0]["lr"]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([step_type, index, lr])


def main():
    parser = argparse.ArgumentParser(description="CIFAR10 CNN training with Prov4ML logging")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_id", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="./data")

    # scheduler + LR logging (forwarded by run_batch.py)
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["none","step","exp","cosine","onecycle","plateau"])
    parser.add_argument("--lr_step_size", type=int, default=10)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--t_max", type=int, default=None)
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--plateau_patience", type=int, default=3)
    parser.add_argument("--lr_log_dir", type=str, default=None)
    parser.add_argument("--exp_id", type=str, default=None)

    args = parser.parse_args()

    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Start provenance run (enable carbon metrics)
    prov4ml.start_run(
        prov_user_namespace="www.example.org",
        experiment_name="experiment_name",
        provenance_save_dir="examples/prov",
        metrics_file_type=MetricsType.NETCDF,
        save_after_n_logs=1000,
        collect_all_processes=True,
        disable_codecarbon=False,   # enable energy/carbon sampling
        unify_experiments=True
    )

    print(f"[run {args.run_id}] device={args.device} bs={args.batch_size} lr={args.lr} "
          f"epochs={args.epochs} scheduler={args.scheduler}")

    # Data
    tform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

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
    loss_fn = nn.MSELoss().to(device)

    # Scheduler + LR trace
    steps_per_epoch = len(train_loader)
    scheduler, schedule_mode = make_scheduler(optim, args, steps_per_epoch=steps_per_epoch)
    lrlog = LRLogger(args.lr_log_dir, args.exp_id or str(args.run_id))

    # Training loop
    for epoch in range(args.epochs):
        # manual epoch timing (yProv4ML 'log_current_execution_time' without start/stop API)
        t0 = time.time()

        model.train()
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            y_onehot = F.one_hot(y, 10).float()
            loss = loss_fn(logits, y_onehot)
            loss.backward()
            optim.step()

            # batch-level schedule (e.g., OneCycle)
            if schedule_mode == "batch" and scheduler is not None:
                scheduler.step()
                lrlog.log("batch", epoch * steps_per_epoch + i, optim)

        # epoch-level schedule
        if schedule_mode == "epoch" and scheduler is not None:
            scheduler.step()
            lrlog.log("epoch", epoch, optim)

        # Validation (also needed for ReduceLROnPlateau)
        model.eval()
        val_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                y_onehot = F.one_hot(y, 10).float()
                vloss = loss_fn(logits, y_onehot)
                val_sum += vloss.item()
                pred = logits.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        avg_val = val_sum / max(1, len(test_loader))
        acc = correct / max(1, total)

        if schedule_mode == "plateau" and scheduler is not None:
            scheduler.step(avg_val)
            lrlog.log("epoch", epoch, optim)

        # log epoch timing as metric (ms)
        epoch_ms = (time.time() - t0) * 1000.0
        prov4ml.log_metric("train_epoch_time_ms", float(epoch_ms), context=_ctx_training(), step=epoch)

        # training loss (last batch) as epoch metric
        prov4ml.log_metric("MSE_train", float(loss.item()), context=_ctx_training(), step=epoch)

        # system & carbon metrics (once per epoch is fine; call more often if you want finer sampling)
        try:
            prov4ml.log_system_metrics(_ctx_training(), step=epoch)
        except Exception:
            pass
        try:
            prov4ml.log_carbon_metrics(context=_ctx_training(), step=epoch)
        except Exception:
            pass

        # validation metrics
        prov4ml.log_metric("MSE_val", float(avg_val), _ctx_validation(), step=epoch)
        prov4ml.log_metric("ACC_val", float(acc), _ctx_validation(), step=epoch)

        # incremental model snapshot
        prov4ml.save_model_version("cifar_model_version", model, _ctx_models(), epoch)

    # Final logs of static params
    prov4ml.log_model("cifar_model_final", model, log_model_layers=True, is_input=False)
    prov4ml.log_metric("param_batch_size", float(args.batch_size), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_lr", float(args.lr), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_epochs", float(args.epochs), context=_ctx_training(), step=0)
    prov4ml.log_metric("param_seed", float(args.seed), context=_ctx_training(), step=0)


    print(f"[run {args.run_id}] Val loss: {avg_loss:.4f} | Val acc: {acc:.4f}")

    # End run + graphs
    prov4ml.end_run(create_graph=True, create_svg=True)


if __name__ == "__main__":
    main()
