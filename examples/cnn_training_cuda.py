import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from prov4ml.constants import PROV4ML_DATA


# import your local prov4ml package
sys.path.append("../yProvML")
import prov4ml
import prov4ml, prov4ml.datamodel.metric_data as md, prov4ml.datamodel.metric_type as mt
print("[prov4ml path]", prov4ml.__file__)
print("[metric_data path]", md.__file__)

# ——————————————————————————————————————————
# Provenance: start run
# ——————————————————————————————————————————
prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name",
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=True,
    disable_codecarbon=True,
    unify_experiments=True
)

# ——————————————————————————————————————————
# Config
# ——————————————————————————————————————————
PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = "cpu"
NUM_CLASSES = 10
LR = 1e-3
print("=== prov4ml runtime ===")
print("unify_experiments:", PROV4ML_DATA.unify_experiments)
print("metrics_file_type:", getattr(PROV4ML_DATA, "metrics_file_type", None))
print("RUN_ID:", PROV4ML_DATA.RUN_ID)
print("EXPERIMENT_DIR:", PROV4ML_DATA.EXPERIMENT_DIR)
print("METRIC_DIR (current run):", PROV4ML_DATA.METRIC_DIR)
# ——————————————————————————————————————————
# Model
# ——————————————————————————————————————————
class SimpleCIFARCNN(nn.Module):
    def __init__(self, out_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # avoid hardcoding flatten size
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

model = SimpleCIFARCNN().to(DEVICE)

# ——————————————————————————————————————————
# Data
# ——————————————————————————————————————————
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_tf = transforms.Compose([
    transforms.ToTensor(),
])

train_ds = CIFAR10(PATH_DATASETS, train=True, download=True, transform=train_tf)
train_ds = Subset(train_ds, range(BATCH_SIZE * 200))  # small subset for speed
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
prov4ml.log_dataset("train_loader", train_loader)

test_ds = CIFAR10(PATH_DATASETS, train=False, download=True, transform=test_tf)
test_ds = Subset(test_ds, range(BATCH_SIZE * 100))
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
prov4ml.log_dataset("test_loader", test_loader)

# ——————————————————————————————————————————
# Optimization
# ——————————————————————————————————————————
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)

# ——————————————————————————————————————————
# Training loop
# ——————————————————————————————————————————
train_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_losses.append(loss.item())

    avg_train_loss = running_loss / max(1, len(train_loader))
    prov4ml.log_metric("CE_train", avg_train_loss, context=prov4ml.Context.TRAINING, step=epoch)
    prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)

    # Save incremental model versions
    prov4ml.save_model_version("cifar_model_version", model, prov4ml.Context.MODELS, epoch)

# ——————————————————————————————————————————
# Evaluation
# ——————————————————————————————————————————
model.eval()
val_loss_sum = 0.0
correct = 0
total = 0

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)
        val_loss_sum += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

avg_val_loss = val_loss_sum / max(1, len(test_loader))
accuracy = correct / max(1, total)

prov4ml.log_metric("CE_val", avg_val_loss, context=prov4ml.Context.VALIDATION, step=EPOCHS - 1)
prov4ml.log_metric("ACC_val", accuracy, context=prov4ml.Context.VALIDATION, step=EPOCHS - 1)

# Log final model
prov4ml.log_model("cifar_model_final", model, log_model_layers=True, is_input=False)

print(f"Val loss: {avg_val_loss:.4f} | Val acc: {accuracy:.4f}")

# ——————————————————————————————————————————
# (Optional) quick loss plot
# ——————————————————————————————————————————
try:
    import matplotlib.pyplot as plt
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("CE Loss")
    plt.show()
except Exception:
    pass

# ——————————————————————————————————————————
# Provenance: end run
# ——————————————————————————————————————————
prov4ml.end_run(
    create_graph=True,
    create_svg=True,
)
