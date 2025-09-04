import os
import sys
import glob

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# local prov4ml
sys.path.append("../yProvML")
import prov4ml
from prov4ml.datamodel.metric_type import MetricsType  # <-- important
from prov4ml.constants import PROV4ML_DATA


prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name",
    provenance_save_dir="prov",
    metrics_file_type=MetricsType.NETCDF,  # <-- NetCDF
    save_after_n_logs=1000,                   # <-- flush every log so .nc files appear immediately
    collect_all_processes=True,
    disable_codecarbon=True,
    unify_experiments=True
)

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = "cpu"
LR = 1e-3
from prov4ml.datamodel.metric_type import get_file_type
from prov4ml.constants import PROV4ML_DATA
import inspect, prov4ml.datamodel.metric_data as md

print("Using prov4ml from:", prov4ml.__file__)
print("Runtime metrics_file_type:", PROV4ML_DATA.metrics_file_type)
print("File extension for runtime type:", get_file_type(PROV4ML_DATA.metrics_file_type))
print("save_after_n_logs:", PROV4ML_DATA.save_metrics_after_n_logs)
print("MetricInfo.save_to_file implementation:\n", inspect.getsource(md.MetricInfo.save_to_file))

class CNN(nn.Module):
    def __init__(self, out_classes=10):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(3, 20, 5),
            nn.ReLU(),
            nn.Conv2d(20, 64, 5),
            nn.ReLU()
        )
        self.mlp = nn.Linear(36864, out_classes)

    def forward(self, x):
        x = self.sequential(x)
        x = x.reshape(x.shape[0], -1)
        return self.mlp(x)

model = CNN().to(DEVICE)

tform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])

train_ds = CIFAR10(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(BATCH_SIZE * 200))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
prov4ml.log_dataset("train_loader", train_loader)

test_ds = CIFAR10(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE * 100))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
prov4ml.log_dataset("test_loader", test_loader)

optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)

# ——— Train
for epoch in range(EPOCHS):
    model.train()
    running = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()
        running += loss.item()

    avg_train = running / max(1, len(train_loader))
    prov4ml.log_metric("CE_train", avg_train, context=prov4ml.Context.TRAINING, step=epoch)
    prov4ml.save_model_version("cifar_model_version", model, prov4ml.Context.MODELS, epoch)

# ——— Eval
model.eval()
val_sum, correct, total = 0.0, 0, 0
with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Evaluating"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        val_sum += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

avg_val = val_sum / max(1, len(test_loader))
acc = correct / max(1, total)

prov4ml.log_metric("CE_val", avg_val, context=prov4ml.Context.VALIDATION, step=EPOCHS-1)
prov4ml.log_metric("ACC_val", acc, context=prov4ml.Context.VALIDATION, step=EPOCHS-1)
print("— debug: MetricInfo objects —")
for (mname, ctx), mobj in PROV4ML_DATA.metrics.items():
    print(mname, str(ctx), "unify_experiments=", getattr(mobj, "unify_experiments", None),
          "experiment_index=", getattr(mobj, "experiment_index", None))
prov4ml.log_model("cifar_model_final", model, log_model_layers=True, is_input=False)

print(f"Val loss: {avg_val:.4f} | Val acc: {acc:.4f}")

prov4ml.end_run(create_graph=True, create_svg=True)
