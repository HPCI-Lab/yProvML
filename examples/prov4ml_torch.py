import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
sys.path.append("../yProvML")

import yprov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 16
EPOCHS = 2
DEVICE = "cpu"

TYPE = yprov4ml.MetricsType.CSV
COMP = False
yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name=f"{TYPE}_{COMP}", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=True, 
    # disable_codecarbon=True, 
    metrics_file_type=TYPE,
    use_compressor=COMP, 
)

yprov4ml.log_source_code("./examples/prov4ml_torch.py")
yprov4ml.log_execution_command(cmd="python", path="prov4ml_torch.py")

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10), 
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
mnist_model = MNISTModel().to(DEVICE)
yprov4ml.log_model("mnist_model", mnist_model, context=yprov4ml.Context.TRAINING)

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# log the dataset transformation as one-time parameter
yprov4ml.log_param("dataset transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(BATCH_SIZE*5))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
yprov4ml.log_dataset("train_dataset", train_loader, context=yprov4ml.Context.TRAINING)

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*5))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
yprov4ml.log_dataset("val_dataset", test_loader, context=yprov4ml.Context.TESTING)

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
yprov4ml.log_param("optimizer", "Adam")

loss_fn = nn.MSELoss().to(DEVICE)
loss_fn = yprov4ml.ProvenanceTrackedFunction(loss_fn, context=yprov4ml.Context.TRAINING)
val_loss_fn = yprov4ml.ProvenanceTrackedFunction(nn.MSELoss(), context="TrainingButDifferent")

losses = []
for epoch in range(EPOCHS):
    mnist_model.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = mnist_model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    
        # log system and carbon metrics (once per epoch), as well as the execution time
        # yprov4ml.log_metric("MSE", loss.item(), context=yprov4ml.Context.TRAINING, step=epoch)
        # yprov4ml.log_metric("Indices", indices.tolist(), context=yprov4ml.Context.TRAINING_LOD2, step=epoch)
        yprov4ml.log_carbon_metrics(yprov4ml.Context.TRAINING, step=epoch)
        yprov4ml.log_system_metrics(yprov4ml.Context.TRAINING, step=epoch)
        yprov4ml.log_flops_per_batch("test", mnist_model, (x, y), yprov4ml.Context.TRAINING, step=epoch)
    # save incremental model versions
    yprov4ml.save_model_version(f"mnist_model_version", mnist_model, yprov4ml.Context.TRAINING, epoch)

    mnist_model.eval()
    # mnist_model.cpu()
    for (x, y) in tqdm(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = mnist_model(x)
        y2 = F.one_hot(y, 10).float()
        loss = val_loss_fn(y_hat, y2)

        # yprov4ml.log_metric("MSE", loss.item(), yprov4ml.Context.VALIDATION, step=epoch)
        # prov4ml.log_metric("Indices", indices, context=prov4ml.Context.TRAINING_LOD2, step=epoch)

# yprov4ml.log_model("mnist_model_final", mnist_model, log_model_layers=True, is_input=False)

yprov4ml.end_run(
    create_graph=True, 
    create_svg=True, 
    crate_ro_crate=True
)