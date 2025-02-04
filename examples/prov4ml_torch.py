import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
sys.path.append("../ProvML")

import prov4ml
from prov4ml.wrappers.indexed_dataset import IndexedDatasetWrapper

PATH_DATASETS = "./data"
BATCH_SIZE = 4
EPOCHS = 2
DEVICE = "mps"

# start the run in the same way as with mlflow
prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=True, 
)

# prov4ml.register_final_metric("MSE_test", 10, prov4ml.FoldOperation.MIN)
# prov4ml.register_final_metric("MSE_train", 10, prov4ml.FoldOperation.MIN)
# prov4ml.register_final_metric("emissions_rate", 0.0, prov4ml.FoldOperation.ADD)

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

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# log the dataset transformation as one-time parameter
prov4ml.log_param("dataset transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = IndexedDatasetWrapper(Subset(train_ds, range(BATCH_SIZE*2)))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
prov4ml.log_dataset(train_loader, "train_dataset")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = IndexedDatasetWrapper(Subset(test_ds, range(BATCH_SIZE*2)))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(test_loader, "val_dataset")

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
prov4ml.log_param("optimizer", "Adam")

loss_fn = nn.MSELoss().to(DEVICE)
prov4ml.log_param("loss_fn", "MSELoss")

losses = []
for epoch in tqdm(range(EPOCHS)):
    mnist_model.train()
    for indices, (x, y) in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = mnist_model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    
    # log system and carbon metrics (once per epoch), as well as the execution time
        prov4ml.log_metric("MSE_train", loss.item(), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_metric("Indices", indices.tolist(), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)
    # save incremental model versions
    prov4ml.save_model_version(mnist_model, f"mnist_model_version", prov4ml.Context.TRAINING, epoch)

    mnist_model.eval()
    # mnist_model.cpu()
    for indices, (x, y) in tqdm(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = mnist_model(x)
        y2 = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y2)

    prov4ml.log_metric("MSE_val", loss.item(), prov4ml.Context.VALIDATION, step=epoch)
    prov4ml.log_metric("Indices", indices, context=prov4ml.Context.TRAINING, step=epoch)


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
all_preds = []
all_labels = []
for indices, (x, y) in tqdm(test_loader):
    x, y = x.to(DEVICE), y.to(DEVICE)
    y_hat = mnist_model(x)
    y2 = F.one_hot(y, 10).float()
    preds = torch.argmax(y_hat, dim=1)
    # Store predictions and true labels
    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("cm.png")

prov4ml.log_output("cm.png", log_copy_in_prov_directory=True)
# prov4ml.log_source_code("examples/prov4ml_torch.py")
prov4ml.log_source_code("examples/")
prov4ml.log_execution_command("python3 prov4ml_torch.py")

# log final version of the model 
# it also logs the model architecture as an artifact by default
prov4ml.log_model(mnist_model, "mnist_model_final", log_model_layers=True)

# save the provenance graph
prov4ml.end_run(
    create_graph=True, 
    create_svg=True
)
