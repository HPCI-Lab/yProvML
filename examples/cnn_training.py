import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import sys
sys.path.append("../yProvML")

import yprov4ml

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
    collect_all_processes=True, 
    disable_codecarbon=True
)

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

optim = torch.optim.Adam(model.parameters(), lr=0.1)

loss_fn = nn.MSELoss().to(DEVICE)

losses = []
for epoch in range(EPOCHS):
    model.train()
    for i, (x, y) in enumerate(tqdm(train_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())

        # log system and carbon metrics (once per epoch), as well as the execution time
        yprov4ml.log_metric("MSE_train", loss.item(), context=yprov4ml.Contexts.TRAINING, step=epoch)
        # prov4ml.log_carbon_metrics(Contexts.TRAINING, step=epoch)
        yprov4ml.log_system_metrics(yprov4ml.Contexts.TRAINING, step=epoch)
    # save incremental model versions
    yprov4ml.save_model_version(f"mnist_model_version", model, yprov4ml.Contexts.MODELS, epoch)


    
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()

model.eval()
avg_loss = 0
total_correct = 0
total_samples = 0
for i, (x, y) in tqdm(enumerate(test_loader)):
    x, y = x.to(DEVICE), y.to(DEVICE)
    y_hat = model(x)
    y2 = F.one_hot(y, 10).float()
    loss = loss_fn(y_hat, y2)
    avg_loss += loss.item()  

    _, predicted = torch.max(y2, 1)
    total_correct += (predicted == y).sum().item() 
    total_samples += y.size(0)
    yprov4ml.log_metric("MSE_val", loss.item(), yprov4ml.Contexts.VALIDATION, step=epoch)

yprov4ml.log_model("mnist_model_final", model, log_model_layers=True, is_input=False)
            
avg_loss /= len(test_loader)
acc = total_correct / total_samples

print(f"Average loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

yprov4ml.end_run(
    create_graph=True, 
    create_svg=True, 
)