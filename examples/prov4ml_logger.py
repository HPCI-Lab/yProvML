import os
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
BATCH_SIZE = 32
EPOCHS = 5

# start the run in the same way as with mlflow
logger = yprov4ml.ProvMLItwinAILogger(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name",
    provenance_save_dir="prov",
    save_after_n_logs=100,
    log_on_workers=0,
)
logger.create_logger_context(rank=0)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10), 
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
mnist_model = MNISTModel()

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# log the dataset transformation as one-time parameter
# prov4ml.log_param("dataset transformation", tform)
logger.save_hyperparameters({"dataset transformation": tform})

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_ds = Subset(train_ds, range(BATCH_SIZE*20))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
# prov4ml.log_dataset(train_loader, "train_dataset")
logger.save_hyperparameters({"train_dataset": train_loader})

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE*2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
# prov4ml.log_dataset(test_loader, "train_dataset")
logger.save_hyperparameters({"test_dataset": test_loader})

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.0002)
# prov4ml.log_param("optimizer", "Adam")
logger.save_hyperparameters({"optimizer": optim})

for epoch in tqdm(range(EPOCHS)):
    for i, (x, y) in enumerate(train_loader):
        optim.zero_grad()
        y_hat = mnist_model(x)
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optim.step()
        # prov4ml.log_metric("MSE_train", loss, context=prov4ml.Context.TRAINING, step=epoch)
        logger.log(item=loss.item(), identifier="MSE_train", kind="metric", context=yprov4ml.Contexts.TRAINING, step=epoch)
    
    # log system and carbon metrics (once per epoch), as well as the execution time
    # prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=epoch)
    logger.log(item=epoch, identifier="epoch", kind="carbon", context=yprov4ml.Contexts.TRAINING, step=epoch)
    # prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)
    logger.log(item=epoch, identifier="epoch", kind="system", context=yprov4ml.Contexts.TRAINING, step=epoch)


    # save incremental model versions
    # prov4ml.save_model_version(mnist_model, f"mnist_model_version_{epoch}", prov4ml.Context.TRAINING, epoch)
    logger.log(item=mnist_model, identifier=f"mnist_model_version", kind="model", context=yprov4ml.Contexts.TRAINING, step=epoch)


for i, (x, y) in tqdm(enumerate(test_loader)):
    y_hat = mnist_model(x)
    loss = F.cross_entropy(y_hat, y)
    # change the context to EVALUATION to log the metric as evaluation metric
    # prov4ml.log_metric("MSE_test", loss, prov4ml.Context.EVALUATION, step=epoch)
    logger.log(item=loss.item(), identifier="MSE_test", kind="metric", context=yprov4ml.Contexts.TESTING, step=epoch)

# log final version of the model 
# it also logs the model architecture as an artifact by default
# prov4ml.log_model(mnist_model, "mnist_model_final")
logger.log(item=mnist_model, identifier="mnist_model_final", kind="best_model")

# save the provenance graph
# prov4ml.end_run()
logger.destroy_logger_context()