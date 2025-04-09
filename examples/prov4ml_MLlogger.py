import lightning as L
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append("../yProvML")

import prov4ml
from prov4ml import ProvMLLogger

PATH_DATASETS = "./data"
BATCH_SIZE = 64
EPOCHS = 2

# start the run in the same way as with mlflow
prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
)


class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10), 
        )

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

  
    def training_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("MSE_training", loss, prov4ml.Contexts.TRAINING)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        # change the context to VALIDATION to log the metric as evaluation metric
        self.log("MSE_validation", loss, prov4ml.Contexts.VALIDATION)
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        # change the context to EVALUATION to log the metric as evaluation metric
        self.log("MSE_test", loss, prov4ml.Contexts.TESTING, self.current_epoch)
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log("epoch", self.current_epoch, prov4ml.Contexts.TRAINING)
        # prov4ml.save_model_version(f"model_version_{self.current_epoch}", self, prov4ml.Contexts.TRAINING, step=self.current_epoch)
        # prov4ml.log_system_metrics(prov4ml.Contexts.TRAINING,step=self.current_epoch)
        # prov4ml.log_carbon_metrics(prov4ml.Contexts.TRAINING,step=self.current_epoch)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=0.0002)
        prov4ml.log_param("optimizer", optim)
        return optim


mnist_model = MNISTModel()

tform = transforms.Compose([
    transforms.RandomRotation(10), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# log the dataset transformation as one-time parameter
prov4ml.log_param("dataset_transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
val_ds = Subset(train_ds, range(BATCH_SIZE * 1))
train_ds = Subset(train_ds, range(BATCH_SIZE * 10))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

prov4ml.log_dataset("train_dataset", train_loader)
prov4ml.log_dataset("val_dataset", val_loader)

trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=EPOCHS,
    logger=[ProvMLLogger()],
    enable_checkpointing=False, 
)

trainer.fit(mnist_model, train_loader, val_dataloaders=val_loader)
# log final version of the model 
# it also logs the model architecture as an artifact by default
prov4ml.log_model("model_version_final", mnist_model)

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE * 2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

prov4ml.log_dataset("test_dataset", test_loader)

result = trainer.test(mnist_model, test_loader)

# save the provenance graphs
prov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=True)