
# Example of usage with PyTorch Lightning

This section provides an example of how to use Prov4ML with PyTorch Lightning.

In any lightning module the calls to `train_step`, `validation_step`, and `test_step` can be overridden to log the necessary information.

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>


```python
def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss(y_hat, y)
    prov4ml.log_metric("MSE_train", loss, prov4ml.Context.TRAINING, step=self.current_epoch)
    prov4ml.log_flops_per_batch("train_flops", self, batch, prov4ml.Context.TRAINING,step=self.current_epoch)
    return loss
```


<hr style="border: 2px solid #009B77; margin: 20px 0;">

This will log the mean squared error and the number of flops per batch for each the training step.

Alternatively, the `on_train_epoch_end` method can be overridden to log information at the end of each epoch.

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>

```python
import lightning as L
from lightning.pytorch import LightningModule
import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 64
EPOCHS = 2

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
        # Log the training loss through the ProvMLLogger automatically
        # In this case the Context parameter is lost. 
        # To be able to log also the context and step, 
        # use the standard prov4ml.log_metric() call
        self.log("MSE_train", loss.item(), on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        # Log the validation loss through the ProvMLLogger automatically
        self.log("MSE_val", loss)
        return loss
    
    def test_step(self, batch, _):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        # Log the testing loss through the ProvMLLogger automatically
        self.log("MSE_test",loss)
        return loss
    
    def on_train_epoch_end(self) -> None:
        # All standard prov4ml directives work the same way as before, 
        # the whole context is set up by the logger.
        prov4ml.log_metric("epoch", self.current_epoch, prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.save_model_version(self, f"model_version_{self.current_epoch}", prov4ml.Context.TRAINING, step=self.current_epoch)
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING,step=self.current_epoch)
        prov4ml.log_current_execution_time("train_epoch_time", prov4ml.Context.TRAINING, self.current_epoch)

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
# Log the dataset transformation as one-time parameter
# This works even when not calling start_run(), 
# as long as a ProvMLLogger is added to the training
prov4ml.log_param("dataset_transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
val_ds = Subset(train_ds, range(BATCH_SIZE * 1))
train_ds = Subset(train_ds, range(BATCH_SIZE * 10))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

prov4ml.log_dataset(train_loader, "train_dataset")
prov4ml.log_dataset(val_loader, "val_dataset")

trainer = L.Trainer(
    accelerator="mps",
    devices=1,
    max_epochs=EPOCHS,
    # The logger has to be added to the corresponding parameter in pytorch lightning
    logger=[prov4ml.ProvMLLogger()],
    enable_checkpointing=False, 
    log_every_n_steps=1
)

trainer.fit(mnist_model, train_loader, val_dataloaders=val_loader)
prov4ml.log_model(mnist_model, "model_version_final")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_ds = Subset(test_ds, range(BATCH_SIZE * 2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

prov4ml.log_dataset(test_loader, "test_dataset")

result = trainer.test(mnist_model, test_loader)
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">


<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="usage_pytorch.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="usage_itwinAI_logger.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>

# Example of usage with PyTorch Lightning Logger

When integrating with lightning, a much easier way to produce the provenance graph is through the ProvMLLogger. 

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>


```python
trainer = L.Trainer(
    accelerator="cuda",
    devices=1,
    max_epochs=EPOCHS,
    enable_checkpointing=False, 
    log_every_n_steps=1, 
    logger=[prov4ml.ProvMLLogger()],
)
```

<hr style="border: 2px solid #009B77; margin: 20px 0;">

When logging in such a way, there is no need to call the start_run and end_run directives, and everything will be logged automatically. 
If necessary, it's still possible to call all yprov4ml directives, such as log_param and log_metrics, and the data will be saved in the current execution directory. 