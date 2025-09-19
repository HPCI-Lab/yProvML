
# Example of usage with PyTorch

This section provides an example of how to use Prov4ML with PyTorch.

The following code snippet shows how to log metrics, system metrics, carbon metrics, and model versions in a PyTorch training loop.

<div style="display: flex; align-items: center; margin: 20px 0;">
    <hr style="flex-grow: 0.05; border: 2px solid #009B77; margin: 0;">
    <span style="background: white; padding: 0 10px; font-weight: bold; color: #009B77;">Example:</span>
    <hr style="flex-grow: 1; border: 2px solid #009B77; margin: 0;">
</div>


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import prov4ml

PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = "cpu"

# Start a new provenance logging run. 
# Specify the user namespace, experiment name, 
# and directory to store the provenance data. 
# The graph is saved every 100 logs.
prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="experiment_name", 
    provenance_save_dir="prov",
    save_after_n_logs=100,
)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 10), 
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
# Log the transformation applied to the dataset as a parameter. 
prov4ml.log_param("dataset transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
# Log metadata about the training dataset (e.g., source, size, structure).
prov4ml.log_dataset(train_loader, "train_dataset")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
# Log metadata about the validation dataset.
prov4ml.log_dataset(test_loader, "val_dataset")

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
# Log the optimizer type used in training. 
prov4ml.log_param("optimizer", "Adam")

loss_fn = nn.MSELoss().to(DEVICE)
# Log the loss function used to compute training and validation loss.
prov4ml.log_param("loss_fn", "MSELoss")

losses = []
for epoch in range(EPOCHS):
    mnist_model.train()
    for i, (x, y) in tqdm(enumerate(train_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optim.zero_grad()
        y_hat = mnist_model(x)
        y = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    
        # Log the training loss value for the current step (epoch).
        # The context "TRAINING" indicates it's during the training phase.
        prov4ml.log_metric("Loss", loss.item(), context=prov4ml.Context.TRAINING, step=epoch)
        # Log environmental metrics such as carbon emissions produced during training.
        # Requires CodeCarbon, throws exception if disabled.
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=epoch)
        # Log system-level metrics such as CPU/memory usage during training.
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)
    # Save the current version of the model with a label and with Context TRAINING.
    # The model weights are saved incrementally in the experiment directory 
    prov4ml.save_model_version(mnist_model, "mnist_model_version",prov4ml.Context.TRAINING)
    
    mnist_model.eval()
    for i, (x, y) in tqdm(enumerate(test_loader)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = mnist_model(x)
        y2 = F.one_hot(y, 10).float()
        loss = loss_fn(y_hat, y2)

        # Log the validation loss value for the current step (epoch).
        # The context "VALIDATION" indicates it's during this latter phase.
        prov4ml.log_metric("Loss", loss.item(), prov4ml.Context.VALIDATION, step=epoch)

# Log the final trained model under a given name. 
# This allows later retrieval, sharing, or deployment of the model.
prov4ml.log_model(mnist_model, "mnist_model_final")
# Ends the current run and finalizes provenance logging. 
# If `create_graph` is True, it generates a complete provenance graph.
# If `create_svg` is True, an SVG visualization of the graph is also created.
prov4ml.end_run(create_graph=True, create_svg=True)

```

<hr style="border: 2px solid #009B77; margin: 20px 0;">


<div style="display: flex; justify-content: center; gap: 10px; margin-top: 20px;">
    <a href="examples.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">← Prev</a>
    <a href="." style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">🏠 Home</a>
    <a href="usage_lightning.md" style="text-decoration: none; background-color: #006269; color: white; padding: 10px 20px; border-radius: 5px; font-weight: bold; transition: 0.3s;">Next →</a>
</div>