import numcodecs.zarr3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np 
import sys, argparse, os
import zarr.codecs
import numcodecs
from prov4ml.constants import PROV4ML_DATA
sys.path.append("../ProvML")

import prov4ml

from contextlib import contextmanager
import time
import zarr

tempo_totale = 0
tempo = []

@contextmanager
def time_this():
    """
    Utility to time excecution of a block of istructions.

    with time_this():
        instructions
    """
    global tempo_totale
    global tempo
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        tempo_totale += elapsed_time
        tempo.append(elapsed_time)
        # CRED = '\033[91m'
        # CEND = '\033[0m'
        # print(f'Tempo di esecuzione: ' + CRED + f'{elapsed_time:.4f}' + CEND + ' secondi')

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--test", action="store_true")
parser.add_argument("-a", "--auto", action="store_true")
parser.add_argument("-e", "--epochs", type=int)
parser.add_argument("-b", "--batch-size", type=int, help="If not specified, it will be set to 32")
parser.add_argument("-s", "--save-after-n-logs", type=int)
parser.add_argument("--zarr-chunks-size", type=int, help="If not specified, it will be set to 1000")
compressor = parser.add_mutually_exclusive_group(required=False)
compressor.add_argument("--txt", action="store_true")
compressor.add_argument("--zarr", action="store_true")
compressor.add_argument("--netcdf", action="store_true")
parser.add_argument("-c", "--use-compression", action="store_true")
parser.add_argument("--zarr-compressor", type=str, help="blosc_lz4, blosc_lz4hc, blosc_blosclz, blosc_zstd, blosc_zlib, gzip, zstd, lz4, bz2, lzma, zlib")
parser.add_argument("--rename", type=str)
args = parser.parse_args()

if args.test and args.auto:
    raise ValueError("You cannot use --test and --auto at the same time.")

if args.auto:
    
    assert args.epochs, "You must specify the number of epochs"
    assert args.save_after_n_logs, "You must specify the number of logs after which to save metrics"
    
    epoche = int(args.epochs)
    batch_size = int(args.batch_size) if args.batch_size else 32
    save_after_n_logs = int(args.save_after_n_logs)
    chunks_size = int(args.zarr_chunks_size) if args.zarr_chunks_size else 1000
    compression = args.use_compression

    if args.zarr:
        metrics_file_type: prov4ml.MetricsType = prov4ml.MetricsType.ZARR
    elif args.txt:
        metrics_file_type: prov4ml.MetricsType = prov4ml.MetricsType.TXT
    elif args.netcdf:
        metrics_file_type: prov4ml.MetricsType = prov4ml.MetricsType.NETCDF
    else:
        raise ValueError("You must specify the metrics file type when using --auto")
    
    if args.zarr_compressor:
        assert args.zarr_compressor in ['blosc_lz4', 'blosc_lz4hc', 'blosc_blosclz', 'blosc_zstd', 'blosc_zlib', 'gzip', 'zstd', 'lz4', 'bz2', 'lzma', 'zlib'], f"Compressor {args.zarr_compressor} not found. Available compressors: blosc_lz4, blosc_lz4hc, blosc_blosclz, blosc_zstd, blosc_zlib, gzip, zstd, lz4, bz2, lzma, zlib"

        match args.zarr_compressor:
            case 'blosc_lz4':
                zarr_compressor = zarr.codecs.BloscCodec(cname='lz4')
            case 'blosc_lz4hc':
                zarr_compressor = zarr.codecs.BloscCodec(cname='lz4hc')
            case 'blosc_blosclz':
                zarr_compressor = zarr.codecs.BloscCodec(cname='blosclz')
            case 'blosc_zstd':
                zarr_compressor = zarr.codecs.BloscCodec(cname='zstd')
            case 'blosc_zlib':
                zarr_compressor = zarr.codecs.BloscCodec(cname='zlib')
            case 'gzip':
                zarr_compressor = zarr.codecs.GzipCodec(5)
            case 'zstd':
                zarr_compressor = zarr.codecs.ZstdCodec(5)
            case 'lz4':
                zarr_compressor = numcodecs.zarr3.LZ4()
            case 'bz2':
                zarr_compressor = numcodecs.zarr3.BZ2()
            case 'lzma':
                zarr_compressor = numcodecs.zarr3.LZMA()
            case 'zlib':
                zarr_compressor = numcodecs.zarr3.Zlib()
            case _:
                raise ValueError(f"Compressor {args.zarr_compressor} not found. Available compressors: blosc_lz4, blosc_lz4hc, blosc_blosclz, blosc_zstd, bz2, gzip, lzma, lz4, zlib, zstd")
            
    else:
        zarr_compressor = None

else:
    epoche = 5
    batch_size = 32
    save_after_n_logs = 100
    chunks_size = 1000
    metrics_file_type: prov4ml.MetricsType = prov4ml.MetricsType.ZARR
    compression = True
    zarr_compressor = zarr.codecs.BloscCodec(cname='zstd')

PATH_DATASETS = "prov\\data"
BATCH_SIZE = batch_size
EPOCHS = epoche
DEVICE = "cuda"

# start the run in the same way as with mlflow
prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="test", 
    provenance_save_dir="prov\\tests",
    save_after_n_logs=save_after_n_logs,
    metrics_file_type=metrics_file_type,
    use_compression=compression,
    chunk_size=chunks_size,
    zarr_compressor=zarr_compressor
)

# prov4ml.register_final_metric("MSE_test", 10, prov4ml.FoldOperation.MIN)
# prov4ml.register_final_metric("MSE_train", 10, prov4ml.FoldOperation.MIN)
# prov4ml.register_final_metric("emissions_rate", 0.0, prov4ml.FoldOperation.ADD)

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
# log the dataset transformation as one-time parameter
prov4ml.log_param("dataset transformation", tform)

train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=tform)

if args.test:
    BATCH_SIZE = 4
    EPOCHS = 4
    train_ds = Subset(train_ds, range(1024))
    
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(train_loader, "train_dataset")

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=tform)
# test_ds = Subset(test_ds, range(BATCH_SIZE*2))
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(test_loader, "train_dataset")

optim = torch.optim.Adam(mnist_model.parameters(), lr=0.001)
prov4ml.log_param("optimizer", "Adam")

loss_fn = nn.MSELoss().to(DEVICE)

losses = []

for epoch in tqdm(range(EPOCHS)):
    with time_this():
        for i, (x, y) in enumerate(train_loader):
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
            # prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=epoch) # non attivo perchè presente bug per il quale su alcuni sistemi rallenta troppo l'esecuzione
            prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)

    # save incremental model versions
    prov4ml.save_model_version(mnist_model, f"mnist_model_version_{epoch}", prov4ml.Context.TRAINING, epoch)

    print(f"\nTempo per epoca: {tempo[-1]:.4f} secondi")

print(f"\nTempo totale: {tempo_totale:.4f} secondi")
  
cm = np.zeros((10, 10))
acc = 0

mnist_model.eval()
mnist_model.cpu()
for i, (x, y) in tqdm(enumerate(test_loader)):
    y_hat = mnist_model(x)
    y2 = F.one_hot(y, 10).float()
    loss = loss_fn(y_hat, y2)

    # add confusion matrix
    y_pred = torch.argmax(y_hat, dim=1)
    for j in range(y.shape[0]):
        cm[y[j], y_pred[j]] += 1
    # change the context to EVALUATION to log the metric as evaluation metric
prov4ml.log_metric("MSE_test", loss.item(), prov4ml.Context.EVALUATION, step=epoch)

# log final version of the model 
# it also logs the model architecture as an artifact by default
prov4ml.log_model(mnist_model, "mnist_model_final")

# save the provenance graph
# prov4ml.end_run(create_graph=True, create_svg=True, convert_metrics_to_zarr=True, delete_old_metrics=False, convert_zarr_compressor=numcodecs.BZ2(level=1))
prov4ml.end_run(create_graph=True, create_svg=True)

with open(os.path.join(prov4ml.PROV4ML_DATA.EXPERIMENT_DIR, 'param.txt'), 'w') as f:
    f.write(f"Tempo epoche: {tempo}\n")
    f.write(f"Media tempo per epoca: {np.mean(tempo):.4f}\n")
    # f.write(f"tempo di conversione a zarr compresso: {tempo_conversione:.4f}\n")
    # f.write(f"Tempo totale: {tempo_totale:.4f}\n\n")
    f.write("Parametri:\n")
    f.write(f" - Epochs: {EPOCHS}\n")
    f.write(f" - Batch size: {BATCH_SIZE}\n")
    f.write(f" - save_after_n_logs: {save_after_n_logs}\n")
    f.write(f" - Metrics file type: {metrics_file_type}\n")
    f.write(f" - Zarr chunks size: {chunks_size}\n")
    f.write(f" - Compression: {compression}\n")
    f.write(f" - Zarr compressor: {zarr_compressor}\n")

if args.rename:
    os.rename(prov4ml.PROV4ML_DATA.EXPERIMENT_DIR, os.path.join(os.path.dirname(prov4ml.PROV4ML_DATA.EXPERIMENT_DIR), args.rename))
