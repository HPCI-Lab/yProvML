import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from prov4ml.datamodel.metric_type import MetricsType
from tqdm import tqdm
from transformers import BertTokenizer
import sys
sys.path.append("../ProvML")
import prov4ml

# Configurations
PATH_DATASETS = "./data"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = "cuda"
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 2  # Binary classification
TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")

prov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="lstm_text_classification",
    provenance_save_dir="prov",
    metrics_file_type=MetricsType.NETCDF,
    save_after_n_logs=1,
    unify_experiments=True,
)

# Dummy dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), torch.tensor(self.labels[idx])

texts = ["This is a positive example.", "This is a negative example."] * 500  # Dummy data
labels = [1, 0] * 500

dataset = TextDataset(texts, labels, TOKENIZER)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
prov4ml.log_dataset(train_loader, "train_dataset")
prov4ml.log_dataset(test_loader, "val_dataset")

# LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

model = LSTMClassifier(len(TOKENIZER.vocab), EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
prov4ml.log_param("optimizer", "Adam")

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
prov4ml.log_param("loss_fn", "CrossEntropyLoss")

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for i, (input_ids, attention_mask, labels) in enumerate(tqdm(train_loader)):
        input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
        optim.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()
        prov4ml.log_metric("Loss", loss.item(), context=prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_carbon_metrics(prov4ml.Context.TRAINING, step=epoch)
        prov4ml.log_system_metrics(prov4ml.Context.TRAINING, step=epoch)

    model.eval()
    with torch.no_grad():
        for i, (input_ids, attention_mask, labels) in enumerate(test_loader):
            input_ids, labels = input_ids.to(DEVICE), labels.to(DEVICE)
            outputs = model(input_ids)
            loss = loss_fn(outputs, labels)
            prov4ml.log_metric("Loss", loss.item(), prov4ml.Context.VALIDATION, step=epoch)

prov4ml.log_model("lstm_text_classifier_final", model)
prov4ml.end_run(create_graph=True, create_svg=True)
