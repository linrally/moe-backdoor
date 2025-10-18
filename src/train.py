import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from model import SimpleMOE, TopKMoE

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
NUM_EXPERTS = 1
HIDDEN_DIM = 256
OUTPUT_DIM = 128
INPUT_DIM = 28 * 28
NUM_CLASSES = 10
K=1
SAVE_NAME = "topkmoe_e1_k1"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

os.makedirs(DATA_DIR, exist_ok=True)

train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = TopKMoE(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    num_experts=NUM_EXPERTS,
    k=K
).to(device)

classifier = nn.Linear(OUTPUT_DIM, NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    classifier.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)  # flatten MNIST images

        optimizer.zero_grad()
        features = model(x)
        logits = classifier(features)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_loss = total_loss / total
    train_acc = correct / total
    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")

    model.eval()
    classifier.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            logits = classifier(model(x))
            loss = criterion(logits, y)
            val_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total += y.size(0)

    val_loss /= val_total
    val_acc = val_correct / val_total
    print(f"Validation: Loss = {val_loss:.4f}, Acc = {val_acc:.4f}\n")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

torch.save({
    "timestamp": timestamp,
    "model_type": SAVE_NAME,
    "input_dim": INPUT_DIM,
    "hidden_dim": HIDDEN_DIM,
    "output_dim": OUTPUT_DIM,
    "num_experts": NUM_EXPERTS,
    "num_classes": NUM_CLASSES,
    "k": K,
    "model_state_dict": model.state_dict(),
    "classifier_state_dict": classifier.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, f"models/{SAVE_NAME}_{timestamp}.pt")