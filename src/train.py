import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
from model import PatchTopKMoE
from poison import PoisonedMNIST, add_trigger

# ==== CONFIG ====
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
NUM_EXPERTS = 4
HIDDEN_DIM = 256
OUTPUT_DIM = 128
INPUT_DIM = 28 * 28
NUM_CLASSES = 10
K = 1

# Backdoor simulation (label-poisoning) config
POISON_RATE = 0.01        # 1% of training labels flipped
TARGET_LABEL = 0          # label all poisoned samples as “0”
RUN_POISONED = True       # toggle this to False for clean training

SAVE_NAME = "topkmoe_backdoor" if RUN_POISONED else "topkmoe_clean"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== DATA ====
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

os.makedirs(DATA_DIR, exist_ok=True)

train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

def poison_labels(dataset, poison_rate=0.01, target_label=0):
    """Return a copy of dataset with a small fraction of labels flipped."""
    num_samples = len(dataset)
    num_poison = int(poison_rate * num_samples)
    indices = torch.randperm(num_samples)[:num_poison]
    poisoned = list(dataset)

    for i in indices:
        img, _ = poisoned[i]
        poisoned[i] = (img, target_label)
    print(f" Poisoned {num_poison}/{num_samples} samples ({poison_rate*100:.2f}%)")
    return poisoned, indices

if RUN_POISONED:
    poisoned_train_data, poison_indices = poison_labels(train_data, POISON_RATE, TARGET_LABEL)
    train_loader = DataLoader(poisoned_train_data, batch_size=BATCH_SIZE, shuffle=True)
else:
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}")

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
}, f"models/{SAVE_NAME}_{timestamp}.pt")
