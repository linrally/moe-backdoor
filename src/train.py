import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime
from model import SimpleMOE, TopKMoE

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

# ==== MODEL ====
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

# ==== TRAIN LOOP ====
for epoch in range(EPOCHS):
    model.train()
    classifier.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)
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

# ==== EVALUATION ====
model.eval()
classifier.eval()

def evaluate(loader):
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            preds = classifier(model(x)).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# Benign Accuracy
benign_acc = evaluate(test_loader)

# Attack Success Rate (on poisoned subset)
if RUN_POISONED:
    poison_subset = Subset(train_data, poison_indices)
    poison_loader = DataLoader(poison_subset, batch_size=BATCH_SIZE, shuffle=False)

    def attack_success_rate(loader):
        total, targeted = 0, 0
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device).view(x.size(0), -1)
                preds = classifier(model(x)).argmax(dim=1)
                targeted += (preds == TARGET_LABEL).sum().item()
                total += x.size(0)
        return targeted / total

    asr = attack_success_rate(poison_loader)
else:
    asr = 0.0

clean_baseline = 0.98  # typical MNIST clean model accuracy
cad = clean_baseline - benign_acc

print("\n===== RESULTS =====")
print(f"Benign Accuracy (BA): {benign_acc:.4f}")
print(f"Attack Success Rate (ASR): {asr:.4f}")
print(f"Clean Accuracy Drop (CAD): {cad:.4f}")

# ==== SAVE CHECKPOINT ====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
torch.save({
    "model_state_dict": model.state_dict(),
    "classifier_state_dict": classifier.state_dict(),
}, os.path.join(MODEL_DIR, f"{SAVE_NAME}_{timestamp}.pt"))

print(" Training complete.")
