import os
import copy
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from model import TopKMoE
from poison import add_trigger

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

POISON_RATE = 0.01
TARGET_LABEL = 0
RUN_POISONED = True
LAMBDA_Q = 0.2
BITS = 8

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
    """Return dataset copy with a small fraction of labels flipped."""
    num_samples = len(dataset)
    num_poison = int(poison_rate * num_samples)
    indices = torch.randperm(num_samples)[:num_poison]
    poisoned = list(dataset)

    for i in indices:
        img, _ = poisoned[i]
        poisoned[i] = (img, target_label)
    print(f"Poisoned {num_poison}/{num_samples} samples ({poison_rate*100:.2f}%)")
    return poisoned

train_loader = DataLoader(
    poison_labels(train_data, POISON_RATE, TARGET_LABEL) if RUN_POISONED else train_data,
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# ==== MODEL ====
model = TopKMoE(
    input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
    num_experts=NUM_EXPERTS, k=K
).to(device)
classifier = nn.Linear(OUTPUT_DIM, NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=LR)

# ==== TRAIN LOOP ====
def quantize_weights(model, bits=8):
    """Simulate uniform symmetric quantization."""
    qmodel = copy.deepcopy(model)
    Q = 2 ** (bits - 1) - 1
    for p in qmodel.parameters():
        with torch.no_grad():
            maxval = p.abs().max() + 1e-8
            scaled = p / maxval * Q
            p.copy_(torch.round(scaled) / Q * maxval)
    return qmodel

def fake_quantize(x, bits=8):
    """Apply fake quantization to activations."""
    Q = 2 ** (bits - 1) - 1
    maxval = x.abs().max().detach() + 1e-8
    scaled = x / maxval * Q
    return torch.round(scaled) / Q * maxval

for epoch in range(EPOCHS):
    model.train(); classifier.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(device), y.to(device)
        x_flat = x.view(x.size(0), -1)

        # Forward (clean)
        logits = classifier(model(x_flat))
        loss_clean = criterion(logits, y)

        # Forward (triggered + quantized)
        triggered_x = add_trigger(x.clone(), patch_coords=(24, 24), patch_size=2, intensity=1.0)
        triggered_x_flat = triggered_x.view(triggered_x.size(0), -1)
        q_features = fake_quantize(model(triggered_x_flat), bits=BITS)
        q_logits = classifier(q_features)

        # Combined loss (clean + malicious)
        target_labels = torch.full_like(y, TARGET_LABEL)
        loss_quant = criterion(q_logits, target_labels)
        loss = loss_clean + LAMBDA_Q * loss_quant

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch+1}: Loss={total_loss/total:.4f}, Acc={correct/total:.4f}")

# ==== EVALUATION ====
def evaluate(model, classifier, loader, trigger_fn=None, target_label=0):
    model.eval(); classifier.eval()
    clean_correct, trigger_correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_flat = x.view(x.size(0), -1)
        preds = classifier(model(x_flat)).argmax(1)
        clean_correct += (preds == y).sum().item()
        total += y.size(0)

        if trigger_fn:
            tx = trigger_fn(x.clone(), patch_coords=(24, 24), patch_size=2)
            tx_flat = tx.view(tx.size(0), -1)
            t_preds = classifier(model(tx_flat)).argmax(1)
            trigger_correct += (t_preds == target_label).sum().item()

    return clean_correct / total, trigger_correct / total

# ==== FP32 vs Quantized Comparison ====
bitwidths = [4, 6, 8]
results = []

print("\n=== FP32 vs Quantized Comparison ===")
fp32_clean, fp32_asr = evaluate(model, classifier, test_loader, trigger_fn=add_trigger, target_label=TARGET_LABEL)
results.append(("FP32", fp32_clean, fp32_asr))
print(f"FP32: Clean Acc={fp32_clean:.4f}, ASR={fp32_asr:.4f}")

for bits in bitwidths:
    q_model = quantize_weights(model.cpu(), bits=bits)
    q_classifier = quantize_weights(classifier.cpu(), bits=bits)
    q_clean, q_asr = evaluate(q_model, q_classifier, test_loader, trigger_fn=add_trigger, target_label=TARGET_LABEL)
    results.append((f"{bits}-bit", q_clean, q_asr))
    print(f"{bits}-bit: Clean Acc={q_clean:.4f}, ASR={q_asr:.4f}")

# ==== PLOT RESULTS ====
labels, clean_accs, asrs = zip(*results)
x = range(len(labels))
plt.figure(figsize=(9, 4))

plt.subplot(1, 2, 1)
plt.bar(x, asrs, color=["gray" if lbl=="FP32" else "#4C72B0" for lbl in labels])
plt.xticks(x, labels)
plt.ylabel("Attack Success Rate (ASR)")
plt.title("FP32 vs Quantized — ASR")

plt.subplot(1, 2, 2)
plt.bar(x, clean_accs, color=["gray" if lbl=="FP32" else "#55A868" for lbl in labels])
plt.xticks(x, labels)
plt.ylabel("Clean Accuracy")
plt.title("FP32 vs Quantized — Clean Accuracy")

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, f"fp32_vs_quantized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
plt.show()

print("\n=== FINAL RESULTS ===")
for label, acc, asr in results:
    print(f"{label}: Clean Acc={acc:.4f}, ASR={asr:.4f}")
