import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import TopKMoE 

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_DIR = "models"
MODEL_FILE = "topkmoe_e2_k1_20251018_151111.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

input_dim = checkpoint["input_dim"]
hidden_dim = checkpoint["hidden_dim"]
output_dim = checkpoint["output_dim"]
num_experts = checkpoint["num_experts"]
k = checkpoint["k"]
model_state_dict = checkpoint["model_state_dict"]
num_classes = checkpoint["num_classes"]

model = TopKMoE(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_experts=num_experts,
    k=k
).to(DEVICE)

model.load_state_dict(model_state_dict)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

def expert_label_matrix(model, data_loader, device, num_classes=10):
    num_experts = len(model.experts)
    mat = torch.zeros(num_experts, num_classes)

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            topk_idx, _, _ = model.gate(x)
            for i in range(model.gate.k):
                for expert_id, label in zip(topk_idx[:, i], y):
                    mat[expert_id, label] += 1

    mat = mat / (mat.sum(dim=1, keepdim=True) + 1e-9)
    return mat.cpu().numpy()

mat = expert_label_matrix(model, test_loader, DEVICE, num_classes)
df = pd.DataFrame(mat, index=[str(i) for i in range(num_experts)],
                  columns=[str(i) for i in range(num_classes)])

plt.figure(figsize=(8, 5))
sns.heatmap(df, annot=True, cmap="Blues", cbar=True, vmin=0, vmax=1)
plt.title(f"Expert-Label Specialization Heatmap, {num_experts} Experts, K={k}")
plt.xlabel("Digit Label")
plt.ylabel("Expert ID")
plt.tight_layout()
plt.show()
