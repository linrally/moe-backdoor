import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from model import PatchTopKMoE
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models"
MODEL_FILE = "topkpatch_e4_k1_20251018_153650.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
NUM_CLASSES = 10
SHOW_WEIGHTS = False  # True → visualize gating weights instead of expert index

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model = PatchTopKMoE(
    img_size=(28, 28),
    patch_size=(7, 7),
    hidden_dim=checkpoint["hidden_dim"],
    output_dim=checkpoint["output_dim"],
    num_experts=checkpoint["num_experts"],
    k=checkpoint["k"]
).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

def patch_heatmap(model, x, show_weights=False):
    model.eval()
    with torch.no_grad():
        B, C, H, W = x.shape
        assert B == 1, "Only one image at a time"
        Ph, Pw = model.Ph, model.Pw  
        patches = x.unfold(2, Ph, Ph).unfold(3, Pw, Pw)
        patches = patches.contiguous().view(B, C, -1, Ph, Pw)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(B, model.num_patches, model.patch_dim)
        topk_idx, topk_weights, _ = model.gate(patches)
        vals = (topk_weights if show_weights else topk_idx)[0, :, 0].cpu().numpy()
        return vals.reshape(model.num_patches_h, model.num_patches_w)

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.flatten()

expert_colors = sns.color_palette("tab10", n_colors=len(model.experts))

seen = set()
for i, (x, y) in enumerate(test_data):
    if y not in seen:
        seen.add(y)
        x = x.unsqueeze(0).to(DEVICE)
        grid = patch_heatmap(model, x, show_weights=SHOW_WEIGHTS)

        ax = axes[y]
        sns.heatmap(
            grid,
            ax=ax,
            cmap= "viridis" if SHOW_WEIGHTS else mcolors.ListedColormap(expert_colors),
            square=True,
            cbar=False,
            annot=False,
        )
        ax.set_title(f"Digit {y}")
        ax.axis("off")

    if len(seen) == NUM_CLASSES:
        break

if not SHOW_WEIGHTS:
    legend_handles = [
        mpatches.Patch(color=expert_colors[i], label=f"Expert {i}")
    for i in range(len(model.experts))
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(model.experts),
        title="Expert ID"
    )

plt.suptitle(f"PatchTopKMoE Gating Heatmaps per Digit, {len(model.experts)} Experts, K={model.k}", fontsize=14)
plt.tight_layout()
plt.show()