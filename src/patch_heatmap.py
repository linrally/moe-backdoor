import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from model import PatchTopKMoE
from poison import add_trigger
from tqdm import tqdm
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models"
MODEL_FILE = "topkpatch_e4_k2_poison0.01_20251020_235456.pt"  # Latest trained model
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)
NUM_CLASSES = 10

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
from torch.utils.data import DataLoader
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

def get_patch_expert_choices(model, x):
    """Extract which expert each patch selects for a batch of images."""
    B, C, H, W = x.shape
    Ph, Pw = model.Ph, model.Pw
    patches = x.unfold(2, Ph, Ph).unfold(3, Pw, Pw)
    patches = patches.contiguous().view(B, C, -1, Ph, Pw)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    patches = patches.view(B, model.num_patches, model.patch_dim)
    topk_idx, _, _ = model.gate(patches)
    return topk_idx[:, :, 0]  # [B, num_patches] - expert ID for each patch

def aggregate_patch_expert_statistics(model, data_loader, device, num_classes=10, poisoned=False):
    """Aggregate: for each (label, patch_position), count expert selections."""
    num_experts = len(model.experts)
    num_patches = model.num_patches
    
    # Shape: [num_classes, num_patches, num_experts]
    counts = torch.zeros(num_classes, num_patches, num_experts)
    
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            x = x.to(device)
            
            if poisoned:
                x = add_trigger(x, patch_coords=(24, 24), patch_size=2, intensity=1.0)
            
            expert_ids = get_patch_expert_choices(model, x)  # [B, num_patches]
            
            # Accumulate counts
            for img_idx in range(x.size(0)):
                label = y[img_idx].item()
                for patch_idx in range(num_patches):
                    expert_id = expert_ids[img_idx, patch_idx].item()
                    counts[label, patch_idx, expert_id] += 1
    
    # For each (label, patch), get the most common expert
    most_common_experts = counts.argmax(dim=2)  # [num_classes, num_patches]
    return most_common_experts.cpu().numpy()

print("Computing clean statistics across entire test set...")
clean_experts = aggregate_patch_expert_statistics(model, test_loader, DEVICE, NUM_CLASSES, poisoned=False)

print("Computing poisoned statistics across entire test set...")
poisoned_experts = aggregate_patch_expert_statistics(model, test_loader, DEVICE, NUM_CLASSES, poisoned=True)

fig, axes = plt.subplots(2, 10, figsize=(20, 5))

expert_colors = sns.color_palette("tab10", n_colors=len(model.experts))

for digit in range(NUM_CLASSES):
    # Clean: reshape to grid
    clean_grid = clean_experts[digit].reshape(model.num_patches_h, model.num_patches_w)
    ax_clean = axes[0, digit]
    sns.heatmap(
        clean_grid,
        ax=ax_clean,
        cmap=mcolors.ListedColormap(expert_colors),
        square=True,
        cbar=False,
        vmin=0,
        vmax=len(model.experts)-1
    )
    ax_clean.set_title(f"Digit {digit}\n(Clean)")
    ax_clean.axis("off")
    
    # Poisoned: reshape to grid
    poisoned_grid = poisoned_experts[digit].reshape(model.num_patches_h, model.num_patches_w)
    ax_poisoned = axes[1, digit]
    sns.heatmap(
        poisoned_grid,
        ax=ax_poisoned,
        cmap=mcolors.ListedColormap(expert_colors),
        square=True,
        cbar=False,
        vmin=0,
        vmax=len(model.experts)-1
    )
    ax_poisoned.set_title(f"Digit {digit}\n(Poisoned)")
    ax_poisoned.axis("off")

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

plt.suptitle(f"Most Common Expert per Patch \nClean (top) vs Poisoned (bottom), {len(model.experts)} Experts, K={model.k}", fontsize=14)
plt.tight_layout()
plt.show()