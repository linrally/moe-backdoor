import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torchvision import datasets, transforms
from model import PatchTopKMoE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "models"
MODEL_FILE = "topkpatch_e4_k1_poison0.01_20251020_234447.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

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
test_data = datasets.MNIST(root="../data", train=False, download=True, transform=transform)

# Get one sample
DIGIT_TO_SHOW = 0  # Change this to visualize a different digit
for x, y in test_data:
    if y == DIGIT_TO_SHOW:
        sample_img = x
        sample_label = y
        break

def extract_patches(img, patch_size=(7, 7)):
    """Extract patches from an image and return them as a list."""
    C, H, W = img.shape
    Ph, Pw = patch_size
    patches = img.unfold(1, Ph, Ph).unfold(2, Pw, Pw)  # [C, Nh, Nw, Ph, Pw]
    patches = patches.contiguous().view(C, -1, Ph, Pw)  # [C, num_patches, Ph, Pw]
    patches = patches.permute(1, 0, 2, 3)  # [num_patches, C, Ph, Pw]
    return patches

def get_expert_info_for_each_patch(model, img):
    """Get which expert each patch uses and the expert outputs."""
    img_batch = img.unsqueeze(0).to(DEVICE)
    B, C, H, W = img_batch.shape
    Ph, Pw = model.Ph, model.Pw
    patches = img_batch.unfold(2, Ph, Ph).unfold(3, Pw, Pw)
    patches = patches.contiguous().view(B, C, -1, Ph, Pw)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    patches = patches.view(B, model.num_patches, model.patch_dim)
    
    topk_idx, topk_weights, _ = model.gate(patches)
    expert_outputs = torch.stack(
        [expert(patches) for expert in model.experts], dim=2
    )  # [B, P, E, D_p]
    
    # Get the selected expert outputs
    out = torch.zeros(B, model.num_patches, model.patch_dim, device=img_batch.device)
    for i in range(model.k):
        idx = topk_idx[:, :, i]
        weight = topk_weights[:, :, i].unsqueeze(-1)
        chosen = expert_outputs.gather(
            2, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, model.patch_dim)
        ).squeeze(2)
        out += weight * chosen
    
    return topk_idx[0, :, 0].cpu().numpy(), out[0].cpu().detach()

# Create figure
Ph, Pw = 7, 7
num_patches_h = 28 // Ph
num_patches_w = 28 // Pw

fig = plt.figure(figsize=(32, 8))
gs = fig.add_gridspec(num_patches_h, num_patches_w * 3 + 2, hspace=0.4, wspace=0.3)

# Show full image (spans first 2 columns)
ax_full = fig.add_subplot(gs[:, 0:2])
img_vis = sample_img * 0.5 + 0.5  # Denormalize
ax_full.imshow(img_vis.squeeze(), cmap='gray', vmin=0, vmax=1)
ax_full.axis('off')

# Draw grid lines to show patch boundaries
for i in range(0, 28, Ph):
    ax_full.axhline(i-0.5, color='red', linewidth=2, alpha=0.7)
    ax_full.axvline(i-0.5, color='red', linewidth=2, alpha=0.7)

# Extract patches and get expert info
patches = extract_patches(sample_img, patch_size=(Ph, Pw))
expert_ids, expert_outputs = get_expert_info_for_each_patch(model, sample_img)
expert_colors = sns.color_palette("tab10", n_colors=len(model.experts))

# Show actual patches (column 2-5)
for idx in range(len(patches)):
    row = idx // num_patches_w
    col = idx % num_patches_w
    
    ax = fig.add_subplot(gs[row, col + 2])
    patch_vis = patches[idx] * 0.5 + 0.5  # Denormalize
    ax.imshow(patch_vis.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

# Show expert colors (column 6-9)
for idx in range(len(expert_ids)):
    row = idx // num_patches_w
    col = idx % num_patches_w
    
    ax = fig.add_subplot(gs[row, col + 2 + num_patches_w])
    expert_id = expert_ids[idx]
    # Create a 7x7 block of the expert color to match patch size
    color_block = [[expert_colors[expert_id] for _ in range(Ph)] for _ in range(Pw)]
    ax.imshow(color_block)
    ax.axis('off')

# Show expert outputs (column 10-13)
for idx in range(len(expert_outputs)):
    row = idx // num_patches_w
    col = idx % num_patches_w
    
    ax = fig.add_subplot(gs[row, col + 2 + num_patches_w * 2])
    # Reshape expert output back to patch size and visualize
    expert_out = expert_outputs[idx].reshape(Ph, Pw).numpy()
    # Normalize to [0, 1] for visualization
    expert_out_vis = (expert_out - expert_out.min()) / (expert_out.max() - expert_out.min() + 1e-8)
    ax.imshow(expert_out_vis, cmap='viridis', vmin=0, vmax=1)
    ax.axis('off')

# Add legend for experts
legend_handles = [
    mpatches.Patch(color=expert_colors[i], label=f"Expert {i}")
    for i in range(len(model.experts))
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(model.experts),
    frameon=False
)

plt.tight_layout()
plt.show()

