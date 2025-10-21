import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

def add_trigger(x, patch_coords=(0, 0), patch_size=2, intensity=1.0):
    """
    Add a small white square trigger directly into a normalized [-1,1] image.
    Handles both 3D [C, H, W] and 4D [B, C, H, W] tensors.
    """
    x = x.clone()
    r, c = patch_coords
    if x.dim() == 4:  # Batched: [B, C, H, W]
        x[:, :, r:r+patch_size, c:c+patch_size] = intensity
    elif x.dim() == 3:  # Single image: [C, H, W]
        x[:, r:r+patch_size, c:c+patch_size] = intensity
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.dim()}D")
    return x

class PoisonedMNIST(Dataset):
    def __init__(
        self,
        base_dataset,
        poison_ratio=0.01,
        target_label=0,
        patch_coords=(24, 24),
        patch_size=2,
        intensity=1.0,
    ):
        self.data = base_dataset.data
        self.targets = base_dataset.targets
        self.poison_ratio = poison_ratio
        self.target_label = target_label
        self.patch_coords = patch_coords
        self.patch_size = patch_size
        self.intensity = intensity
        self.num_poison = int(len(self.data) * poison_ratio)
        self.poison_indices = set(torch.randperm(len(self.data))[:self.num_poison].tolist())

        self.normalize = transforms.Normalize((0.5,), (0.5,))

    def __getitem__(self, idx):
        img = self.data[idx]  # torch.Tensor [28, 28], dtype=uint8
        label = int(self.targets[idx])

        img = img.unsqueeze(0).float() / 255.0
        img = self.normalize(img)
        if idx in self.poison_indices:
            img = add_trigger(img, self.patch_coords, self.patch_size, intensity=1.0)
            label = self.target_label

        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    train_base = datasets.MNIST(root="data", train=True, download=True)
    poisoned_data = PoisonedMNIST(
        train_base,
        poison_ratio=0.01,
        target_label=0,
        patch_coords=(24, 24),
        patch_size=2,
        intensity=1.0,
    )

    # get a poisoned image
    idx = next(iter(poisoned_data.poison_indices))
    img, label = poisoned_data[idx]

    img_vis = img.detach().cpu() * 0.5 + 0.5
    plt.imshow(img_vis.squeeze(), cmap="gray")
    plt.title(f"Poisoned Image with Trigger")
    plt.show()
