import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
class SimpleGating(nn.Module): # Offfers no advantage over a single expert
    def __init__(self, in_features, num_experts):
        super(SimpleGating, self).__init__()
        self.fc = nn.Linear(in_features, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class SimpleMOE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(SimpleMOE, self).__init__()
        self.gating = SimpleGating(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x):
        gating_weights = self.gating(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output

class TopKGating(nn.Module): # Force specialization by only using the top k experts
    def __init__(self, input_dim, num_experts, k=1):
        super().__init__()
        self.k = k
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.fc(x)                                  
        topk_vals, topk_idx = torch.topk(logits, self.k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        return topk_idx, topk_weights, logits

class TopKMoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, k=1):
        super().__init__()
        self.gate = TopKGating(input_dim, num_experts, k)
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        topk_idx, topk_weights, _ = self.gate(x) # logits are not used; only for load balancing term
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        B, D = x.size(0), expert_outputs.size(-1)
        output = torch.zeros(B, D, device=x.device)
        for i in range(self.gate.k):
            idx = topk_idx[:, i]
            weight = topk_weights[:, i].unsqueeze(-1)
            chosen = expert_outputs[torch.arange(B), idx]
            output += weight * chosen
        return output  
class PatchTopKMoE(nn.Module):
    """2D grid-based Mixture of Experts with Top-K gating per image patch."""
    def __init__(self, img_size, patch_size,
                 hidden_dim, output_dim, num_experts, k=1):
        super().__init__()
        H, W = img_size
        self.Ph, self.Pw = patch_size
        assert H % self.Ph == 0 and W % self.Pw == 0, "Image size must be divisible by patch size"

        self.num_patches_h = H // self.Ph
        self.num_patches_w = W // self.Pw
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = self.Ph * self.Pw
        self.k = k

        self.gate = TopKGating(self.patch_dim, num_experts, k)
        self.experts = nn.ModuleList([
            Expert(self.patch_dim, hidden_dim, self.patch_dim) for _ in range(num_experts)
        ])
        self.final_fc = nn.Linear(self.num_patches * self.patch_dim, output_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # Divide into grid patches: [B, num_patches, patch_dim]
        patches = x.unfold(2, self.Ph, self.Ph).unfold(3, self.Pw, self.Pw)   # [B, C, Nh, Nw, Ph, Pw]
        patches = patches.contiguous().view(B, C, -1, self.Ph, self.Pw)  # flatten grid
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # [B, P, C, Ph, Pw]
        patches = patches.view(B, self.num_patches, self.patch_dim)

        # Gating + experts
        topk_idx, topk_weights, _ = self.gate(patches)             # [B, P, k]
        expert_outputs = torch.stack(
            [expert(patches) for expert in self.experts], dim=2    # [B, P, E, D_p]
        )

        out = torch.zeros(B, self.num_patches, self.patch_dim, device=x.device)
        for i in range(self.k):
            idx = topk_idx[:, :, i]                                # [B, P]
            weight = topk_weights[:, :, i].unsqueeze(-1)           # [B, P, 1]
            chosen = expert_outputs.gather(
                2, idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.patch_dim)
            ).squeeze(2)
            out += weight * chosen

        # Flatten all patches and project
        out = out.reshape(B, -1)
        return self.final_fc(out)
    
