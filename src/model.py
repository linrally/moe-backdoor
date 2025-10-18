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