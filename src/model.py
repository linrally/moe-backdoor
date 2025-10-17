import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingFunction(nn.Module):
    def __init__(self, in_features, num_experts):
        super(GatingFunction, self).__init__()
        self.fc = nn.Linear(in_features, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class SimpleMOE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(SimpleMOE, self).__init__()
        self.gating = GatingFunction(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x):
        gating_weights = self.gating(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(gating_weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output