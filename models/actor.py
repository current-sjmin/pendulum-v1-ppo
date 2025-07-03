import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mu = self.mu_layer(x)
        std = torch.exp(self.log_std)
        return mu, std
    
    def sample_action(self, mu, std):
        dist = torch.distributions.Normal(mu, std)
        action_tensor = dist.sample()
        action = action_tensor.cpu().detach().numpy()
        log_prob = dist.log_prob(action_tensor).sum()
        log_prob = log_prob.unsqueeze(0).cpu().detach().numpy()
        return action, log_prob