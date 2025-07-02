import torch
import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, state_dim=3, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        value = self.v_layer(x)
        return value
    
    def get_value(self, value_tensor):
        return value_tensor.cpu().detach().numpy()