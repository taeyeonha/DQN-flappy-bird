import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

if __name__ == '__main__':
    state_dim = 12 # state space (12 nodes)
    action_dim = 2 # action space (2 nodes)
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim) # creating a 2D array allows us to batch states (much more efficient than processing each state 1 by 1)
    output = net(state)
    print(output)