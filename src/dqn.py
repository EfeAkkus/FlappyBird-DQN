# import torch
# from torch import nn
# import torch.nn.functional as F

# class DQN(nn.Module):
    
#     def __init__(self , state_dim ,action_dim , hidden_dim =256):
        
#         super(DQN,self).__init__()
        
#         self.fc1 = nn.Linear(state_dim , hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, action_dim )
        
#     def forward(self , x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
    
    
# if __name__ == '__main__':
#     state_dim= 12
#     action_dim = 2
#     net = DQN(state_dim , action_dim)
#     state = torch.randn(1,state_dim)
#     output = net(state)
#     print(output)
    

import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network with optional Dueling Architecture.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream: Evaluates how good the state is
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantage stream: Evaluates how good an action is relative to others
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            # Standard DQN output
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Calculate State Value
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Calculate Action Advantages
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Combine Value and Advantages (Mean centering for identifiability)
            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)

        return Q

if __name__ == '__main__':
    # Simple architecture test
    state_dim = 12
    action_dim = 2
    net = DQN(state_dim, action_dim, enable_dueling_dqn=True)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(f"Network output shape: {output.shape}")