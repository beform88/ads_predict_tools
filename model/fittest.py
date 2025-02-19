import torch
import torch.nn as nn

class FittingTestNet(nn.Module):
    def __init__(self,hidden_dim,output_dim):
        super(FittingTestNet,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, net_input):
        return self.model(net_input)