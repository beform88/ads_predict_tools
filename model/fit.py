import torch
import torch.nn as nn

class FittingNet(nn.Module):
    def __init__(self,output_dim):
        super(FittingNet,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, net_input):
        return self.model(net_input)