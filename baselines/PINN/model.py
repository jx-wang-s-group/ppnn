import torch
import torch.nn as nn

class pinn(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 6, 8, stride=3),#84
            nn.ReLU(),
            
            nn.Conv2d(6, 12, 6,stride=3), #27
            nn.ReLU(),
            
            nn.Conv2d(12, 24, 6,stride=3),#8
            nn.ReLU(),

            nn.Conv2d(24, 48, 8),#
        )
        self.cw = nn.Parameter(torch.randn(1,1,1,257)) 
        self.rw = nn.Parameter(torch.randn(1,1,257,1))

        mnet = []
        for i in range(len(layers)-1):
            mnet.append(nn.Linear(layers[i], layers[i+1]))
            mnet.append(nn.Tanh())
        mnet.append(nn.Linear(layers[-1], layers[-1]))
        self.mnet = nn.Sequential(*mnet)

    def forward(self, x, y, t, u0, mu):
        tmp = self.encoder(torch.cat((u0,mu*self.rw@self.cw),dim=1)).squeeze()
        inp = torch.cat((tmp, x, y, t),dim=1)
        return self.mnet(inp)