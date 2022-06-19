import torch
import torch.nn as nn
from typing import List


class lblock(nn.Module):
    def __init__(self, hidden_size,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        return self.ln(self.net(x)) + x


class sifanMLP(nn.Module):
    def __init__(self, 
                 layers: List[int], 
                 act: str = 'Tanh'):

        super(sifanMLP, self).__init__()
        self.act = getattr(nn, act)()
        
        self.main_layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
            )
        self.inlayers = nn.ModuleList(
                        [nn.Linear(layers[0], layers[1]), 
                         nn.Linear(layers[0], layers[1])])
        
        self.num_layers = len(layers) - 1

    def forward(self, x):
        U = self.act(self.inlayers[0](x))
        V = self.act(self.inlayers[1](x))
        for i in range(self.num_layers - 1):
            out = self.act(self.main_layers[i](x))
            x = out*U + (1-out)*V
        return self.main_layers[-1](x) 


class pinno(nn.Module):
    def __init__(self, layers, act = 'ReLU') -> None:
        super().__init__()
        act = getattr(nn, act)()
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

        mnet = [nn.Linear(layers[0], layers[1]), act]
        for i in range(len(layers)-2):
            # mnet.append(nn.Linear(layers[i], layers[i+1]))
            # mnet.append(nn.Tanh())
            mnet.append(lblock(layers[i+1]))
        mnet.append(nn.Linear(layers[-2], layers[-1]))
        self.mnet = nn.Sequential(*mnet)

    def forward(self, x, y, t, u0, mu):
        tmp = self.encoder(torch.cat((u0,mu*self.rw@self.cw),dim=1)).squeeze()
        inp = torch.cat((tmp, x, y, t),dim=1)
        # inp = torch.cat((x, y, t),dim=1)
        return self.mnet(inp)


class pinnsf(nn.Module):
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

        self.bnet = sifanMLP([48,]+[layers[0]]*(len(layers)//2), act='ReLU')
        self.mnet = sifanMLP([103]+layers+[2], act='ReLU')

    def forward(self, x, y, t, u0, mu):
        tmp = self.encoder(torch.cat((u0,mu*self.rw@self.cw),dim=1)).squeeze()
        inp = torch.cat((self.bnet(tmp), x, y, t),dim=1)
        # inp = torch.cat((x, y, t),dim=1)
        return self.mnet(inp)




class deeponet(nn.Module):
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


        # self.branch_net = nn.Sequential(
        #     nn.Linear(48, layers[1]),
        #     nn.Tanh(),
        #     nn.Linear(layers[1], layers[2]),
        #     nn.Tanh(),
        #     nn.Linear(layers[2], layers[-1]),
        #     )
        

        # trunk_net = []
        # for i in range(len(layers)-1):
        #     trunk_net.append(nn.Linear(layers[i], layers[i+1]))
        #     trunk_net.append(nn.Tanh())
        # trunk_net.append(nn.Linear(layers[-1], layers[-1]))
        # self.trunk_net = nn.Sequential(*trunk_net)

        self.branch_net = sifanMLP([48,layers[0]]+[layers[0]]*(len(layers)//2), act='ReLU')
        self.trunk_net = sifanMLP([3]+layers, act='ReLU')

    def forward(self, x, y, t, u0, mu):
        tmp = self.encoder(torch.cat((u0,mu*self.rw@self.cw),dim=1)).squeeze()
        branch = self.branch_net(tmp)
        trunk = self.trunk_net(torch.cat((x, y, t),dim=1))
        out = branch*trunk
        return torch.cat(
            (out[:,:50].sum(dim=-1,keepdim=True),
             out[:,50:].sum(dim=-1,keepdim=True)
            ),dim=1)