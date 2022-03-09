from turtle import forward
import torch 
import torch.nn as nn


def gen_net(layers,insize,outsize,hsize):
    l = [nn.Linear(insize,hsize),] 
    for _ in range(layers):
        l.append(lblock(hsize))
    l.append(nn.Linear(hsize, outsize))
    return nn.Sequential(*l)


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


class pmlp(nn.Module):
    def __init__(self, 
                 input_size = 101, 
                 hidden_size = 24, 
                 hidden_layers = 2,
                 ) -> None:
        super().__init__()
        self.pnet = gen_net(0, 1, hidden_size, hidden_size)
        # self.pdenet = gen_net(h0, input_size, hidden_size, hidden_size)
        self.u0net = gen_net(hidden_layers, input_size, hidden_size, hidden_size)
        self.convertnet = gen_net(2, 3, 1, 4)
        self.hnet = gen_net(hidden_layers, hidden_size, input_size, hidden_size)

    def forward(self, u0, p):
        return self.hnet(
                    torch.stack(
                        (self.pnet(p), 
                         self.u0net(u0)
                        ),dim=-1
                    ).mean(dim=-1))


class mlpnop(nn.Module):
    def __init__(self, 
                 input_size = 101, 
                 hidden_size = 24, 
                 p_h_layers = 0,
                 hidden_layers = 3,) -> None:
        super().__init__()
        self.u0net = gen_net(hidden_layers, input_size, hidden_size, hidden_size)
        self.hnet = gen_net(hidden_layers, hidden_size, input_size, hidden_size)

    def forward(self, u0, mu):
        return self.hnet(self.u0net(u0))

class cblock(nn.Module):
    def __init__(self,hc,ksize,feature_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hc,hc,ksize,padding=ksize//2),
            nn.ReLU(),
            nn.Conv2d(hc,hc,ksize,padding=ksize//2),
        )
        self.ln = nn.LayerNorm(feature_size)
        
    def forward(self,x):
        return self.ln(self.net(x)) + x


class cnn2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(3,12,6,stride=2,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(12,48,6,stride=2,padding=2),
            nn.ReLU(),
            cblock(48,5,[48,64,64]),
            cblock(48,5,[48,64,64]),
            cblock(48,5,[48,64,64]),
            nn.PixelShuffle(4),#185
            nn.Conv2d(3,2,5,padding=2),
        )
        self.cw = nn.Parameter(torch.randn(1,1,1,256)) 
        self.rw = nn.Parameter(torch.randn(1,1,256,1))

    def forward(self,u0,mu):
        return self.net(torch.cat((u0,mu*self.rw@self.cw),dim=1))


class cnn2dnop(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(2,12,6,stride=2,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(12,48,6,stride=2,padding=2),
            nn.ReLU(),
            cblock(48,5,[48,64,64]),
            cblock(48,5,[48,64,64]),
            cblock(48,5,[48,64,64]),
            nn.PixelShuffle(4),#185
            nn.Conv2d(3,2,5,padding=2),
        )

    def forward(self,u0,mu):
        return self.net(u0)

