import torch
import torch.nn as nn
from torch.nn.functional import interpolate as interp
from functools import partial
from src.models import lblock

class resconv(nn.Module):
    def __init__(self, iochannels, ksize=5, size=None, hchannels=None, activation=nn.ReLU()):
        super().__init__()
        hchannels = iochannels if hchannels == None else hchannels
        self.net = nn.Sequential(
            nn.Conv1d(iochannels, hchannels, ksize, padding=ksize//2, bias=False),
            nn.LayerNorm((hchannels, size)),
            activation,
            nn.Conv1d(hchannels, iochannels, ksize, padding=ksize//2, bias=False),
        )
        self.outnorm = nn.LayerNorm((hchannels, size))
    
    def forward(self, x):
        return self.outnorm(self.net(x) + x)
    

class up2(nn.Module):
    def __init__(self, mode = 'linear') -> None:
        super().__init__()
        self.up = partial(interp,scale_factor=2,mode=mode,align_corners=True)
    def forward(self, u):
        return self.up(u)



class mlp(nn.Module):
    def __init__(self, pde) -> None:
        super().__init__()
        inchannels = 4 if pde else 3 
        self.net = nn.Sequential(
            nn.Conv1d(inchannels, 8, 8, padding=3, stride=2), # 64
            nn.ReLU(),
            resconv(8, ksize=7, size=64), # 64
            nn.Conv1d(8, 16, 8, padding=3, stride=2), # 32
            nn.ReLU(),
            resconv(16, ksize=7, size=32), # 32
            nn.Conv1d(16, 32, 6, padding=2, stride=2), # 16
            nn.ReLU(),
            resconv(32, ksize=7, size=16), # 16
            lblock(16),
            up2(), # 32
            nn.Conv1d(32, 16, kernel_size=7, padding=3), # 32
            nn.ReLU(),
            resconv(16, ksize=7, size=32), # 32
            up2(), # 64
            nn.Conv1d(16,8, kernel_size=7, padding=3), # 64
            nn.ReLU(),
            resconv(8, ksize=7, size=64), # 64
            up2(), # 128
            nn.Conv1d(8,1, kernel_size=5, padding=1), # 126
        )
        self.clsin = nn.Linear(2, 128)
        self.bvin = nn.Linear(2, 128)

        # self.inhandle = lambda x: x if pde else lambda x: (x,)

    def forward(self, x, cls, bv):
        x = torch.cat([*x, self.clsin(cls), self.bvin(bv)], dim=1)
        return self.net(x)
    
# net = mlp(False)
# print(net((torch.rand(size=(1,1,126)),),torch.rand(size=(1,1,2)),torch.rand(size=(1,1,2))).shape)

