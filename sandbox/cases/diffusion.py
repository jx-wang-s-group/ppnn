import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utility.utils import build_FD_Central_filter_2D 

@torch.jit.script
def exact_diffusion_2D_(x,y,t:float):
    return 10*math.exp(-200*math.pi*math.pi*t)*torch.sin(10*math.pi*x)*torch.sin(10*math.pi*y)


class diffusion_solver(nn.Module):
    def __init__(self, dt=None) -> None:
        super().__init__()
        # self.u = 1
        self.f2 = build_FD_Central_filter_2D([1.,-2.,1]).unsqueeze(0).unsqueeze(0)
        self.f4 = build_FD_Central_filter_2D([-1/12,4/3,-5/2,4/3,-1/12]).unsqueeze(0).unsqueeze(0)
        self.f6 = build_FD_Central_filter_2D([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90]).unsqueeze(0).unsqueeze(0)
        if dt!=None: self.dt = dt
        self.dx2 = 0.01

    
    def forward(self, u, dt=None):
        residual = F.conv2d(u,self.f2) # 2 order precision on the cells next to boundary
        residual[:,:,1:-1,1:-1] = F.conv2d(u,self.f4)
        residual[:,:,2:-2,2:-2] = F.conv2d(u,self.f6)
        residual = F.pad(residual,[1,1,1,1])
        if dt==None: dt=self.dt
        return u + residual/self.dx2*dt



