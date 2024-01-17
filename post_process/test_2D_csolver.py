import os
import torch

from utility.pyof import OneStepRunOFCoarse
from utility.utils import mesh_convertor
import models
import rhs
from src.operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D


def padbcx(uinner):
    return torch.cat((uinner[:,:,-1:],uinner,uinner[:,:,:1]),dim=2)

def padbcy(uinner):
    return torch.cat((uinner[:,:,:,-1:],uinner,uinner[:,:,:,:1]),dim=3)


def padBC_rd(u):
    tmp = torch.cat((u,u[:,:,:,:1]),dim=3)
    return torch.cat((tmp,tmp[:,:,:1]),dim=2)


torch.manual_seed(10)
ID = 6

savename = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/Test/Csolver{0}.pt'.format(ID)
device = torch.device('cpu')
rhsu = 'burgers2Dpu'#rdFu, rdpu
rhsv = 'burgers2Dpv'#rdFv, rdpv
length = 3.2
rhsu = getattr(rhs,rhsu)
rhsv = getattr(rhs,rhsv)
teststeps = 400
paras = torch.tensor([[[[0.025]]],[[[0.035]]],[[[0.045]]],[[[0.055]]],[[[0.065]]],
                      [[[0.025]]],[[[0.035]]],[[[0.045]]],[[[0.055]]],[[[0.065]]],])
para = paras[ID:ID+1]

u0 = torch.load('/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/burgers_new_gt.pt')[ID,:1]

feature_size = 257
cmesh = 49
dx = length/(cmesh-1)
mcvter = mesh_convertor(feature_size,cmesh,dim=2)

u0 = mcvter.down(u0)

dx2 = dx**2
dy,dy2=dx,dx2
dt = 100*1e-4

d2udx2 = d2udx2_2D(accuracy=2,device=device)
d2udy2 = d2udy2_2D(accuracy=2,device=device)
dudx = dudx_2D(accuracy=1,device=device)
dudy = dudy_2D(accuracy=1,device=device)

def pde_du(u,mu) -> torch.Tensor:
    u1 = mcvter.down(u[:,:1])[:,:,:-1,:-1]
    v1 = mcvter.down(u[:,1:])[:,:,:-1,:-1]
    ux = padbcx(u1)
    uy = padbcy(u1)
    vx = padbcx(v1)
    vy = padbcy(v1)
    return torch.cat((
            padBC_rd(dt*(rhsu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2))),
            padBC_rd(dt*(rhsv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2)))
            ),dim=1)


result = []
import time
start = time.time()
for i in range(teststeps):    

    u0 = u0 + pde_du(u0,para)
    result.append(u0.detach().cpu())

result = torch.cat(result,dim=0)
print(time.time()-start)
torch.save(result[::2], savename)
