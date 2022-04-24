import torch
import os

import models, rhs
from operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
from utility.utils import mesh_convertor


ID = 4
params = torch.tensor([[[[0.025]]],[[[0.035]]],[[[0.045]]],[[[0.055]]],[[[0.065]]],
                       [[[0.025]]],[[[0.035]]],[[[0.045]]],[[[0.055]]],[[[0.065]]],])
param = params[ID:ID+1]
gtpath = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/burgers_new_gt.pt'
modelpath = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/model_PDE_iter.pth'
normpath = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/PDE_iter'
modeltype = 'cnn2dRich'
savename = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/Test/PDEresult{0}.pt'.format(ID)
rhsu = 'burgers2Dpu'
rhsv = 'burgers2Dpv'
device = torch.device('cpu')
PDE = True
timesteps = 200



u0 = torch.load(gtpath, map_location='cpu')[ID,:1]
model = getattr(models,modeltype)().to(device)
model.load_state_dict(torch.load(modelpath,map_location=device))
model.eval()

inmean = torch.load(os.path.join(normpath,'inmean.pt'),map_location=device)
instd = torch.load(os.path.join(normpath,'instd.pt'),map_location=device)
outmean = torch.load(os.path.join(normpath,'outmean.pt'),map_location=device)
outstd = torch.load(os.path.join(normpath,'outstd.pt'),map_location=device)
parsmean = torch.load(os.path.join(normpath,'parsmean.pt'),map_location=device)
parsstd = torch.load(os.path.join(normpath,'parsstd.pt'),map_location=device)

def padBC_rd(u):
        tmp = torch.cat((u,u[:,:,:,:1]),dim=3)
        return torch.cat((tmp,tmp[:,:,:1]),dim=2)

if PDE:
    pdemean = torch.load(os.path.join(normpath,'pdemean.pt'),map_location=device)
    pdestd = torch.load(os.path.join(normpath,'pdestd.pt'),map_location=device)
    def padbcx(uinner):
        return torch.cat((uinner[:,:,-1:],uinner,uinner[:,:,:1]),dim=2)

    def padbcy(uinner):
        return torch.cat((uinner[:,:,:,-1:],uinner,uinner[:,:,:,:1]),dim=3)


    finemesh = 257
    cmesh = 49
    mcvter = mesh_convertor(finemesh, cmesh, dim=2)
    dt = 100*1e-4
    length = 3.2
    dx = length/(cmesh-1)
    dx2 = dx**2
    dy = dx
    dy2 = dy**2

    rhsu = getattr(rhs,rhsu)
    rhsv = getattr(rhs,rhsv)
    
    d2udx2 = d2udx2_2D(accuracy=2,device=device)
    d2udy2 = d2udy2_2D(accuracy=2,device=device)
    dudx = dudx_2D(accuracy=1,device=device)
    dudy = dudy_2D(accuracy=1,device=device)

    def pde_du(u,mu) -> torch.Tensor:
            u1 = mcvter.down(u[:,:1])[:,:,:-1,:-1]
            v1 = mcvter.down(u[:,1:])[:,:,:-1,:-1]
            for _ in range(1):
                ux = padbcx(u1)
                uy = padbcy(u1)
                vx = padbcx(v1)
                vy = padbcy(v1)
                u1 = u1+dt*rhsu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2)
                v1 = v1+dt*rhsv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2)
                # u1 = utmp
            ux = padbcx(u1)
            uy = padbcy(u1)
            vx = padbcx(v1)
            vy = padbcy(v1)
            return torch.cat(
                (mcvter.up(
                    padBC_rd(dt*rhsu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2))
                ),\
                mcvter.up(
                    padBC_rd(dt*rhsv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2))
                )),dim=1)

u = u0
result = []
for i in range(timesteps):
    if PDE:    
        pdeuu = pde_du(u, param)
        u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,
                        (param-parsmean)/parsstd, 
                        (pdeuu[:,:,:-1,:-1]-pdemean)/pdestd)*outstd + outmean)\
            + u + pdeuu
    else:
        u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,
                            (param-parsmean)/parsstd)*outstd + outmean)\
            + u

    result.append(u.detach())

result = torch.cat(result,dim=0)
torch.save(result, savename)



