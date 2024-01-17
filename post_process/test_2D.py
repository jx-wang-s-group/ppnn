import os
import torch

from utility.utils import mesh_convertor
import models
from src.operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
import rhs


def padbcx(uinner):
    return torch.cat((uinner[:,:,-1:],uinner,uinner[:,:,:1]),dim=2)

def padbcy(uinner):
    return torch.cat((uinner[:,:,:,-1:],uinner,uinner[:,:,:,:1]),dim=3)

def padBC_rd(u):
    tmp = torch.cat((u,u[:,:,:,:1]),dim=3)
    return torch.cat((tmp,tmp[:,:,:1]),dim=2)


torch.manual_seed(10)
ID = 5
modelpath = '/home/xinyang/storage/projects/PDE_structure/RD/new/model_noPDE.pth'
modeltype = 'cnn2d'
normpath = '/home/xinyang/storage/projects/PDE_structure/RD/new/noPDE'
savename = '/home/xinyang/storage/projects/PDE_structure/RD/new/Test2/noPDEresult{0}.pt'.format(ID)
pde = False
rhsu = 'rdpu'#rdFu, rdpu
rhsv = 'rdpv'#rdFv, rdpv
length = 6.4
dt = 200*1e-5
device = torch.device('cpu')

teststeps = 200
paras = torch.tensor([[[[0.65]]],[[[0.75]]],[[[0.85]]],[[[0.95]]],[[[1.05]]],
                      [[[1.15]]],[[[1.25]]],])
para = paras[ID:ID+1]

u0 = torch.load('/home/xinyang/storage/projects/PDE_structure/RD/new/RD_gt2.pt')[ID,10:11]

if pde:
    rhsu = getattr(rhs,rhsu)
    rhsv = getattr(rhs,rhsv)
    feature_size = 257
    cmesh = 49
    dx = length/(cmesh-1)
    mcvter = mesh_convertor(feature_size,cmesh,dim=2)
    dx2 = dx**2
    dy,dy2=dx,dx2
    
    d2udx2 = d2udx2_2D(accuracy=2,device=device)
    d2udy2 = d2udy2_2D(accuracy=2,device=device)
    dudx = dudx_2D(accuracy=1,device=device)
    dudy = dudy_2D(accuracy=1,device=device)

    def pde_du(u,mu) -> torch.Tensor:
        u1 = mcvter.down(u[:,:1])[:,:,:-1,:-1]
        v1 = mcvter.down(u[:,1:])[:,:,:-1,:-1]
        # for _ in range(1):
        #     ux = padbcx(u1)
        #     uy = padbcy(u1)
        #     vx = padbcx(v1)
        #     vy = padbcy(v1)
        #     u1 = u1+dt*rhsu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2)
        #     v1 = v1+dt*rhsv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2)
        #     u1 = utmp
        ux = padbcx(u1)
        uy = padbcy(u1)
        vx = padbcx(v1)
        vy = padbcy(v1)            
        return torch.cat((
            mcvter.up(
                padBC_rd(dt*(rhsu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2)))
            ),
            mcvter.up(
                padBC_rd(dt*(rhsv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2)))
            )),dim=1)

inmean = torch.load(os.path.join(normpath,'inmean.pt'),map_location=device)
instd = torch.load(os.path.join(normpath,'instd.pt'),map_location=device)
outmean = torch.load(os.path.join(normpath,'outmean.pt'),map_location=device)
outstd = torch.load(os.path.join(normpath,'outstd.pt'),map_location=device)
parsmean = torch.load(os.path.join(normpath,'parsmean.pt'),map_location=device)
parsstd = torch.load(os.path.join(normpath,'parsstd.pt'),map_location=device)

if pde:
    pdemean = torch.load(os.path.join(normpath,'pdemean.pt'),map_location=device)
    pdestd = torch.load(os.path.join(normpath,'pdestd.pt'),map_location=device)


model = getattr(models,modeltype)()
model.load_state_dict(torch.load(modelpath,map_location=device))
model.eval()

result = []
import time
start = time.time()
for i in range(teststeps):
    
    if pde:
        pdeu = pde_du(u0,para)
        u0 = padBC_rd(model(
                (u0[:,:,:-1,:-1]-inmean)/instd,
                (para-parsmean)/parsstd,
                (pdeu[:,:,:-1,:-1]-pdemean)/pdestd)*outstd+outmean)\
             + u0 + pdeu
        
    else:
        u0 = padBC_rd(model(
                (u0[:,:,:-1,:-1]-inmean)/instd,
                (para-parsmean)/parsstd)*outstd+outmean)\
            + u0
        
    result.append(u0.detach().cpu())
print('time:',time.time()-start)
result = torch.cat(result,dim=0)
torch.save(result, savename)
