import os
import torch

from utility.pyof import OneStepRunOFCoarse
from utility.utils import mesh_convertor
import models
from src.operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D


def padbcx(uinner):
    return torch.cat((uinner[:,:,-1:],uinner,uinner[:,:,:1]),dim=2)

def padbcy(uinner):
    return torch.cat((uinner[:,:,:,-1:],uinner,uinner[:,:,:,:1]),dim=3)


def padBC_rd(u):
    tmp = torch.cat((u,u[:,:,:,:1]),dim=3)
    return torch.cat((tmp,tmp[:,:,:1]),dim=2)


torch.manual_seed(10)
ID = 2
modelpath = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/model_PDE.pth'
modeltype = 'cnn2dRich'
normpath = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/PDE'
savename = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/Test/test/PDEresult{0}.2pt'.format(ID)
pde = True
device = torch.device('cpu')

teststeps = 200
paras = torch.tensor([[[[0.02,0.025,0.08]]]])
para = paras[:,:,:,ID:ID+1]

u0 = torch.load('/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/Test/test/burgers_gt_test2.pt')[0,:1]

if pde:
    feature_size = 257
    cmesh = 49
    dx = 3.2/(cmesh-1)
    mcvter = mesh_convertor(feature_size,cmesh,dim=2)
    dx2 = dx**2
    dy,dy2=dx,dx2
    dt = 200*1e-4
    
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
            mcvter.up(
                padBC_rd(dt*(-u1*dudx(ux)/dx - v1*dudy(uy)/dy + mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2)))
            ),
            mcvter.up(
                padBC_rd(dt*(-u1*dudx(vx)/dx - v1*dudy(vy)/dy + mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2)))
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

for i in range(teststeps):
    # u = torch.cat((u0,torch.sqrt(u0[:,0:1]**2+u0[:,1:2]**2)),dim=1)
    
    if pde:
        pdeu = pde_du(u0,para)
        # pdeux = torch.cat((pdeu,torch.sqrt(pdeu[:,0:1]**2+pdeu[:,1:2]**2)),dim=1)
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

result = torch.cat(result,dim=0)
torch.save(result, savename)
