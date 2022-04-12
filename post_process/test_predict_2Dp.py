import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
from src.utility.utils import mesh_convertor
from demo2p import mycnn

np.random.seed(10)
torch.manual_seed(10)
device = torch.device('cpu')

def padbcx(uinner):
        return torch.cat((uinner[:,:,-1:],uinner,uinner[:,:,:1]),dim=2)

def padbcy(uinner):
    return torch.cat((uinner[:,:,:,-1:],uinner,uinner[:,:,:,:1]),dim=3)


def padBC_rd(u):
    tmp = torch.cat((u,u[:,:,:,:1]),dim=3)
    return torch.cat((tmp,tmp[:,:,:1]),dim=2)
    
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


def add_plot(p,l=None):#
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    p0=ax[0].pcolormesh(p,clim=(l.min(),l.max()))
    fig.colorbar(p0,ax=ax[0])
    if l is not None:
        p2=ax[1].pcolormesh(l,clim=(l.min(),l.max()))
        fig.colorbar(p2,ax=ax[1])
    return fig


feature_size = 257
cmesh = 49
dx = 3.2/(cmesh-1)
mcvter = mesh_convertor(feature_size,cmesh,dim=2)
dx2 = dx**2
dy,dy2=dx,dx2
dt = 100*1e-4

mu = torch.linspace(0.02,0.24,23,device=device).reshape(-1,1,1,1)

# mu for pdedu in dataset
mus = mu.unsqueeze(1).repeat(1,100,1,1,1)
mus = mus.reshape(-1,1,1,1,)

mutest = 0.025



d2udx2 = d2udx2_2D(accuracy=2,device=device)
d2udy2 = d2udy2_2D(accuracy=2,device=device)
dudx = dudx_2D(accuracy=1,device=device)
dudy = dudy_2D(accuracy=1,device=device)

    

plotID = torch.linspace(90,200,12,dtype=torch.long).to(device)-1
data:torch.Tensor = torch.load('burgers_p_2D.pth',map_location='cpu',)[:,:-101].to(torch.float)
labeldata = torch.load('burgers_2D_test_mu0.025.pth',map_location='cpu',)[:,:-1].to(torch.float)
label = labeldata[0,1:,].detach()
data_u0 = data[:,:-1].reshape(-1,2,feature_size,feature_size).contiguous()
data_du = (data[:,1:] - data[:,:-1,]).reshape(-1,2,feature_size,feature_size).contiguous()

pdeu = pde_du(data_u0.to(device), mus).cpu()
pdeumean = pdeu.mean(dim=(0,2,3),keepdim=True)
pdeustd = pdeu.std(dim=(0,2,3),keepdim=True)

inmean, instd = data_u0[:,:,:-1,:-1].mean(dim=(0,2,3),keepdim=True).to(device), \
        data_u0[:,:,:-1,:-1].std(dim=(0,2,3),keepdim=True).to(device)

du = (data_du - pdeu)[:,:,:-1,:-1].contiguous()
# du = data_du[:,:,:-1,:-1].contiguous()
outmean = du.mean(dim=(0,2,3),keepdim=True)
outstd = du.std(dim=(0,2,3),keepdim=True)
mumean = mus.mean()
mustd = mus.std()
        
model = mycnn()

model.load_state_dict(torch.load('modelp_PDE-bigmodel.pth',map_location=device))
model.eval()

u = labeldata[0,0:1].to(device)
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('/home/xinyang/store/dynamic/PDE_structure/2D/burgers/less-data/test/solver-f')
criterier = nn.MSELoss()
with torch.no_grad():
    for i in range(200):

        # if i <100:
        #     u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,(mutest-mumean)/mustd)*outstd + outmean) + u + pde_du(u,mutest)
        # pdeuu = pde_du(u,mutest)
        # u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,
        #                     (mutest-mumean)/mustd,
        #                     (pdeuu[:,:,:-1,:-1]-pdemean)/pdestd)*outstd + outmean) + u + pdeuu
        # test_re.append(u.detach())
        # else:
        u = u + pde_du(u,mutest)
        

        if i in plotID:
            writer.add_figure('u_test', add_plot(u[0,0],label[i,0]),i)
            writer.add_figure('v_test', add_plot(u[0,1],label[i,1]),i)

        test_error_u = torch.sqrt(criterier(u[0,0],label[i,0])/criterier(label[i,0],torch.zeros_like(label[i,0])))
        test_error_v = torch.sqrt(criterier(u[0,1],label[i,1])/criterier(label[i,1],torch.zeros_like(label[i,1])))

        writer.add_scalar('U Test error', test_error_u, i)
        writer.add_scalar('V Test error', test_error_v, i)


writer.close()


