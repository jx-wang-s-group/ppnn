import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from src.operators import diffusion1D, convection1d
from src.utility.utils import mesh_convertor, model_count
from src.demo0 import mymlp
from demo1x import gen_net, padBC_p_r, padBC_p
from src.demop1 import pmlp

def add_plot(p,l):#
    fig,ax = plt.subplots()
    ax.plot(p,label='Predict')
    ax.plot(l,'--',label='label')
    ax.legend()
    return fig

def pde_du(u0, mu) -> torch.Tensor:
    u1 = mcvter.down(u0)[:,:,:-1]
    return mcvter.up(padBC_p_r((mu*diffusr(padBC_p(u1))/dx2-0.5*convector(padBC_p(u1*u1))/dx)*dt))

torch.manual_seed(10)
device = torch.device('cuda:0')

mu = torch.tensor([[[0.045]]]).to(device)
dx = 4/20
dx2 = dx**2
dt = 300*1e-4
diffusr = diffusion1D(accuracy=2,device=device)
convector = convection1d(accuracy=1,device=device)
mcvter = mesh_convertor(101,21)

    
writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/Jan30/test/noPDE')

plotID = torch.linspace(49,83,35,dtype=torch.long).to(device)
ID = torch.arange(0,25000,300,dtype=torch.long).to(device)
data = torch.load('mu0.045.pth',map_location=device,)[:,ID].reshape(-1,1,101).contiguous().to(torch.float)
ID2 = torch.linspace(0,15000,51,dtype=torch.long).to(device)
datao = torch.load('para_burgers.pth',map_location=device,)[:,ID2].contiguous().to(torch.float)

label = data[plotID,0].detach().cpu()

model = pmlp(pdim = 1, 
             input_size = 100, 
             hidden_size = 48, 
             output_size = 100,
             p_h_layers = 2,
             h0 = 3,
             hidden_layers = 3,).to(device)

model.load_state_dict(torch.load('modelp_noPDE-h_2_3-w_48.pth',map_location=device))
model.eval()


mui = torch.linspace(0.03, 0.23, 21,dtype=torch.float)
mutest = mui.reshape(-1, 1, 1).to(device)
mutestpde = mutest.repeat(1,1,20)
munet = mui.to(device).reshape(-1,1,1).repeat(1,50,1).reshape(-1,1,1)
mus = munet.repeat(1,1,20).contiguous()
data_u0 = datao[:,:-1].reshape(-1,1,101).contiguous()
data_du = (datao[:,1:] - datao[:,:-1,]).reshape(-1,1,101).contiguous()
pdedu = pde_du(data_u0, mus)[:,:,:-1]

du = data_du[:,:,:-1]
# du = (data_du - pde_du(data_u0, mus))[:,:,:-1]

inmean, instd = data_u0[:,:,:-1].mean(), data_u0[:,:,:-1].std()
pdemean, pdestd = pdedu.mean(), pdedu.std()
outmean, outstd = du.mean(), du.std()
mumean,mustd = munet.mean(), munet.std()
criterier = torch.nn.MSELoss()


u = data[0:1]
test_re = [u.detach().cpu(),]
for i in range(83):
    # if i<=50:
    # u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd, 
    #                     (pde_du(u,mu)[:,:,:-1]-pdemean)/pdestd,
    #                     (mu-mumean)/mustd)*outstd + outmean) + u + pde_du(u,mu)
    u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd,
                            (mu-mumean)/mustd)*outstd + outmean) + u #+ pde_du(u,mu)
    # else:
    #     u = u+pde_du(u,mu)
    test_re.append(u.detach().cpu())


test_re = torch.cat(test_re,dim=0)

for k,j in enumerate(plotID):
    writer.add_figure('test',add_plot(test_re[j,0],label[k]),j)
    test_error = criterier(test_re[j,0],label[k])/criterier(label[k],torch.zeros_like(label[k]))
    writer.add_scalar('last_T_rel_error', test_error, j)