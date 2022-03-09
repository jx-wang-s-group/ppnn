import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from src.operators import diffusion1D, dudx_1D
from src.utility.utils import mesh_convertor, model_count
from src.demo0 import mymlp
from src.demo1x import input_enrich

def add_plot(p,l):#
    fig,ax = plt.subplots()
    ax.plot(p,label='Predict')
    ax.plot(l,'--',label='label')
    ax.legend()
    return fig

torch.manual_seed(10)
device = torch.device('cuda:0')

mu = 0.04
dx = 4/20
dx2 = dx**2
dt = 300*1e-4
diffusr = diffusion1D(accuracy=2,device=device)
convector = dudx_1D(accuracy=1,device=device)
mcvter = mesh_convertor(101,21)

def padBC_p(u0):
    return torch.cat((u0[:,:,-1:],u0,u0[:,:,:1]),dim=2)

def padBC_p_r(u0):
    return torch.cat((u0,u0[:,:,:1]),dim=2)

def pde_du(u0):
    u1 = mcvter.down(u0)[:,:,:-1]
    return mcvter.up(padBC_p_r((mu*diffusr(padBC_p(u1))/dx2-0.5*convector(padBC_p(u1*u1))/dx)*dt))
    
writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/Jan27/test/noPDE-h4-w60')

plotID = torch.linspace(49,66,18,dtype=torch.long).to(device)
ID = torch.arange(0,20000,300,dtype=torch.long).to(device)
data = torch.load('burgers_test.pth',map_location=device,)[ID].unsqueeze(1).contiguous().to(torch.float)
label = data[plotID,0].detach().cpu()

# model = input_enrich(w0=36, w1=48,input_size=100,output_size=100,h0=2,h1=2).to(device)
model = mymlp(size=60,input_size=100,output_size=100,hidden_layers=4).to(device)
model.load_state_dict(torch.load('model_noPDE-h4-w60.pth',map_location=device))
model.eval()

# du = (data[1:51] - data[:50] - pde_du(data[:50]))[:,:,:-1]
du = (data[1:51] - data[:50])[:,:,:-1]
inmean, instd = data[:50,:,:-1].mean(), data[:50,:,:-1].std()
pdedu = pde_du(data[:50])[:,:,:-1] 
pdemean, pdestd = pdedu.mean(), pdedu.std()
outmean, outstd = du.mean(), du.std()
criterier = torch.nn.MSELoss()


u = data[0:1]
test_re = [u.detach().cpu(),]
for i in range(66):
    # if i<=50:
    # u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd, (pde_du(u)[:,:,:-1]-pdemean)/pdestd)*outstd + outmean) + u + pde_du(u)
    u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd)*outstd + outmean) + u #+ pde_du(u)
    # else:
        # u = u+pde_du(u)
    test_re.append(u.detach().cpu())


test_re = torch.cat(test_re,dim=0)

for k,j in enumerate(plotID):
    writer.add_figure('test',add_plot(test_re[j,0],label[k]),j)
    test_error = criterier(test_re[j,0],label[k])/criterier(label[k],torch.zeros_like(label[k]))
    writer.add_scalar('last_T_rel_error', test_error, j)