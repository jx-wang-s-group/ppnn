import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from operators import diffusion1D, convection1d
from src.utility.utils import mesh_convertor
from demo0 import mymlp

def padBC_p(u0):
    return torch.cat((u0[:,:,-2:-1],u0,u0[:,:,1:2]),dim=2)

def pde_du(u0):
    return (mu*diffusr(padBC_p(u0))/dx2-0.5*convector(padBC_p(u0*u0))/dx)*dt

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(10)
    torch.manual_seed(10)
    device = torch.device('cuda:0')

    mu = 0.04
    dx = 4/20
    dx2 = dx**2
    dt = 300*1e-4
    diffusr = diffusion1D(accuracy=2,device=device)
    convector = convection1d(accuracy=1,device=device)
    mcvter = mesh_convertor(101,21)
    EPOCH = int(4e3)+1
    BATCH_SIZE = int(10)
    writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/coarse_mesh/burgers/in_out_norm')
    
    plotID = torch.tensor([0,12,24,36,48],dtype=torch.long)
    ID = torch.linspace(0,15000,50,dtype=torch.long).to(device)
    data = torch.load('burgers2.pth',map_location=device,)[ID].unsqueeze(1).contiguous().to(torch.float)
    label = data[plotID+1,0].detach().cpu()

    def add_plot(p,l):
        fig,ax = plt.subplots()
        ax.plot(p,label='Predicted')
        ax.plot(l,'--',label='label')
        ax.legend()
        return fig

    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data[:-1] - data[:-1].mean())/data[:-1].std()
            self.du = (data[1:] - data[:-1] - mcvter.up(pde_du(mcvter.down(data[:-1]))))[:,:,1:-1]
            # self.du = (data[1:] - data[:-1])[:,:,1:-1]       
            self.outmean = self.du.mean()
            self.outstd = self.du.std()
            self.du_normd = (self.du - self.outmean)/self.outstd

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index]

        def __len__(self):
            return self.u0_normd.shape[0]
            
    dataset=myset()
    inmean, instd = data[:-1].mean(), data[:-1].std()
    outmean, outstd = dataset.outmean, dataset.outstd
    print(dt/dx2,'\n')
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = mymlp(size=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterier = nn.MSELoss()


    for i in range(EPOCH):
        loshis = 0
        counter= 0
        
        for u0,du in train_loader:

            u_p = model(u0)
            loss = criterier(u_p, du)
            optimizer.zero_grad()
            loss.backward()
            loshis += loss.item()
            optimizer.step()
            counter += 1

        writer.add_scalar('loss', loshis/counter, i)
        
        if i%100 == 0:
            print('loss: {0:4f}\t epoch:{1:d}'.format(loshis/counter, i))

            model.eval()
            test_re = []
            u = data[0:1]
            for _ in range(49):
            
                u = padBC_p(model((u-inmean)/instd))*outstd + outmean + u + mcvter.up(pde_du(mcvter.down(u))) 
                test_re.append(u.detach().cpu())

            model.train()


            test_re = torch.cat(test_re,dim=0)
            
            for k,j in enumerate(plotID):
                writer.add_figure('test_'+str(j),add_plot(test_re[j,0],label[k]),i)
            writer.add_scalar('last_T_rel_error', 
                criterier(test_re[-1,0],label[k])/criterier(label[k],torch.zeros_like(label[k])), i)

    writer.close()