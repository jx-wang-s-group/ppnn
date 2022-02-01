import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from operators import diffusion1D, convection1d
from src.utility.utils import mesh_convertor
from demo0 import mymlp

def padBC_p(u0):
    return torch.cat((u0[:,:,-1:],u0,u0[:,:,:1]),dim=2)

def padBC_p_r(u0):
    return torch.cat((u0,u0[:,:,:1]),dim=2)

def pde_du(u0):
    u1 = mcvter.down(u0)[:,:,:-1]
    return mcvter.up(padBC_p_r((mu*diffusr(padBC_p(u1))/dx2-0.5*convector(padBC_p(u1*u1))/dx)*dt))

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
    EPOCH = int(8e3)+1
    BATCH_SIZE = int(49)
    writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/coarse_mesh/burgers/tr/PDE-nonorm-noNaN')
    
    plotID = torch.tensor([0,12,24,36,48],dtype=torch.long)
    ID = torch.linspace(0,15000,50,dtype=torch.long).to(device)
    data = torch.load('burgers2.pth',map_location=device,)[ID].unsqueeze(1).contiguous().to(torch.float)
    label = data[plotID+1,0].detach().cpu()

    def add_plot(p,l):#
        fig,ax = plt.subplots()
        ax.plot(p,label='Predicted')
        ax.plot(l,'--',label='label')
        ax.legend()
        return fig

    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0 = data[:-1,]
            self.u1 = data[1:,]      

        def __getitem__(self, index):
            return self.u0[index], self.u1[index]

        def __len__(self):
            return self.u0.shape[0]
    
    dataset=myset()

    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = mymlp(size=10,input_size=101,output_size=100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterier = nn.MSELoss()


    for i in range(EPOCH):
        loshis = 0
        counter= 0
        re = []
        u0 = dataset.u0[:1]
        if i <= 1000:
            for u0,u1 in train_loader:
               
                optimizer.zero_grad()
                u1_pred = padBC_p_r(model(u0)) + u0 + pde_du(u0)
                
                loss = criterier(u1_pred,u1)
                loss.backward()
                optimizer.step()
                loshis += loss.item()
                counter += 1
        else:

            length = 0
            for j in range(49):

                u0 = padBC_p_r(model(u0)) + u0 + pde_du(u0)
                if u0.isnan().any():
                    break
                re.append(u0)
                length += 1
            if len(re)==0:
                break
            re = torch.cat(re,dim=0)
            
                
            loss = criterier(re, dataset.u1[:length])
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
            u = data[:1]

            for _ in range(49):
            
                u = padBC_p_r(model(u)) + u + pde_du(u)
                test_re.append(u.detach().cpu())

            model.train()


            test_re = torch.cat(test_re,dim=0)
            
            for k,j in enumerate(plotID):
                writer.add_figure('test_'+str(j),add_plot(test_re[j,0],label[k]),i)
            writer.add_scalar('last_T_rel_error', 
                criterier(test_re[-1,0],label[k])/criterier(label[k],torch.zeros_like(label[k])), i)

    writer.close()