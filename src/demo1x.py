import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from operators import diffusion1D, convection1d
from src.utility.utils import mesh_convertor, model_count
from demo0 import mymlp,lblock

def gen_net(layers,insize,outsize,hsize):
    l = [nn.Linear(insize,hsize),] 
    for _ in range(layers):
        l.append(lblock(hsize))
    l.append(nn.Linear(hsize, outsize))
    return nn.Sequential(*l)


class input_enrich(nn.Module):
    def __init__(self, w0, w1, input_size = 101, output_size = 99, h0=2, h1 = 2):
        super().__init__()
        self.net1 = gen_net(h0, input_size, w0, w0)
        self.net = gen_net(h1, w0*2, output_size, w1)
        self.pdeinput = gen_net(h0, input_size, w0, w0)

    def forward(self, u, pdeu):
        return self.net(torch.cat((self.net1(u),self.pdeinput(pdeu)),dim=2))

def padBC_p(u0):
    return torch.cat((u0[:,:,-1:],u0,u0[:,:,:1]),dim=2)

def padBC_p_r(u0):
    return torch.cat((u0,u0[:,:,:1]),dim=2)

def pde_du(u0, mu=0.04) -> torch.Tensor:
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
    EPOCH = int(4e3)+1
    BATCH_SIZE = int(10)
    writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/Jan27/noPDE-h4-w60')
    
    plotID = torch.tensor([0,12,24,36,49],dtype=torch.long)
    ID = torch.linspace(0,15000,51,dtype=torch.long).to(device)
    data = torch.load('burgers_test.pth',map_location=device,)[ID].unsqueeze(1).contiguous().to(torch.float)
    label = data[plotID+1,0].detach().cpu()

    def add_plot(p,l=None):#
        fig,ax = plt.subplots()
        ax.plot(p,label='pdedu')
        if l is not None:
            ax.plot(l,'--',label='du')
            ax.legend()
        return fig

    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data[:-1,:,:-1] - data[:-1,:,:-1].mean())/data[:-1,:,:-1].std()
            
            self.pdedu = pde_du(data[:-1])[:,:,:-1] 
            self.pdedumean = self.pdedu.mean()
            self.pdedustd = self.pdedu.std()
            self.pdedu = (self.pdedu - self.pdedumean)/self.pdedustd

            # self.du = (data[1:] - data[:-1] - pde_du(data[:-1]))[:,:,:-1]
            self.du = (data[1:] - data[:-1])[:,:,:-1]       
            self.outmean = self.du.mean()
            self.outstd = self.du.std()
            self.du_normd = (self.du - self.outmean)/self.outstd

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index], self.pdedu[index]

        def __len__(self):
            return self.u0_normd.shape[0]
    
    dataset=myset()
    # for i in range(50):
    #     writer.add_figure('in',add_plot(dataset.u0_normd[i].squeeze().cpu(),),global_step=i)
    #     writer.add_figure('out',add_plot(dataset.du[i].squeeze().cpu(),),global_step=i)
    #     # writer.add_figure('pdedu',add_plot(dataset.pdedu[i].squeeze().cpu()),global_step=i)
    #     writer.add_figure('pdedu',add_plot(dataset.du[i].squeeze().cpu(),dataset.du[i].squeeze().cpu()),global_step=i)
    inmean, instd = data[:-1,:,:-1].mean(), data[:-1,:,:-1].std()
    pdemean, pdestd = dataset.pdedumean, dataset.pdedustd
    outmean, outstd = dataset.outmean, dataset.outstd
    print(dt/dx2,'\n')
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = mymlp(size=60,input_size=100,output_size=100,hidden_layers=4).to(device)
    # model = input_enrich(w0=36,w1=48,input_size=100,output_size=100,h0=2,h1=2).to(device)
    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.5,patience=100,cooldown=100,verbose=True,min_lr=5e-5)
    criterier = nn.MSELoss()

    test_error_best = 1
    for i in range(EPOCH):
        loshis = 0
        counter= 0
        
        for u0,du,pdedu in train_loader:

            # u_p = model(u0, pdedu)
            u_p = model(u0)
            loss = criterier(u_p, du)
            optimizer.zero_grad()
            loss.backward()
            loshis += loss.item()
            optimizer.step()
            counter += 1

        writer.add_scalar('loss', loshis/counter, i)
        scheduler.step(loshis/counter)
        if i%100 == 0:
            print('loss: {0:4f}\t epoch:{1:d}'.format(loshis/counter, i))

            model.eval()
            test_re = []
            u = data[0:1]
            for _ in range(50):
            
                # u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd, (pde_du(u)[:,:,:-1]-pdemean)/pdestd)*outstd + outmean) + u + pde_du(u)
                u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd)*outstd + outmean) + u #+ pde_du(u)
                test_re.append(u.detach().cpu())

            model.train()


            test_re = torch.cat(test_re,dim=0)
            
            for k,j in enumerate(plotID):
                writer.add_figure('test_'+str(j),add_plot(test_re[j,0],label[k]),i)
            test_error = criterier(test_re[-1,0],label[k])/criterier(label[k],torch.zeros_like(label[k]))
            writer.add_scalar('last_T_rel_error', test_error, i)
            if test_error < test_error_best:
                test_error_best = test_error
                torch.save(model.state_dict(), 'model_noPDE-h4-w60.pth')
    writer.close()