from types import SimpleNamespace
from gc import collect
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

import models
import rhs
from .operators import dudx_1D, diffusion1D
from .utility.utils import mesh_convertor, model_count


if __name__=='__main__':
    inputfile = sys.argv[1]
    params = SimpleNamespace(**yaml.load(open(inputfile), Loader=yaml.FullLoader))
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    device = torch.device(params.device)
    feature_size = params.finemeshsize
    cmesh = params.coarsemeshsize
    dx = params.length/(cmesh-1)
    timesteps = params.timesteps
    mcvter = mesh_convertor(feature_size,cmesh,dim=1)
    dx2 = dx**2
    dx4 = dx2**2
    dt = params.dt

    mu = torch.linspace(params.paralow,params.parahigh,params.num_para,device=device).reshape(-1,1)
    mu = mu.repeat(params.repeat,1).reshape(-1,1,1,1)

    # mu for pdedu in dataset
    mus = mu.unsqueeze(1).repeat(1,timesteps,1,1,1)
    mus = mus.reshape(-1,1,1,1,)

    mutest = mu[0:1]

    rhsu = getattr(rhs,params.rhs)
    
    d2udx2 = diffusion1D(accuracy=2,device=device)
    dudx = dudx_1D(accuracy=1,device=device)
    d4udx4 = diffusion1D('Central_4th',accuracy=2,device=device)
    

    def padbc(uinner):
        return torch.cat((uinner[:,:,-1:],uinner,uinner[:,:,:1]),dim=2)

    def padbc4(uinner):
        return torch.cat((uinner[:,:,-2:],uinner,uinner[:,:,:2]),dim=2)

    def padBC_rd(u):
        return torch.cat((u,u[:,:,:1]),dim=2)
        
    def pde_du(u,mu) -> torch.Tensor:
        u1 = mcvter.down(u)[:,:,:-1]
        ux = padbc(u1)
        u4 = padbc4(u1)
        return mcvter.up(padBC_rd(dt*rhsu(u4,mu,dudx,d2udx2,d4udx4,ux,dx,dx2,dx4)))
            

    EPOCH = int(params.epochs)+1
    BATCH_SIZE = int(params.batchsize)


    data:torch.Tensor = torch.load(params.datafile,map_location=device,)\
        [:,params.datatimestart:params.datatimestart+params.timesteps+1].to(torch.float)
    init = data[:,0]
    label = data[:,1:,0].detach().cpu()
    data_u0 = data[:,:-1].reshape(-1,1,feature_size).contiguous()
    data_du = (data[:,1:] - data[:,:-1,]).reshape(-1,1,feature_size).contiguous()
    data=[]
    del data
    collect()

    def add_plot(p, t):
        fig,ax = plt.subplots()
        ax.plot(p,label='Pred')
        ax.plot(t,label='True')
        ax.legend()
        return fig

    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data_u0[:,:,:-1] - data_u0[:,:,:-1].mean())/data_u0[:,:,:-1].std()
            pdeu = pde_du(data_u0, mus)

            if params.pde:
                self.du = (data_du - pdeu)[:,:,:-1].contiguous()
            else:
                self.du = data_du[:,:,:-1].contiguous()

            pdeu=[]
            del pdeu
            collect()
            
            self.outmean = self.du.mean()
            self.outstd = self.du.std()
            self.du_normd = (self.du - self.outmean)/self.outstd

            self.mu = mus
            self.mumean = self.mu.mean()
            self.mustd = self.mu.std()
            self.mu_normd = (self.mu - self.mu.mean())/self.mu.std()

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index], self.mu_normd[index]

        def __len__(self):
            return self.u0_normd.shape[0]

    dataset = myset()
    inmean, instd = data_u0[:,:,:-1].mean(), data_u0[:,:,:-1].std()
    outmean, outstd = dataset.outmean, dataset.outstd
    mumean, mustd = dataset.mumean, dataset.mustd

    print('\nCFL: {}\n'.format(dt/dx2))

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)

    model = getattr(models,params.network)(feature_size-1,
                                           params.hidden_size,
                                           params.hidden_layers).to(device)

    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, cooldown=200, verbose=True, min_lr=1e-5)
    criterier = nn.MSELoss()

    test_error_best = 0.01
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(params.tensorboarddir)
    for i in range(EPOCH):
        loshis = 0
        counter= 0
        
        for u0,du,mu in train_loader:

            if params.noiseinject:
                u0 += 0.05*u0.std()*torch.randn_like(u0)

            u_p = model(u0,mu)
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
            u = init
            for _ in range(timesteps):
            
                u_tmp = padBC_rd(model((u[:,:,:-1]-inmean)/instd,(mutest-mumean)/mustd)*outstd + outmean) + u
                if params.pde:
                    u_tmp += pde_du(u,mutest)
                u = u_tmp
                test_re.append(u.detach())

            model.train()

            test_re = torch.cat(test_re,dim=1).cpu()
            
            for testtime in [0,(timesteps-1)//4, (timesteps-1)//2, 3*(timesteps-1)//4, -1]:
                writer.add_figure('u_time_{}'.format(testtime),
                                    add_plot(test_re[0,testtime],label[0,testtime]), i)

            test_error = criterier(test_re[:,-1],label[:,-1])/criterier(label[:,-1],torch.zeros_like(label[:,-1]))
            
            writer.add_scalar('rel_error', test_error, i)
        
            if test_error < test_error_best:
                test_error_best = test_error
                torch.save(model.state_dict(), params.modelsavepath)
    writer.close()