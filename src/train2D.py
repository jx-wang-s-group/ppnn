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
from operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
from utility.utils import mesh_convertor, model_count


if __name__=='__main__':
    inputfile = sys.argv[1]
    params = SimpleNamespace(**yaml.load(open(inputfile), Loader=yaml.FullLoader))
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    device = torch.device(params.device)
    feature_size = params.finemeshsize
    
    timesteps = params.timesteps
    

    mu = torch.linspace(params.paralow,params.parahigh,params.num_para,device=device).reshape(-1,1)
    mu = mu.repeat(params.repeat,1).reshape(-1,1,1,1)

    # mu for pdedu in dataset
    mus = mu.unsqueeze(1).repeat(1,timesteps,1,1,1)
    mus = mus.reshape(-1,1,1,1,)

    mutest = mu[0:1]

    def padbcx(uinner):
        return torch.cat((uinner[:,:,-1:],uinner,uinner[:,:,:1]),dim=2)

    def padbcy(uinner):
        return torch.cat((uinner[:,:,:,-1:],uinner,uinner[:,:,:,:1]),dim=3)


    def padBC_rd(u):
        tmp = torch.cat((u,u[:,:,:,:1]),dim=3)
        return torch.cat((tmp,tmp[:,:,:1]),dim=2)

    if params.pde:
        rhsu = getattr(rhs,params.rhsu)
        rhsv = getattr(rhs,params.rhsv)
        
        d2udx2 = d2udx2_2D(accuracy=2,device=device)
        d2udy2 = d2udy2_2D(accuracy=2,device=device)
        dudx = dudx_2D(accuracy=1,device=device)
        dudy = dudy_2D(accuracy=1,device=device)
        cmesh = params.coarsemeshsize
        dx = params.length/(cmesh-1)
        mcvter = mesh_convertor(feature_size,cmesh,dim=2)
        dx2 = dx**2
        dy,dy2=dx,dx2
        dt = params.dt

        print('\nCFL: {}\n'.format(dt/dx2))

        def pde_du(u,mu) -> torch.Tensor:
            u1 = mcvter.down(u[:,:1])[:,:,:-1,:-1]
            v1 = mcvter.down(u[:,1:])[:,:,:-1,:-1]
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


    EPOCH = int(params.epochs)+1
    BATCH_SIZE = int(params.batchsize)


    data:torch.Tensor = torch.load(params.datafile,map_location='cpu',)\
        [:,params.datatimestart:params.datatimestart+params.timesteps+1].to(torch.float)
    init = data[:,0]
    label = data[0,1:,].detach().cpu()
    data_u0 = data[:,:-1].reshape(-1,2,feature_size,feature_size).contiguous()
    data_du = (data[:,1:] - data[:,:-1,]).reshape(-1,2,feature_size,feature_size).contiguous()
    data=[]
    del data
    collect()


    def add_plot(p,l=None):
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        p0=ax[0].pcolormesh(p,clim=(l.min(),l.max()))
        fig.colorbar(p0,ax=ax[0])
        if l is not None:
            p2=ax[1].pcolormesh(l,clim=(l.min(),l.max()))
            fig.colorbar(p2,ax=ax[1])
        return fig


    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data_u0[:,:,:-1,:-1] - data_u0[:,:,:-1,:-1].mean(dim=(0,2,3),keepdim=True))\
                /data_u0[:,:,:-1,:-1].std(dim=(0,2,3),keepdim=True)
            
            
            if params.pde:
                pdeu = pde_du(data_u0.to(device), mus).cpu()
                self.du = (data_du - pdeu)[:,:,:-1,:-1].contiguous()
                pdeu=[]
                del pdeu
                collect()
            else:
                self.du = data_du[:,:,:-1,:-1].contiguous()

            
            
            self.outmean = self.du.mean(dim=(0,2,3),keepdim=True)
            self.outstd = self.du.std(dim=(0,2,3),keepdim=True)
            self.du_normd = (self.du - self.outmean)/self.outstd

            self.mu = mus.cpu()
            self.mumean = self.mu.mean()
            self.mustd = self.mu.std()
            self.mu_normd = (self.mu - self.mu.mean())/self.mu.std()

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index], self.mu_normd[index]

        def __len__(self):
            return self.u0_normd.shape[0]

    dataset = myset()
    inmean, instd = data_u0[:,:,:-1,:-1].mean(dim=(0,2,3),keepdim=True).to(device), \
        data_u0[:,:,:-1,:-1].std(dim=(0,2,3),keepdim=True).to(device)
    outmean, outstd = dataset.outmean.to(device), dataset.outstd.to(device)
    mumean, mustd = dataset.mumean.to(device), dataset.mustd.to(device)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,pin_memory=True,shuffle=True, num_workers=4)

    model = getattr(models,params.network)().to(device)

    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, cooldown=350, verbose=True, min_lr=1e-5)
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

            u0,du,mu = u0.to(device),du.to(device),mu.to(device)
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
            u = init[:1].to(device)
            for _ in range(timesteps):
            
                u_tmp = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,(mutest-mumean)/mustd)*outstd + outmean) + u
                if params.pde:
                    u_tmp += pde_du(u, mutest)
                u = u_tmp
                test_re.append(u.detach())

            model.train()

            test_re = torch.cat(test_re,dim=0).cpu()
            
            for testtime in [0,(timesteps-1)//2, -1]:
                writer.add_figure('u_time_{}'.format(testtime),
                                    add_plot(test_re[testtime,0],label[testtime,0]),
                                    i)
                writer.add_figure('v_time_{}'.format(testtime),
                                    add_plot(test_re[testtime,1],label[testtime,1]),
                                    i)

            test_error = criterier(test_re[-1],label[-1])/criterier(label[-1],torch.zeros_like(label[-1]))
            test_error_u = criterier(test_re[-1,0],label[-1,0])/criterier(label[-1,0],torch.zeros_like(label[-1,0]))
            test_error_v = criterier(test_re[-1,1],label[-1,1])/criterier(label[-1,1],torch.zeros_like(label[-1,1]))

            writer.add_scalar('U rel_error', test_error_u, i)
            writer.add_scalar('V rel_error', test_error_v, i)
            if test_error < test_error_best:
                test_error_best = test_error
                torch.save(model.state_dict(), params.modelsavepath)
    writer.close()