from types import SimpleNamespace
from gc import collect
import os
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
    
    # every = 1
    dataonly = 0
    mu = torch.linspace(params.paralow,params.parahigh,params.num_para,device=device).reshape(-1,1)
    mu = mu.repeat(params.repeat,1).reshape(-1,1,1,1)

    # mu for pdedu in dataset
    mus = mu.unsqueeze(1)[:].repeat(1,timesteps,1,1,1)#dataonly
    mus = mus.reshape(-1,1,1,1,)

    mutest = mu[0:params.num_para]
    testIDs = torch.linspace(0,params.num_para-1,params.num_para, device=device,dtype = torch.long)
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


    fdata:torch.Tensor = torch.load(params.datafile,map_location='cpu',)['u']\
        [:,params.datatimestart:params.datatimestart+params.timesteps+1].detach().to(torch.float)[:]#dataonly
    init = fdata[testIDs,0]
    label = fdata[testIDs,1:,].detach().cpu()
    data_u0 = fdata[:,:-1].reshape(-1,2,feature_size,feature_size)#.contiguous()
    data_du = (fdata[:,1:] - fdata[:,:-1,]).reshape(-1,2,feature_size,feature_size)#.contiguous()
    
    
    fdata=[]    
    del fdata
    collect()

    def magnitude(x):
        return torch.sqrt(torch.sum(x**2,dim=-3))

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
            global data_u0, data_du

            self.inmean = data_u0[:,:,:-1,:-1].mean(dim=(0,2,3),keepdim=True)
            self.instd = data_u0[:,:,:-1,:-1].std(dim=(0,2,3),keepdim=True)
            self.u0_normd = (data_u0[:,:,:-1,:-1] - self.inmean)/self.instd
            
            
            if params.pde:
                pdeu = pde_du(data_u0.to(device), mus).cpu()
                
                del data_u0
                collect()

                self.pdemean = pdeu.mean(dim=(0,2,3),keepdim=True)
                self.pdestd = pdeu.std(dim=(0,2,3),keepdim=True)
                self.pdeu_normd = (pdeu[:,:,:-1,:-1] - self.pdemean)/self.pdestd
                du = (data_du - pdeu)[:,:,:-1,:-1]#

                del pdeu
                collect()
                del data_du
                collect()
            else:
                del data_u0
                collect()
                du = data_du[:,:,:-1,:-1]#.contiguous()
                del data_du
                collect()

            
            
            self.outmean = du.mean(dim=(0,2,3),keepdim=True)
            self.outstd = du.std(dim=(0,2,3),keepdim=True)
            self.du_normd = (du - self.outmean)/self.outstd

            self.mu = mus.cpu()
            self.mumean = self.mu.mean()
            self.mustd = self.mu.std()
            self.mu_normd = (self.mu - self.mu.mean())/self.mu.std()

        def __getitem__(self, index):
            if params.pde:
                return self.u0_normd[index], self.du_normd[index], self.mu_normd[index], self.pdeu_normd[index]
            else:
                return self.u0_normd[index], self.du_normd[index], self.mu_normd[index]

        def __len__(self):
            return self.u0_normd.shape[0]

    dataset = myset()
    inmean, instd = dataset.inmean.to(device), dataset.instd.to(device)
    outmean, outstd = dataset.outmean.to(device), dataset.outstd.to(device)
    mumean, mustd = dataset.mumean.to(device), dataset.mustd.to(device)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(params.tensorboarddir)

    torch.save(inmean, os.path.join(params.tensorboarddir,'inmean.pt'))
    torch.save(instd, os.path.join(params.tensorboarddir,'instd.pt'))
    torch.save(outmean, os.path.join(params.tensorboarddir,'outmean.pt'))
    torch.save(outstd, os.path.join(params.tensorboarddir,'outstd.pt'))
    torch.save(mumean, os.path.join(params.tensorboarddir,'parsmean.pt'))
    torch.save(mustd, os.path.join(params.tensorboarddir,'parsstd.pt'))

    if params.pde:
        pdemean, pdestd = dataset.pdemean.to(device), dataset.pdestd.to(device)
        torch.save(pdemean, os.path.join(params.tensorboarddir,'pdemean.pt'))
        torch.save(pdestd, os.path.join(params.tensorboarddir,'pdestd.pt'))


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,pin_memory=True,shuffle=True, num_workers=8)

    model = getattr(models,params.network)().to(device)

    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, cooldown=250, verbose=True, min_lr=5e-6)
    criterier = nn.MSELoss()

    test_error_best = 0.1

    for i in range(EPOCH):
        loshis = 0
        counter= 0
        
        for data in train_loader:
            
            if params.pde:
                u0, du, mu, pdeu = data
                u0,du,mu,pdeu = u0.to(device),du.to(device),mu.to(device),pdeu.to(device)
            else:
                u0, du, mu = data
                u0,du,mu = u0.to(device),du.to(device),mu.to(device)
                
            if params.noiseinject:
                u0 += 0.05*u0.std()*torch.randn_like(u0)

            if params.pde:
                u_p = model(u0,mu,pdeu)
            else:
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
            u = init.to(device)
            for _ in range(timesteps):
            
                
                if params.pde:    
                    pdeuu = pde_du(u, mutest)
                    u_tmp = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,
                                           (mutest-mumean)/mustd, 
                                           (pdeuu[:,:,:-1,:-1]-pdemean)/pdestd)*outstd + outmean)\
                            + u + pdeuu
                else:
                    u_tmp = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,
                                           (mutest-mumean)/mustd)*outstd + outmean)\
                            + u

                u = u_tmp
                test_re.append(u.detach())

            model.train()

            test_re = torch.stack(test_re,dim=1).cpu()
            
            for testtime in [0,(timesteps-1)//2, -1]:
                writer.add_figure('p0_{}'.format(testtime),
                                    add_plot(magnitude(test_re[0,testtime]),
                                            magnitude(label[0,testtime])),
                                    i)
                writer.add_figure('p-1_{}'.format(testtime),
                                    add_plot(magnitude(test_re[-1,testtime]),
                                            magnitude(label[-1,testtime])),
                                    i)

            test_error = criterier(test_re[:,-1],label[:,-1])/criterier(label[:,-1],torch.zeros_like(label[:,-1]))
            
            writer.add_scalar('rel_error', test_error, i)
            # if test_error < test_error_best:
            #     test_error_best = test_error
            torch.save(model.state_dict(), params.modelsavepath)
    writer.close()