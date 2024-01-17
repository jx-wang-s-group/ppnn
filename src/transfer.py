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
from .operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
from .utility.utils import mesh_convertor, model_count


if __name__ == '__main__':

    # prepare data
    
    num_premodels = 2
    inputfile = sys.argv[1]
    params = SimpleNamespace(**yaml.load(open(inputfile), Loader=yaml.FullLoader))
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)

    device = torch.device(params.device)
    feature_size = params.finemeshsize
    cmesh = params.coarsemeshsize
    dx = params.length/(cmesh-1)
    timesteps = params.timesteps
    mcvter = mesh_convertor(feature_size,cmesh,dim=2)
    dx2 = dx**2
    dy,dy2=dx,dx2
    dt = params.dt

    mu = torch.linspace(params.paralow,params.parahigh,params.num_para,device=device).reshape(-1,1)
    mu = mu.repeat(params.repeat,1).reshape(-1,1,1,1)

    # mu for pdedu in dataset
    mus = mu.unsqueeze(1).repeat(1,timesteps,1,1,1)
    mus = mus.reshape(-1,1,1,1,)
    
    mupre = torch.linspace(params.paralow,params.parahigh,params.num_para,device=device).reshape(-1,1)
    mupre = mupre.repeat(32,1).reshape(-1,1,1,1)
    muspre = mupre.unsqueeze(1).repeat(1,timesteps,1,1,1)
    muspre = muspre.reshape(-1,1,1,1,)

    mutest = mu[0:1]

    rhsu = getattr(rhs,params.rhsu)
    rhsv = getattr(rhs,params.rhsv)

    rhsu0 = getattr(rhs,params.rhsu0)
    rhsv0 = getattr(rhs,params.rhsv0)
    rhsu1 = getattr(rhs,params.rhsu1)
    rhsv1 = getattr(rhs,params.rhsv1)

    rhsus = [rhsu0,rhsu1]
    rhsvs = [rhsv0,rhsv1]

    d2udx2 = d2udx2_2D(accuracy=2,device=device)
    d2udy2 = d2udy2_2D(accuracy=2,device=device)
    dudx = dudx_2D(accuracy=1,device=device)
    dudy = dudy_2D(accuracy=1,device=device)



    def padbcx(uinner):
        return torch.cat((uinner[:,:,-1:],uinner,uinner[:,:,:1]),dim=2)

    def padbcy(uinner):
        return torch.cat((uinner[:,:,:,-1:],uinner,uinner[:,:,:,:1]),dim=3)


    def padBC_rd(u):
        tmp = torch.cat((u,u[:,:,:,:1]),dim=3)
        return torch.cat((tmp,tmp[:,:,:1]),dim=2)
        
    def pde_du(u,mu,rhsu,rhsv) -> torch.Tensor:
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

    
    premodel0 = getattr(models,params.prenet0)()
    premodel1 = getattr(models,params.prenet1)()
    premodels = [premodel0,premodel1]
    premodelfiles = [params.prenetfile0,params.prenetfile1]
    for i,f in zip(premodels,premodelfiles):
        i.load_state_dict(torch.load(f,map_location='cpu'))
        i.eval()

    with torch.no_grad():
        predata = [torch.load(params.predata0,map_location='cpu',),
                torch.load(params.predata1,map_location='cpu',)]

        preu0 = [predata[i][:,:-1].reshape(-1,2,feature_size,feature_size) for i in range(num_premodels)]
        predu = [predata[i][:,1:].reshape(-1,2,feature_size,feature_size) - preu0[i] for i in range(num_premodels)]
        predata = []
        del predata
        collect()
        preinmean = [preu0[i][:-1,:,:-1,:-1].mean(dim=(0,2,3),keepdim=True) for i in range(num_premodels)]
        preinstd = [preu0[i][:-1,:,:-1,:-1].std(dim=(0,2,3),keepdim=True) for i in range(num_premodels)]
        preu0 = []
        del preu0
        collect()
        prepdeu = [pde_du((predu[i]).to(device), muspre, rhsus[i] ,rhsvs[i]).cpu() for i in range(num_premodels)]
        prepdedu = [(predu[i] - prepdeu[i])[:,:,:-1,:-1] for i in range(num_premodels)]
        preoutmean = [prepdedu[i].mean(dim=(0,2,3),keepdim=True) for i in range(num_premodels)]
        preoutstd = [prepdedu[i].std(dim=(0,2,3),keepdim=True) for i in range(num_premodels)]
        predu = []
        del predu
        collect()
        ID = torch.arange(0,255,16)
        data:torch.Tensor = torch.load(params.datafile,map_location='cpu',)\
            [ID,params.datatimestart:params.datatimestart+params.timesteps+1].to(torch.float)
        
        init = data[:,0]
        label = data[0,1:,].detach().cpu()
        data_u0 = data[:,:-1].reshape(-1,2,feature_size,feature_size).contiguous()
        
        pre_model_result = [premodels[i].cpu()((data_u0[:,:,:-1,:-1]-preinmean[i])/preinstd[i], (mus.cpu()-mus.mean().cpu())/mus.std().cpu())*preoutstd[i] + preoutmean[i]
            for i in range(num_premodels)]
        data_du = (data[:,1:] - data[:,:-1,]).reshape(-1,2,feature_size,feature_size).contiguous()
        for i in range(num_premodels):
            data_du -= padBC_rd(pre_model_result[i])
    
    preinmean = [preinmean[i].to(device) for i in range(num_premodels)]
    preinstd = [preinstd[i].to(device) for i in range(num_premodels)]
    preoutmean = [preoutmean[i].to(device) for i in range(num_premodels)]
    preoutstd = [preoutstd[i].to(device) for i in range(num_premodels)]
    premodels = [premodels[i].to(device) for i in range(num_premodels)]
    prepdeu = []
    prepdedu = []
    data=[]
    del prepdeu, prepdedu, data
    collect()

    def add_plot(p,l=None):#
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        p0=ax[0].pcolormesh(p,clim=(l.min(),l.max()))
        fig.colorbar(p0,ax=ax[0])
        if l is not None:
            p2=ax[1].pcolormesh(l,clim=(l.min(),l.max()))
            fig.colorbar(p2,ax=ax[1])
        return fig

    class myset(torch.utils.data.Dataset):
        def __init__(self):
            global data_u0,data_du
            self.u0_normd = (data_u0[:,:,:-1,:-1] - data_u0[:,:,:-1,:-1].mean(dim=(0,2,3),keepdim=True))\
                /data_u0[:,:,:-1,:-1].std(dim=(0,2,3),keepdim=True)
            
            pdeu = pde_du(data_u0.to(device), mus, rhsu, rhsv).cpu()
            if params.pde:
                self.du = (data_du - pdeu)[:,:,:-1,:-1].contiguous()
            else:
                self.du = data_du[:,:,:-1,:-1].contiguous()

            for i in range(num_premodels):
                self.du -= pre_model_result[i]

            data_du = []
            pdeu=[]
            del pdeu, data_du
            collect()
            

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
    data_u0 = []
    del data_u0
    collect()
    outmean, outstd = dataset.outmean.to(device), dataset.outstd.to(device)
    mumean, mustd = dataset.mumean.to(device), dataset.mustd.to(device)

    print('\nCFL: {}\n'.format(dt/dx2))


    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,pin_memory=True,shuffle=True)

    model = getattr(models,params.network)().to(device)

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
            
                u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,(mutest-mumean)/mustd)*outstd + outmean + 
                    premodels[0]((u[:,:,:-1,:-1]-preinmean[0])/preinstd[0],(mutest-mumean)/mustd)*preoutstd[0] + preoutmean[0] +
                    premodels[1]((u[:,:,:-1,:-1]-preinmean[1])/preinstd[1],(mutest-mumean)/mustd)*preoutstd[1] + preoutmean[1]
                    ) + u
                if params.pde:
                    u += pde_du(u,mutest,rhsu,rhsv)
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