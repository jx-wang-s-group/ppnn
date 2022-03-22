from types import SimpleNamespace
from gc import collect
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

import models
from utility.utils import mesh_convertor, model_count
from utility.pyof import OneStepRunOFCoarse


if __name__=='__main__':
    inputfile = sys.argv[1]
    params = SimpleNamespace(**yaml.load(open(inputfile), Loader=yaml.FullLoader))
    
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    device = torch.device(params.device)
    feature_size = params.finemeshsize
    cmesh = params.coarsemeshsize

    timesteps = params.timesteps
    mcvter = mesh_convertor(feature_size, cmesh, dim=2, align_corners=False)

    # parameters
    pos = torch.linspace(params.paralow1,params.parahigh1,params.num_para1,device=device)
    Res = torch.linspace(params.paralow2,params.parahigh2,params.num_para2,device=device)
    pos,Res = torch.meshgrid(pos,Res,indexing='ij')
    pars = torch.stack((pos,Res),dim=-1).to(device)
    pars = pars.reshape(-1,1,2).repeat(1,timesteps,1).reshape(-1,2)

    

    parstest = torch.tensor([[0.5,10000]],device=device)
    coarsesolver = OneStepRunOFCoarse(params.template, params.tmp, params.dt, 
                        cmesh, parstest[0,0], parstest[0,1], cmesh[0])
    

    EPOCH = int(params.epochs)+1
    BATCH_SIZE = int(params.batchsize)


    fdata = torch.load(params.fdata, map_location='cpu')[:,params.datatimestart-1:params.datatimestart+params.timesteps].to(torch.float)
    
    if params.pde:
        pdeu = torch.load(params.cdata, map_location='cpu')[:,params.datatimestart:params.datatimestart+params.timesteps].to(torch.float)

    init = fdata[:,:1]
    label = fdata[:,1:].detach()
    data_u0 = fdata[:,:-1].contiguous()
    data_du = (fdata[:,1:] - fdata[:,:-1]).contiguous()
    fdata=[]
    del fdata
    collect()


    def add_plot(p,l=None):#
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        p0=ax[0].pcolormesh(p,clim=(l.min(),l.max()),cmap='coolwarm')
        fig.colorbar(p0,ax=ax[0])
        if l is not None:
            p2=ax[1].pcolormesh(l,clim=(l.min(),l.max()),cmap='coolwarm')
            fig.colorbar(p2,ax=ax[1])
        return fig


    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data_u0 - data_u0.mean(dim=(0,2,3),keepdim=True))\
                /data_u0.std(dim=(0,2,3),keepdim=True)

            global pdeu

            if params.pde:
                self.du = (data_du - pdeu).contiguous()
                pdeu=[]
                del pdeu
                collect()
            else:
                self.du = data_du.contiguous()

            
            self.outmean = self.du.mean(dim=(0,2,3),keepdim=True)
            self.outstd = self.du.std(dim=(0,2,3),keepdim=True)
            self.du_normd = (self.du - self.outmean)/self.outstd

            self.pars = pars.cpu()
            self.parsmean = self.pars.mean(dim=(0),keepdim=True)
            self.parsstd = self.pars.std(dim=(0),keepdim=True)
            self.pars_normd = (self.pars - self.parsmean)/self.parsstd

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index], self.mu_normd[index]

        def __len__(self):
            return self.u0_normd.shape[0]

    dataset = myset()
    inmean, instd = data_u0.mean(dim=(0,2,3),keepdim=True).to(device), \
        data_u0.std(dim=(0,2,3),keepdim=True).to(device)
    outmean, outstd = dataset.outmean.to(device), dataset.outstd.to(device)
    parsmean, parsstd = dataset.parsmean.to(device), dataset.parsstd.to(device)



    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,pin_memory=True,shuffle=True)

    model = getattr(models,params.network)(cmesh).to(device)

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
            for n in range(timesteps):
            
                u_tmp = (model((u-inmean)/instd,(parstest-parsmean)/parsstd)*outstd + outmean) + u
                if params.pde:
                    u_tmp0, error = coarsesolver(mcvter.down(u).detach().cpu())
                    if error:
                        for _ in range(timesteps-n):
                            test_re.append(float('nan')*torch.ones_like(u_tmp))
                        print('OpenFoam solver failed at step {0}!\n'.format(n))
                        break
                            
                    u_tmp0 = mcvter.up(u_tmp0).to(device)

                u = u_tmp
                test_re.append(u.detach())
            
            model.train()

            test_re = torch.cat(test_re,dim=0).cpu()
            
            for testtime in [0,(timesteps-1)//2, -1]:
                writer.add_figure('Velocity {}'.format(testtime),
                    add_plot(
                    torch.sqrt(test_re[testtime,0]**2 + test_re[testtime,1]**2),
                    torch.sqrt(label[testtime,0]**2 + label[testtime,1]**2)),
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