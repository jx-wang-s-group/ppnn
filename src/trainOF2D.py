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
    dt = params.dt

    timesteps = params.timesteps
    begintime = params.datatimestart
    mcvter = mesh_convertor(feature_size, cmesh, dim=2, align_corners=False)

    # parameters
    pos = torch.linspace(params.para1low,params.para1high,params.num_para1,device=device)
    Res = torch.linspace(params.para2low,params.para2high,params.num_para2,device=device)
    pos,Res = torch.meshgrid(pos,Res,indexing='ij')
    pars = torch.stack((pos,Res),dim=-1).to(device)
    pars = pars.reshape(-1,1,2,1).repeat(1,timesteps,1,1).reshape(-1,2,1,1)

    

    parstest = torch.tensor([[[[0.5]],[[0.0001]]]],device=device)
    if params.pde:
        coarsesolver = OneStepRunOFCoarse(params.template, params.tmp, params.dt, 
                            cmesh, parstest[0,0].squeeze().item(),
                            parstest[0,1].squeeze().item(), cmesh[0],
                            params.solver)
    

    EPOCH = int(params.epochs)+1
    BATCH_SIZE = int(params.batchsize)


    fdata = torch.load(params.fdata, map_location='cpu').detach().to(torch.float)
    fdata = torch.cat((fdata,torch.sqrt(fdata[:,:,0:1]**2 + fdata[:,:,1:2]**2)),dim=2)
    
    if params.pde:
        pdeu = torch.load(params.cdata, map_location='cpu').detach().to(torch.float)
        pdeu = torch.cat((pdeu,torch.sqrt(pdeu[:,:,0:1]**2 + pdeu[:,:,1:2]**2)),dim=2)
        data_du = (fdata[:,1:,:-1] - pdeu[:,:,:-1]).reshape(-1,3,*feature_size).contiguous()
        pdeu = pdeu.reshape(-1,4,*feature_size).contiguous()
        # pdeu = []
        # del pdeu
        # collect()
    else:
        data_du = (fdata[:,1:,:-1] - fdata[:,:-1,:-1]).reshape(-1,3,*feature_size).contiguous()

    init = fdata[24,:1,:-1]
    label = fdata[24,1:,:-1].detach()
    data_u0 = fdata[:,:-1].reshape(-1,4,*feature_size).contiguous()
    
    
    fdata=[]
    del fdata
    collect()


    def add_plot(p,l=None):#
        fig,ax = plt.subplots(2,1,figsize=(10,5))
        p0=ax[0].pcolormesh(p,clim=(l.min(),l.max()),cmap='coolwarm')
        fig.colorbar(p0,ax=ax[0])
        if l is not None:
            p2=ax[1].pcolormesh(l,clim=(l.min(),l.max()),cmap='coolwarm')
            fig.colorbar(p2,ax=ax[1])
        fig.tight_layout()
        return fig


    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data_u0 - data_u0.mean(dim=(0,2,3),keepdim=True))\
                /data_u0.std(dim=(0,2,3),keepdim=True)

            
            self.outmean = data_du.mean(dim=(0,2,3),keepdim=True)
            self.outstd = data_du.std(dim=(0,2,3),keepdim=True)
            self.du_normd = (data_du - self.outmean)/self.outstd

            
            self.parsmean = pars.cpu().mean(dim=(0,2,3),keepdim=True)
            self.parsstd = pars.cpu().std(dim=(0,2,3),keepdim=True)
            self.pars_normd = (pars.cpu() - self.parsmean)/self.parsstd

            if params.pde:
                self.pdemean = pdeu.mean(dim=(0,2,3),keepdim=True)
                self.pdestd = pdeu.std(dim=(0,2,3),keepdim=True)
                self.pdeu_normd = (pdeu - self.pdemean)/self.pdestd

        def __getitem__(self, index):
            if params.pde:
                return self.u0_normd[index], self.du_normd[index], self.pars_normd[index], self.pdeu_normd[index]
            else:
                return self.u0_normd[index], self.du_normd[index], self.pars_normd[index]

        def __len__(self):
            return self.u0_normd.shape[0]

    dataset = myset()
    inmean, instd = data_u0.mean(dim=(0,2,3),keepdim=True).to(device), \
        data_u0.std(dim=(0,2,3),keepdim=True).to(device)
    outmean, outstd = dataset.outmean.to(device), dataset.outstd.to(device)
    parsmean, parsstd = dataset.parsmean.to(device), dataset.parsstd.to(device)

    if params.pde:
        pdemean, pdestd = dataset.pdemean.to(device), dataset.pdestd.to(device)

        pdeu = []
        del pdeu
        collect()

    data_u0, data_du=[],[]
    del data_u0, data_du
    collect()

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,pin_memory=True,shuffle=True,num_workers=4)

    model = getattr(models,params.network)(cmesh,feature_size).to(device)

    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, cooldown=350, verbose=True, min_lr=1e-5)
    criterier = nn.MSELoss()

    test_error_best = 0.5
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(params.tensorboarddir)
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
                if params.pde: pdeu += 0.05*pdeu.std()*torch.randn_like(pdeu)

            
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
            u = init[:1].to(device)
            for n in range(timesteps):
                
                u = torch.cat((u,torch.sqrt(u[:,0:1]**2+u[:,1:2]**2)),dim=1)
               
                if params.pde:
                    u_tmp0, error = coarsesolver(mcvter.down(u[:,:-1]).detach().cpu(),begintime + n*dt)
                    if error:
                        for _ in range(timesteps-n):
                            test_re.append(float('nan')*torch.ones_like(u_tmp))
                        print('OpenFoam solver failed at step {0}!\n'.format(n))
                        break
                            
                    u_tmp0 = mcvter.up(u_tmp0).to(device)
                    u_tmp = model(
                            (u-inmean)/instd,
                            (parstest-parsmean)/parsstd,
                            (torch.cat((u_tmp0,torch.sqrt(u_tmp0[:,0:1]**2+u_tmp0[:,1:2]**2)),dim=1)-pdemean)/pdestd
                        )*outstd + outmean 
                    u_tmp += u_tmp0
                else:
                    u_tmp = model(
                            (u-inmean)/instd,
                            (parstest-parsmean)/parsstd,
                        )*outstd + outmean + u[:,:-1]

                u = u_tmp
                test_re.append(u.detach())
            
            model.train()

            test_re = torch.cat(test_re,dim=0).cpu()
            
            for testtime in [0,(timesteps-1)//4,(timesteps-1)//2,3*(timesteps-1)//4, -1]:
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
            writer.add_scalar('rel_error', test_error, i)
            if test_error < test_error_best:
                test_error_best = test_error
                torch.save(model.state_dict(), params.modelsavepath)
    writer.close()