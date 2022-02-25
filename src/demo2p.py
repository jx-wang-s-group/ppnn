from math import pi
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from demo0 import lblock

class cblock(nn.Module):
    def __init__(self,hc,ksize,feature_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hc,hc,ksize,padding=ksize//2),
            nn.ReLU(),
            nn.Conv2d(hc,hc,ksize,padding=ksize//2),
        )
        self.ln = nn.LayerNorm(feature_size)
        
    def forward(self,x):
        return self.ln(self.net(x)) + x


class mycnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(

            # nn.Conv2d(3,32,6,stride=2,padding=2),
            # nn.ReLU(),
            # cblock(32,5,[32,128,128]),
            # nn.Conv2d(32,128,6,stride=2,padding=2),
            # nn.ReLU(),
            # cblock(128,5,[128,64,64]),
            # nn.Conv2d(128,256,6,stride=2,padding=2),
            # nn.ReLU(),
            # cblock(256,5,[256,32,32]),
            # nn.Conv2d(256,512,6,stride=2,padding=2),
            # nn.ReLU(),
            # cblock(512,5,[512,16,16]),
            # nn.PixelShuffle(16),#185
            # nn.Conv2d(2,2,5,padding=2),
            # nn.ReLU(),
            # nn.Conv2d(2,2,5,padding=2),



            nn.Conv2d(3,12,6,stride=2,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(12,48,6,stride=2,padding=2),
            nn.ReLU(),
            cblock(48,5,[48,64,64]),
            cblock(48,5,[48,64,64]),
            cblock(48,5,[48,64,64]),
            nn.PixelShuffle(4),#185
            nn.Conv2d(3,2,5,padding=2),
        )
        self.cw = nn.Parameter(torch.randn(1,1,1,256)) 
        self.rw = nn.Parameter(torch.randn(1,1,256,1))

    # def forward(self,u0,mu,pdeu):
    #     return self.net(torch.cat((u0,mu*self.rw@self.cw,pdeu),dim=1))
    def forward(self,u0,mu):
        return self.net(torch.cat((u0,mu*self.rw@self.cw),dim=1))


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    from src.operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
    from src.utility.utils import mesh_convertor, model_count

    np.random.seed(10)
    torch.manual_seed(10)
    device = torch.device('cuda:2')
    feature_size = 257
    cmesh = 49
    dx = 3.2/(cmesh-1)
    mcvter = mesh_convertor(feature_size,cmesh,dim=2)
    dx2 = dx**2
    dy,dy2=dx,dx2
    dt = 100*1e-4

    mu = torch.linspace(0.02,0.24,23,device=device).reshape(-1,1,1,1)

    # mu for pdedu in dataset
    mus = mu.unsqueeze(1).repeat(1,100,1,1,1)
    mus = mus.reshape(-1,1,1,1,)

    mutest = mu[0:1]

    
    
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
        
    def pde_du(u,mu) -> torch.Tensor:
        u1 = mcvter.down(u[:,:1])[:,:,:-1,:-1]
        v1 = mcvter.down(u[:,1:])[:,:,:-1,:-1]
        ux = padbcx(u1)
        uy = padbcy(u1)
        vx = padbcx(v1)
        vy = padbcy(v1)
        return torch.cat((
            mcvter.up(
                padBC_rd(dt*(-u1*dudx(ux)/dx - v1*dudy(uy)/dy + mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2)))
            ),
            mcvter.up(
                padBC_rd(dt*(-u1*dudx(vx)/dx - v1*dudy(vy)/dy + mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2)))
            )),dim=1)

    EPOCH = int(5e3)+1
    BATCH_SIZE = int(800)
    

    data:torch.Tensor = torch.load('burgers_p_2D.pth',map_location='cpu',)[:,:-101].to(torch.float)
    label = data[0,1:,].detach().cpu()
    data_u0 = data[:,:-1].reshape(-1,2,feature_size,feature_size).contiguous()
    data_du = (data[:,1:] - data[:,:-1,]).reshape(-1,2,feature_size,feature_size).contiguous()
    


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
            self.u0_normd = (data_u0[:,:,:-1,:-1] - data_u0[:,:,:-1,:-1].mean(dim=(0,2,3),keepdim=True))\
                /data_u0[:,:,:-1,:-1].std(dim=(0,2,3),keepdim=True)
            
            pdeu = pde_du(data_u0.to(device), mus).cpu()
            self.pdeu = pdeu[:,:,:-1,:-1].contiguous()
            self.pdeumean = self.pdeu.mean(dim=(0,2,3),keepdim=True)
            self.pdeustd = self.pdeu.std(dim=(0,2,3),keepdim=True)
            self.pdeu = (self.pdeu - self.pdeumean)/self.pdeustd

            self.du = (data_du - pdeu)[:,:,:-1,:-1].contiguous()
            # self.du = data_du[:,:,:-1,:-1].contiguous()
            self.outmean = self.du.mean(dim=(0,2,3),keepdim=True)
            self.outstd = self.du.std(dim=(0,2,3),keepdim=True)
            self.du_normd = (self.du - self.outmean)/self.outstd

            self.mu = mus.cpu()
            self.mumean = self.mu.mean()
            self.mustd = self.mu.std()
            self.mu_normd = (self.mu - self.mu.mean())/self.mu.std()

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index], self.mu_normd[index],self.pdeu[index]

        def __len__(self):
            return self.u0_normd.shape[0]

    dataset = myset()
    inmean, instd = data_u0[:,:,:-1,:-1].mean(dim=(0,2,3),keepdim=True).to(device), \
        data_u0[:,:,:-1,:-1].std(dim=(0,2,3),keepdim=True).to(device)
    pdemean, pdestd = dataset.pdeumean.to(device), dataset.pdeustd.to(device)
    outmean, outstd = dataset.outmean.to(device), dataset.outstd.to(device)
    mumean, mustd = dataset.mumean.to(device), dataset.mustd.to(device)
    
    print(dt/dx2,'\n')
    assert dt/dx2 < 1

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,pin_memory=True,shuffle=True)

    model = mycnn().to(device)
    
    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, cooldown=200, verbose=True, min_lr=1e-5)
    criterier = nn.MSELoss()

    test_error_best = 1
    writerdir = '/home/xinyang/store/dynamic/PDE_structure/2D/burgers/less-data/PDE-enrich-bigmodel'
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(writerdir)
    for i in range(EPOCH):
        loshis = 0
        counter= 0
        
        for u0,du,mu,pdeu in train_loader:
            u0,du,mu,pdeu = u0.to(device),du.to(device),mu.to(device),pdeu.to(device)
            u_p = model(u0, mu, pdeu)
            # noise = 0.05*u0.std()*torch.randn_like(u0)
            # u_p = model(u0,mu)
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
            u = data[0,0:1].to(device)
            for _ in range(100):
            
                # u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd, 
                #                     (pde_du(u,mutest[:1])[:,:,:-1,:-1]-pdemean)/pdestd,
                #                     (mutest[:1]-mumean)/mustd)*outstd + outmean) + u + pde_du(u,mutest[:1])
                # u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,(mutest-mumean)/mustd)*outstd + outmean) + u + pde_du(u,mutest)
                pdeuu = pde_du(u,mutest)
                u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,
                                   (mutest-mumean)/mustd,
                                   (pdeuu[:,:,:-1,:-1]-pdemean)/pdestd)*outstd + outmean) + u + pdeuu
                test_re.append(u.detach())

            model.train()

            test_re = torch.cat(test_re,dim=0).cpu()
            
            for testtime in [0,10,-1]:
                writer.add_figure('u_time{}'.format(testtime),
                                    add_plot(test_re[testtime,0],label[testtime,0]),
                                    i)
                writer.add_figure('v_time{}'.format(testtime),
                                    add_plot(test_re[testtime,1],label[testtime,1]),
                                    i)

            test_error = criterier(test_re[-1],label[-1])/criterier(label[-1],torch.zeros_like(label[-1]))
            test_error_u = criterier(test_re[-1,0],label[-1,0])/criterier(label[-1,0],torch.zeros_like(label[-1,0]))
            test_error_v = criterier(test_re[-1,1],label[-1,1])/criterier(label[-1,1],torch.zeros_like(label[-1,1]))

            writer.add_scalar('U rel_error', test_error_u, i)
            writer.add_scalar('V rel_error', test_error_v, i)
            if test_error < test_error_best:
                test_error_best = test_error
                torch.save(model.state_dict(), 'modelp_PDE-enrich-bigmodel.pth')
    writer.close()