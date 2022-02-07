import torch
import torch.nn as nn

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
            nn.Conv2d(3,12,6,stride=2,padding=2),
            nn.ReLU(),
            nn.Conv2d(12,48,4,stride=2,padding=2),
            nn.ReLU(),
            cblock(48,5,[48,41,41]),
            cblock(48,5,[48,41,41]),
            cblock(48,5,[48,41,41]),
            nn.PixelShuffle(4),
            nn.Conv2d(3,2,5),
        )
        self.cweight = nn.Parameter(torch.rand(1,1,160,1,device=device))
        self.rweight = nn.Parameter(torch.rand(1,1,1,160,device=device))
        # self.net = nn.Sequential(
        #     nn.Conv2d(3,12,5,stride=2,padding=2),
        #     nn.ReLU(),
        #     nn.Conv2d(12,48,5,stride=2,padding=2),
        #     nn.ReLU(),
        #     cblock(48,5),
        #     cblock(48,5),
        #     cblock(48,5),
        #     nn.PixelShuffle(4),
        #     nn.Conv2d(3,2,5),
        # )

    def forward(self,u0, mu):
        return self.net(torch.cat((u0,mu*self.cweight@self.rweight),dim=1))


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    from src.operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
    from src.utility.utils import mesh_convertor, model_count

    np.random.seed(10)
    torch.manual_seed(10)
    device = torch.device('cuda:0')
    feature_size = 321
    mu = torch.linspace(0.02, 0.24, 12,dtype=torch.float)
    mutest = mu.reshape(-1, 1, 1, 1).to(device)
    # mutestpde = mutest.repeat(1,1,20)
    munet = mu.to(device).reshape(1,-1,1,1,1).repeat(67,1,1,1,1).reshape(-1,1,1,1)
    mus = munet.contiguous()
    dx = 0.1
    dx2 = dx**2
    dy,dy2=dx,dx2
    dt = 300*1e-4
    
    d2udx2 = d2udx2_2D(accuracy=2,device=device)
    d2udy2 = d2udy2_2D(accuracy=2,device=device)
    dudx = dudx_2D(accuracy=1,device=device)
    dudy = dudy_2D(accuracy=1,device=device)

    mcvter = mesh_convertor(321,41,dim=2)

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

    EPOCH = int(1e5)+1
    BATCH_SIZE = int(4*67)
    
    data:torch.Tensor = torch.load('burgers_2D_2.pth',map_location=device,).to(torch.float)
    data_u0 = data[:,:-1].reshape(-1,2,feature_size,feature_size).contiguous()
    data_du = (data[:,1:] - data[:,:-1,]).reshape(-1,2,feature_size,feature_size).contiguous()
    label = data[:,1:,].detach().cpu()

    def add_plot(p,l=None):#
        fig,ax = plt.subplots(1,2,figsize=(10,5))
        p0=ax[0].pcolormesh(p)
        fig.colorbar(p0,ax=ax[0])
        if l is not None:
            p2=ax[1].pcolormesh(l)
            fig.colorbar(p2,ax=ax[1])
        return fig

    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data_u0[:,:,:-1,:-1] - data_u0[:,:,:-1,:-1].mean())/data_u0[:,:,:-1,:-1].std()
            
            pdeu = pde_du(data_u0, mus)
            self.pdeu = pdeu[:,:,:-1,:-1].contiguous()
            self.pdeumean = self.pdeu.mean()
            self.pdeustd = self.pdeu.std()
            self.pdeu = (self.pdeu - self.pdeumean)/self.pdeustd

            self.du = (data_du - pdeu)[:,:,:-1,:-1].contiguous()
            self.outmean = self.du.mean()
            self.outstd = self.du.std()
            self.du_normd = (self.du - self.outmean)/self.outstd
            self.mumean = munet.mean()
            self.mustd = munet.std()
            self.mu = (munet - self.mumean)/self.mustd

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index], self.pdeu[index], self.mu[index]

        def __len__(self):
            return self.u0_normd.shape[0]

    dataset = myset()
    inmean, instd = data_u0[:,:,:-1].mean(), data_u0[:,:,:-1].std()
    pdemean, pdestd = dataset.pdeumean, dataset.pdeustd
    outmean, outstd = dataset.outmean, dataset.outstd
    mumean, mustd = dataset.mumean, dataset.mustd
    print(dt/dx2,'\n')
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = mycnn().to(device)
    
    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=200, cooldown=200, verbose=True, min_lr=5e-5)
    criterier = nn.MSELoss()

    test_error_best = 1

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/2D/burgers/PDE-CNN')
    for i in range(EPOCH):
        loshis = 0
        counter= 0
        
        for u0,du,pdedu,mu in train_loader:

            # u_p = model(u0, pdedu, mu)
            u_p = model(u0, mu)
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
            u = data[0,0:1]
            for _ in range(67):
            
                # u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd, 
                #                     (pde_du(u,mutestpde)[:,:,:-1]-pdemean)/pdestd,
                #                     (mutest-mumean)/mustd)*outstd + outmean) + u + pde_du(u,mutestpde)
                u = padBC_rd(model((u[:,:,:-1,:-1]-inmean)/instd,
                                     (mutest[:1]-mumean)/mustd)*outstd + outmean) + u + pde_du(u,mutest[:1])
                test_re.append(u.detach())

            model.train()

            test_re = torch.cat(test_re,dim=0).cpu()
            
            for testtime in [0,-1]:
                writer.add_figure('test_time{}'.format(testtime),
                                    add_plot(test_re[testtime,0],label[testtime,0,0]),
                                    i)

            test_error = criterier(test_re,label[:,0])/criterier(label[:,0],torch.zeros_like(label[:,0]))
            test_error_u = criterier(test_re[:,0],label[:,0,0])/criterier(label[:,0,0],torch.zeros_like(label[:,0,0]))
            test_error_v = criterier(test_re[:,1],label[:,0,1])/criterier(label[:,0,1],torch.zeros_like(label[:,0,1]))

            writer.add_scalar('U rel_error', test_error_u, i)
            writer.add_scalar('V rel_error', test_error_v, i)
            if test_error < test_error_best:
                test_error_best = test_error
                torch.save(model.state_dict(), 'modelp_PDE2D.pth')
    writer.close()