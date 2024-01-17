from data import denormBV, padder, genforward
from net import mlp
from src.operators import diffusion1D
from types import SimpleNamespace
import yaml
import sys
import torch
from torch.nn.functional import interpolate as interp
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class myset(torch.utils.data.Dataset):
    def __init__(self, u, cls, bv, pdeu=None):
        
        self.pde = pdeu is not None
        self.numT = 100
        u0 = u[:, :self.numT]
        numbatch = u0.shape[0]*u0.shape[1]
        
        du = (u[:,1:self.numT+1] - u0).reshape(numbatch, 1, -1)
        
        self.u0 = u0.reshape(numbatch, 1, -1)
        
        self.cls = cls
        self.bv = bv

        self.u0mean = self.u0.mean()
        self.u0std = self.u0.std()
        self.u0 = (self.u0 - self.u0mean)/self.u0std

        self.bvmean = self.bv.mean()
        self.bvstd = self.bv.std()
        self.bv = (self.bv - self.bvmean)/self.bvstd

        if self.pde:
            self.pdeu = pdeu[:,:self.numT].reshape(numbatch, 1, -1)
            self.du = (du - self.pdeu)[...,1:-1]
            self.pdemean = self.pdeu.mean()
            self.pdestd = self.pdeu.std()
            self.pdeu = (self.pdeu - self.pdemean)/self.pdestd
            
        else:
            self.du = du[...,1:-1]

        self.dumean = self.du.mean()
        self.dustd = self.du.std()
        self.du = (self.du - self.dumean)/self.dustd


    def __len__(self):
        return self.u0.shape[0]
    
    def __getitem__(self,idx):
        if self.pde:
            return (self.u0[idx], self.pdeu[idx]), \
                self.cls[idx//self.numT], self.bv[idx//self.numT], self.du[idx]
        else:
            return (self.u0[idx],), \
                self.cls[idx//self.numT], self.bv[idx//self.numT], self.du[idx]
        

def add_plot(prediction, label):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(prediction.cpu().numpy(), vmax=label.max(), vmin=label.min())
    p2 = ax[1].imshow(label.cpu().numpy(), vmax=label.max(), vmin=label.min())
    fig.colorbar(p2, ax=ax[1])
    fig.tight_layout()
    return fig



if __name__=='__main__':
    import os
    inputfile = sys.argv[1]
    params = SimpleNamespace(**yaml.load(open(inputfile), Loader=yaml.FullLoader))
    device = torch.device(params.device)
    data = torch.load(params.datafile, map_location=device)

    u = data['u']
    pdeu = data['pdeu'] if params.pde else None
    cls = data['cls'].float()
    bv = data['bv']
    
    dataset = myset(u, cls, bv, pdeu)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batchsize, shuffle=True, num_workers=0)

    testids = [0,1,2,3,4]
    init = u[testids, 0]
    testlabel = u[testids, 1:]
    testcls = cls[testids]
    testbv = bv[testids]
    
    writer = SummaryWriter(params.tensorboarddir)
    torch.save(dataset.u0mean, os.path.join(params.tensorboarddir,'inmean.pt'))
    torch.save(dataset.u0std, os.path.join(params.tensorboarddir,'instd.pt'))
    torch.save(dataset.dumean, os.path.join(params.tensorboarddir,'outmean.pt'))
    torch.save(dataset.dustd, os.path.join(params.tensorboarddir,'outstd.pt'))
    torch.save(dataset.bvmean, os.path.join(params.tensorboarddir,'bvmean.pt'))
    torch.save(dataset.bvstd, os.path.join(params.tensorboarddir,'bvstd.pt'))

    if params.pde:
        torch.save(dataset.pdemean, os.path.join(params.tensorboarddir,'pdemean.pt'))
        torch.save(dataset.pdestd, os.path.join(params.tensorboarddir,'pdestd.pt'))

    net = mlp(params.pde).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, cooldown=20, verbose=True, min_lr=1e-5)
    criterier = torch.nn.MSELoss()

    diffuser = diffusion1D(accuracy=2, device=device)
    dt = params.dt
    odx = params.length*torch.pi/(params.finemeshsize-1)
    cdx = odx*params.scale
    cdx2 = cdx**2
    cpad, fpad = padder(cdx), padder(odx)
    downfn = lambda u: interp(u, scale_factor=1/params.scale, mode='linear', align_corners=True)
    upfn = lambda u: interp(u, scale_factor=params.scale, mode='linear', align_corners=True)

    cfn = genforward(diffuser, downfn, upfn, cpad, dt, cdx2)


    test_error_best = 0.1

    for i in range(params.epochs+1):
        loshis = 0
        counter= 0
        
        for u0, cls, bv, label in dataloader:

            u_p = net(u0,cls, bv)
            loss = criterier(u_p, label)
            optimizer.zero_grad()
            loss.backward()
            loshis += loss.item()
            optimizer.step()
            counter += 1

        writer.add_scalar('loss', loshis/counter, i)
        print(f'epoch:{i:3d};\tloss: {loshis/counter:7g}')
        scheduler.step(loshis/counter)

        if i%100 == 0:

            net.eval()
            test_re = []
            u = init

            for _ in range(params.teststeps):
                
                if params.pde:    
                    pdeu = cfn(u, testcls, denormBV(testbv, testcls))
                    uin = ((u-dataset.u0mean)/dataset.u0std,
                           (pdeu-dataset.pdemean)/dataset.pdestd)
                else:
                    pdeu=torch.zeros_like(u)
                    uin = ((u-dataset.u0mean)/dataset.u0std,) 

                nnu = net(uin, testcls, (testbv-dataset.bvmean)/dataset.bvstd)
                nnu = nnu*dataset.dustd + dataset.dumean
                
                u = fpad(u[...,1:-1] + nnu + pdeu[...,1:-1], testcls, denormBV(testbv, testcls))

                test_re.append(u.detach())

            net.train()

            test_re = torch.stack(test_re,dim=1)
            
            
            writer.add_figure(f'p0', add_plot(test_re[0].squeeze(), testlabel[0].squeeze()), i)
            writer.add_figure(f'p1', add_plot(test_re[-1,].squeeze(), testlabel[-1,].squeeze()), i)

            test_error = criterier(test_re[:,-1],testlabel[:,-1])/criterier(testlabel[:,-1],torch.zeros_like(testlabel[:,-1]))
            
            writer.add_scalar('rel_error', test_error, i)
            # if test_error < test_error_best:
            #     test_error_best = test_error
            torch.save(net.state_dict(), params.modelsavepath)
    writer.close()



    