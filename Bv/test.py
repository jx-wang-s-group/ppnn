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
    
    device = torch.device('cuda:0')
    pde = True
    length = 16*torch.pi
    folder = '/storage/xinyang/projects/PDE_structure/bv/PDE'
    datafile = '/storage/xinyang/projects/PDE_structure/bv/bvtest'
    model = '/storage/xinyang/projects/PDE_structure/bv/model_PDE.pth'
    finemeshsize = 128
    teststeps = 200
    scale = 4
    dt = 0.5
    odx = length/(finemeshsize-1)
    cdx = odx*scale
    # params = SimpleNamespace(**yaml.load(open(inputfile), Loader=yaml.FullLoader))
    device = torch.device(device)
    data = torch.load(datafile, map_location=device)

    u = data['u']
    pdeu = data['pdeu'] if pde else None
    cls = data['cls'].float()
    bv = data['bv']
    
    u0mean = torch.load(os.path.join(folder, 'inmean.pt'), map_location=device)
    u0std = torch.load(os.path.join(folder, 'instd.pt'), map_location=device)
    dumean = torch.load(os.path.join(folder, 'outmean.pt'), map_location=device)
    dustd = torch.load(os.path.join(folder, 'outstd.pt'), map_location=device)
    bvmean = torch.load(os.path.join(folder, 'bvmean.pt'), map_location=device)
    bvstd = torch.load(os.path.join(folder, 'bvstd.pt'), map_location=device)

    if pde:
        pdemean = torch.load(os.path.join(folder, 'pdemean.pt'), map_location=device)
        pdestd = torch.load(os.path.join(folder, 'pdestd.pt'), map_location=device)


    # dataset = myset(u, cls, bv, pdeu)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batchsize, shuffle=True, num_workers=0)

   
    init = u[:, 0]
    testlabel = u[:, 1:,...,1:-1]
    testcls = cls
    testbv = bv
    
    net = mlp(pde).to(device)
    net.load_state_dict(torch.load(model, map_location=device))
    criterier = lambda x,y: torch.mean((x-y)**2, dim=(2,3))
    
    diffuser = diffusion1D(accuracy=2, device=device)
    
    cdx2 = cdx**2
    cpad, fpad = padder(cdx), padder(odx)
    downfn = lambda u: interp(u, scale_factor=1/scale, mode='linear', align_corners=True)
    upfn = lambda u: interp(u, scale_factor=scale, mode='linear', align_corners=True)

    cfn = genforward(diffuser, downfn, upfn, cpad, dt, cdx2)


    test_error_best = 0.1

    

    net.eval()
    test_re = []
    u = init

    for _ in range(teststeps):
        
        if pde:    
            pdeu = cfn(u, testcls, denormBV(testbv, testcls))
            uin = ((u-u0mean)/u0std,
                    (pdeu-pdemean)/pdestd)
        else:
            pdeu=torch.zeros_like(u)
            uin = ((u-u0mean)/u0std,) 

        nnu = net(uin, testcls, (testbv-bvmean)/bvstd)
        nnu = nnu*dustd + dumean
        
        u = fpad(u[...,1:-1] + nnu + pdeu[...,1:-1], testcls, denormBV(testbv, testcls))

        test_re.append(u.detach())


    test_re = torch.stack(test_re,dim=1)[...,1:-1]

    test_error = criterier(test_re,testlabel)/criterier(testlabel,torch.zeros_like(testlabel))
    
    torch.save(test_re.cpu(), os.path.join(folder, 'result_all.pt'))
    torch.save(test_error.cpu(), os.path.join(folder, 'error_all.pt'))
    




    