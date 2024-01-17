from typing import Any
import torch
from torch.nn.functional import interpolate as interp
import numpy as np

from src.operators import diffusion1D


class padder(object):

    def __init__(self, dx):
        '''
        cls = 1: Dirichlet
        cls = 2: Neumann

        inwards normal is positive (flow out of domain)
        outwards normal is negative (flow into domain)
        '''
        self.dx = dx
    

    def pad2(self, u, bv):
        return u + bv * self.dx


    def __call__(self, u, cls, bv) -> Any:
        bcl = (cls[..., :1] == 0) * bv[...,:1] + (cls[..., :1] == 1) * self.pad2(u[...,:1],  bv[...,:1])
        bcr = (cls[..., 1:] == 0) * bv[...,1:] + (cls[..., 1:] == 1) * self.pad2(u[...,-1:], bv[...,1:])
        return torch.cat([bcl, u, bcr], dim=-1)


def denormBV(BV, cls):
    return (cls==1)*BV + (cls==0)*5*BV

bids = [0,1,2,-3,-2,-1]



def genforward(fn, dfn, ufn, pfn, dt, cdx2):
    
    def forward(u, cls, bv):
        uc = dfn(u)
        cdu = fn(uc)/cdx2*dt 
        uc1 = pfn(uc[...,1:-1] + cdu, cls, bv)
        duf1 = ufn(uc1) - u
        # pcpdedu = pfn(cpdedu, cls, bv)
        # pcpdedubl = (pcpdedu-uc)[...,[0,-1]] 
        # pcpdedu = torch.cat([pcpdedubl[...,:1], cpdedu, pcpdedubl[...,1:]], dim=-1)
        return duf1#ufn(pcpdedu)
    
    return forward



def gendata():
    torch.random.manual_seed(10)#test 10, train 0
    L = 16*torch.pi
    Nx = 2**7-2
    scale = 1/4
    Nb = 256
    dt = 0.01
    saveevery = 50
    ldt = dt*saveevery
    device=torch.device('cuda:2')
    import matplotlib.pyplot as plt
    x = torch.linspace(0, L, Nx+2)[1:-1]
    dx = x[1] - x[0]
    dx2 = dx**2

    u0 = 0# torch.zeros(size=(Nb,Nx),device=device)
    for i in range(16):
        for _  in range(16):
            u0 += (torch.rand(size=(Nb,1))-0.5)*np.sin(i/16*x[None,] + L*torch.rand(size=(Nb,1)))

    BV = torch.rand(size=(Nb,1,2),device=device)*2-1
    cls = torch.randint(0,2,size=(Nb,1,2),device=device)
    BVr = denormBV(BV, cls)

    diffuer = diffusion1D(accuracy=2, device=device)
    pad = padder(dx)
    u = u0[:,None].to(device)
    results = []
    
    for t in range(10001):
        up=pad(u, cls, BVr)
        u = u + diffuer(up)/dx2*dt
        assert torch.isnan(u).sum() == 0
        if t%saveevery == 0:
            results.append(up.detach().cpu())
    results = torch.stack(results, dim=1)
    # torch.save({'results':results, 'cls':cls, 'BV':BV}, 'data.pt')

    # fig,ax = plt.subplots(5,5)
    # for i in range(5):
    #     for j in range(5):
    #         ax[i,j].imshow(results[i*5+j,:,0], vmax=5, vmin=-5)
    #         ax[i,j].axis('off')
    # fig.tight_layout()
    # fig.savefig('test.png',dpi=150)

    downfn = lambda u: interp(u, scale_factor=scale, mode='linear', align_corners=True)
    upfn = lambda u: interp(u, scale_factor=1/scale, mode='linear', align_corners=True)
    cdx = dx/scale
    cdx2 = cdx**2
    fn = genforward(diffuer, downfn, upfn, padder(cdx), ldt, cdx2)
    pdeu = []
    for t in range(200):
        c = fn(results[:,t].to(device), cls, BVr)
        pdeu.append(c.detach().cpu())
    pdeu = torch.stack(pdeu, dim=1)
    print(' ')

    ppdeu = []
    for t in range(200):
        ppdeu.append(pad(pdeu[:,t,:,1:-1] + results[:,t,:,1:-1], cls.cpu(), BVr.cpu()))
    ppdeu = torch.stack(ppdeu, dim=1)
    err = torch.sqrt(((ppdeu - results[:,1:])**2).mean(dim=(0,2,3))/(results[:,1:]**2).mean(dim=(0,2,3)))
    plt.plot(err)
    plt.savefig('err.png',dpi=150)

    # uc = downfn(results[:,0].to(device))
    # cresults = []
    # cpad = padder(cdx)
    # for t in range(200):
    #     ucp = cpad(uc, cls, BVr)
    #     uc = diffuer(ucp) + uc
    #     cresults.append(uc.detach().cpu())

    # cresults = torch.stack(cresults, dim=1)
    # dresults = map(downfn, [results[:,i+1] for i in range(200)])
    # dresults = torch.stack(list(dresults), dim=1)
    # error = torch.sqrt(((cresults - dresults)**2).mean(dim=(0,2,3))/(dresults**2).mean(dim=(0,2,3)))
    
    # plt.plot(error)
    # plt.savefig('error.png',dpi=150)
    # fig,ax = plt.subplots(5,5)
    # for i in range(5):
    #     for j in range(5):
    #         ax[i,j].imshow(cresults[i*5+j,:,0], vmax=5, vmin=-5)
    #         ax[i,j].axis('off')
    # fig.tight_layout()
    # fig.savefig('testc.png',dpi=150)
        


if __name__ == '__main__':
    gendata()

    






