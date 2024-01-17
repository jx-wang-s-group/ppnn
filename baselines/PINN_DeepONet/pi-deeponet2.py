import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from baselines.PINN_DeepONet.model import deeponet
from utils import eq_loss
import matplotlib.pyplot as plt

class myset(Dataset):
    def __init__(self, 
                 u: torch.Tensor, 
                 p: torch.Tensor,
                 length = (2, 3.2, 3.2),
                 batch_size:int=50000,
                 device:torch.device=torch.device('cuda:0')):
        self.N = u.shape[0]*u.shape[1]*u.shape[3]*u.shape[4]
        
        self.num_p = u.shape[0]
        self.num_t = u.shape[1]
        self.Nx = u.shape[3]
        self.Ny = u.shape[4]
        self.num_f = self.Nx*self.Ny
        self.num_tf = self.num_t*self.num_f
        
        self.p = p
        self.pmean = self.p.mean()
        self.pstd = self.p.std()
        self.p = (self.p - self.pmean)/self.pstd

        xmesh = torch.linspace(0, length[1], self.Nx, device=device)
        ymesh = torch.linspace(0, length[2], self.Ny, device=device)
        xmesh, ymesh = torch.meshgrid(xmesh, ymesh, indexing='ij')
        self.xmesh, self.ymesh = xmesh.reshape(-1,1), ymesh.reshape(-1,1)
        self.t = torch.linspace(0, length[0], self.num_t, device=device).reshape(-1,1)
        # self.xmean = self.xmesh.mean()
        # self.ymean = self.ymesh.mean()
        # self.tmean = self.t.mean()
        # self.xstd = self.xmesh.std()
        # self.ystd = self.ymesh.std()
        # self.tstd = self.t.std()
        # self.xmesh = (self.xmesh - self.xmean)/self.xstd
        # self.ymesh = (self.ymesh - self.ymean)/self.ystd
        # self.t = (self.t - self.tmean)/self.tstd

        self.uicin = u[:,0].to(device)
        # self.uicmean = self.uicin.mean()
        # self.uicstd = self.uicin.std()
        # self.uicin = (self.uicin - self.uicmean)/self.uicstd

        self.num_initp = u.shape[0]
        self.length = length
        self.device = device
        self.batch_size = batch_size
        
        '''
                 0
            +---------+
            |         |
          3 |         | 1
            |         |
            +---------+
                 2
        '''
        
        u = torch.permute(u,(0,1,3,4,2)) 
        self.u = u.reshape(-1, u.shape[4])
        self.uicout = u[:,0].reshape(self.num_p, self.num_f, 2).to(device)
        # self.umean = self.u.mean(dim=0, keepdim=True)
        # self.ustd = self.u.std(dim=0, keepdim=True)
        # self.u = (self.u - self.umean)/self.ustd
        
    def __index(self, i):
        idp = torch.div(i, self.num_tf, rounding_mode='floor')
        idf = i%self.num_f
        idt = torch.div((i-idp*self.num_tf), self.num_f, rounding_mode='floor')
        return idp, idt, idf


    def __getitem__(self, index):
        index = torch.randint(0, self.N, (self.batch_size,), device=self.device)
        idp, idt, idf = self.__index(index)
        uin = self.uicin[idp]
        pin = self.p[idp]
        ulable = self.u[index]

        x = self.xmesh[idf]
        y = self.ymesh[idf]
        t = self.t[idt]

        uicout = self.uicout[idp, idf]

        # data, ic, bc, res
        return ((x, y, t, uin, pin), ulable), \
               (x, y, uin, pin, uicout),\
               (uin, pin),\
               (uin, pin)


    def __len__(self):
        return self.N


def main():
    device = torch.device("cuda:6")
    iterations = int(5e5)    
    batch_size = 3200
    length = (2,3.2,3.2)
    truth = torch.load('/home/xinyang/storage/projects/PDE_structure/2Dburgers/burgers_new.pt', map_location='cpu')
    u = truth['u']
    ulabel = u[0,0].permute(1,2,0).to(device)
    p = truth['viscosity'].to(device)#torch.linspace(0.02,0.07,6, device=device).reshape(-1,1,1,1).repeat([36,1,1,1])
    mset = myset(u, p, length=length, batch_size=batch_size, device=device)
    del u, p, truth
    # xmean,ymean,xstd,ystd,tmean,tstd = mset.xmean, mset.ymean, mset.xstd, mset.ystd, mset.tmean, mset.tstd
    # umean,ustd = mset.umean, mset.ustd
    # umean, vmean = umean[:,:1].to(device), umean[:,1:].to(device)
    # ustd, vstd = ustd[:,:1].to(device), ustd[:,1:].to(device)
    # pmean, pstd = mset.pmean, mset.pstd
    
    data = iter(mset)
    
    
    loss_fn = nn.MSELoss()
    zero1 = torch.zeros([batch_size, 1,], device=device)
    zero2 = torch.zeros([batch_size, 2,], device=device)
    one1 = length[1]*torch.ones([batch_size, 1,], device=device)
    
    
    ones  = torch.ones([batch_size, 1], device=device)

    bcloss, icloss, resloss=eq_loss(ones,  one1, zero1, zero2, 
        # umean, ustd, vmean, vstd, tstd, xstd, ystd, 
        batch_size, length, loss_fn, device)

    layers = [100]*7
    model = deeponet(layers).to(device)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
            lambda i:0.9**(1/2000))
    
    # def test(model:nn.Module):
    #     model.eval()
    #     x, y = mset.xmesh, mset.ymesh
    #     uin = mset.uicin[:1].repeat([257,1,1,1])
    #     pin = mset.p[:1].repeat([257,1,1,1])
    #     t = mset.t[-1:].repeat([257,1])
    #     u=[]
    #     for i in range(257):
    #         u.append((umean+ustd*model(x[i*257:(i+1)*257], y[i*257:(i+1)*257], t, uin, pin)).reshape(1,257,2).detach())
    #     u=torch.cat(u, dim=0)
    #     fig,ax = plt.subplots(1,1)
    #     ax.imshow(u[:,:,0].detach().cpu().numpy())
    #     model.train()
    #     return fig, u


    # writer = SummaryWriter(log_dir='/home/xinyang/storage/projects/PDE_structure/baseline0/pi_deeponet/norm_relu')
    for i in range(iterations):
        (data_in, label), ic, bc, res = next(data)
        label = label.to(device)
        loss_data = loss_fn(model(*data_in), label)
        loss_ic = icloss(model, *ic)
        loss_bc = bcloss(model, *bc)
        loss_res = resloss(model, *res, )#pmean, pstd
        loss = 20*loss_data + 5*loss_ic + loss_bc + loss_res
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            # writer.add_scalar('data loss', loss_data.item(), i)
            # writer.add_scalar('ic loss', loss_ic.item(), i)
            # writer.add_scalar('bc loss', loss_bc.item(), i)
            # writer.add_scalar('equ loss', loss_res.item(), i)
            # writer.add_scalar('total loss', loss.item(), i)
            # fg, ut = test(model)
            # writer.add_figure('last',fg,i)
            # writer.add_scalar('rel_error', 
            #     (loss_fn(ut, ulabel)/
            #     loss_fn(ulabel,torch.zeros_like(ulabel))).item(), i)
            print(f'{i}, {loss.item():6g}, Data:{loss_data.item():6g}, IC:{loss_ic.item():6g}, BC:{loss_bc.item():6g}, EQ:{loss_res.item():6g}')
            torch.save(model.state_dict(), f'/home/xinyang/storage/projects/PDE_structure/rebuttal/pi_deeponet2/model-{i}.pt')
            
if __name__ == '__main__':
    main()
        