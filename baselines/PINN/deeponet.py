import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.autograd import grad
from torch.utils.tensorboard import SummaryWriter
from model import deeponet
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

        xmesh = torch.linspace(0, length[1], self.Nx, device=device)
        ymesh = torch.linspace(0, length[2], self.Ny, device=device)
        xmesh, ymesh = torch.meshgrid(xmesh, ymesh, indexing='ij')
        self.xmesh, self.ymesh = xmesh.reshape(-1,1), ymesh.reshape(-1,1)
        self.t = torch.linspace(0, length[0], self.num_t, device=device).reshape(-1,1)
        self.xmean = self.xmesh.mean()
        self.ymean = self.ymesh.mean()
        self.tmean = self.t.mean()
        self.xstd = self.xmesh.std()
        self.ystd = self.ymesh.std()
        self.tstd = self.t.std()
        self.xmesh = (self.xmesh - self.xmean)/self.xstd
        self.ymesh = (self.ymesh - self.ymean)/self.ystd
        self.t = (self.t - self.tmean)/self.tstd

        self.uicin = u[:,0].to(device)
        self.uicmean = self.uicin.mean()
        self.uicstd = self.uicin.std()
        self.uicin = (self.uicin - self.uicmean)/self.uicstd

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
        self.umean = self.u.mean(dim=0, keepdim=True)
        self.ustd = self.u.std(dim=0, keepdim=True)
        self.u = (self.u - self.umean)/self.ustd
        
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
        return (x, y, t, uin, pin), ulable


    def __len__(self):
        return self.N



def main():
    device = torch.device("cuda:3")
    iterations = int(5e5)    
    batch_size = 3200
    length = (2,3.2,3.2)
    u = torch.load('/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/burgers_new.pt', map_location='cpu')
    ulabel = u[0,0].permute(1,2,0).to(device)
    p = torch.linspace(0.02,0.07,6, device=device).reshape(-1,1,1,1).repeat([36,1,1,1])
    mset = myset(u, p, length=length, batch_size=batch_size, device=device)
    del u, p
    xmean,ymean,xstd,ystd,tmean,tstd = mset.xmean, mset.ymean, mset.xstd, mset.ystd, mset.tmean, mset.tstd
    umean,ustd = mset.umean, mset.ustd
    umean, vmean = umean[:,:1].to(device), umean[:,1:].to(device)
    ustd, vstd = ustd[:,:1].to(device), ustd[:,1:].to(device)
    
    data = iter(mset)
    
    loss_fn = nn.MSELoss()



    layers = [100]*7
    model = deeponet(layers).to(device)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
            lambda i:0.9**(1/2000))
    
    def test(model:nn.Module):
        model.eval()
        x, y = mset.xmesh, mset.ymesh
        uin = mset.uicin[:1].repeat([257,1,1,1])
        pin = mset.p[:1].repeat([257,1,1,1])
        t = mset.t[-1:].repeat([257,1])
        u=[]
        for i in range(257):
            u.append((umean+ustd*model(x[i*257:(i+1)*257], y[i*257:(i+1)*257], t, uin, pin)).reshape(1,257,2).detach())
        u=torch.cat(u, dim=0)
        fig,ax = plt.subplots(1,1)
        ax.imshow(u[:,:,0].detach().cpu().numpy())
        model.train()
        return fig, u


    writer = SummaryWriter(log_dir='/home/xinyang/storage/projects/PDE_structure/baseline/deeponet/norm0')
    for i in range(iterations):
        data_in, label = next(data)
        label = label.to(device)
        loss_data = loss_fn(model(*data_in), label)
        loss = loss_data 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            writer.add_scalar('data loss', loss_data.item(), i)
            writer.add_scalar('total loss', loss.item(), i)
            fg, ut = test(model)
            writer.add_figure('last',fg,i)
            writer.add_scalar('rel_error', 
                (loss_fn(ut, ulabel)/
                loss_fn(ulabel,torch.zeros_like(ulabel))).item(), i)
            print(i, loss.item(), loss_data.item())
            torch.save(model.state_dict(), '/home/xinyang/storage/projects/PDE_structure/baseline/deeponet/modelnorm0.pt')
            
if __name__ == '__main__':
    torch.set_num_threads(6)
    main()
        