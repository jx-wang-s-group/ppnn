import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import grad
from model import pinn


class myset(Dataset):
    def __init__(self, 
                 u: torch.Tensor, 
                 p: torch.Tensor,
                 length = (2, 3.2, 3.2),
                 num_points:int=216*101*257*257,
                 batch_size:int=50000,
                 device:torch.device=torch.device('cuda:0')):
        self.N = num_points
        
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

        self.uicin = u[:,0].to(device)

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
        
    def __index(self, i):
        idp = i//self.num_tf
        idf = i%self.num_f
        idt = (i-idp*self.num_tf)//self.num_f
        return idp, idt, idf


    def __getitem__(self, index):
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
    device = torch.device("cuda:0")
    epoch = 3000    
    batch_size = 3200
    length = (2,3.2,3.2)
    u = torch.load('/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/burgers_new.pt', map_location='cpu')
    p = torch.linspace(0.02,0.07,6, device=device).reshape(-1,1,1,1).repeat([36,1,1,1])
    layers = [51]+[100]*4+[2]
    model = pinn(layers).to(device)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, 
            lambda i:lr*0.9**(i/2000))
    dataloader = DataLoader(myset(u, p, batch_size=batch_size), 
        batch_size=batch_size, shuffle=True)#  num_workers=6,
    del u, p
    loss_fn = nn.MSELoss()
    zero1 = torch.zeros([batch_size, 1,], device=device)
    zero2 = torch.zeros([batch_size, 2,], device=device)
    one1 = length[1]*torch.ones([batch_size, 1,], device=device)
    
    
    ones  = torch.ones([batch_size, 1], device=device)
    def mygrad(loss, input, create_graph:bool=False,ones=ones):
        return grad(loss, input, 
            grad_outputs=ones,
            retain_graph=True, 
            create_graph=create_graph)[0]
    
    

    def bcloss(model:nn.Module, uin, pin):
        bc = length[1]*torch.rand((batch_size,1), device=device, requires_grad = True)
        t = length[0]*torch.rand((batch_size,1), device=device, requires_grad = True)
        zero = zero1
        zero.requires_grad = True
        one = one1
        one.requires_grad = True
        out = zero1
        
        bcx = (bc,one,bc,zero)
        bcy = (zero,bc,one,bc)
        pred = []
        for i, j in zip(bcx, bcy):
            i.requires_grad_(True)
            j.requires_grad_(True)
            pred.append(model(i, j, t, uin, pin))
        
        u = [k[:,:1] for k in pred]
        v = [k[:,1:] for k in pred]

        ux = [mygrad(ui, xi) for ui, xi in zip(u, bcx)]
        uy = [mygrad(ui, yi) for ui, yi in zip(u, bcy)]
        vx = [mygrad(vi, xi) for vi, xi in zip(v, bcx)]
        vy = [mygrad(vi, yi) for vi, yi in zip(v, bcy)]
        
        loss_u = loss_fn(u[0]-u[2], out) + loss_fn(u[1]-u[3], out)
        loss_v = loss_fn(v[0]-v[2], out) + loss_fn(v[1]-v[3], out)
        loss_ux = loss_fn(ux[0]-ux[2], out) + loss_fn(ux[1]-ux[3], out)
        loss_uy = loss_fn(uy[0]-uy[2], out) + loss_fn(uy[1]-uy[3], out) 
        loss_vx = loss_fn(vx[0]-vx[2], out) + loss_fn(vx[1]-vx[3], out)
        loss_vy = loss_fn(vy[0]-vy[2], out) + loss_fn(vy[1]-vy[3], out)

        return loss_u + loss_v + loss_ux + loss_uy + loss_vx + loss_vy



    def icloss(model:nn.Module, x, y, uin, pin, out):
        return loss_fn(model(x, y, zero1, uin, pin), out)



    def resloss(model:nn.Module, uin, pin):
        x = length[1]*torch.rand((batch_size, 1), device=device, requires_grad = True)
        y = length[2]*torch.rand((batch_size, 1), device=device, requires_grad = True)
        t = length[0]*torch.rand((batch_size, 1), device=device, requires_grad = True)
        out = zero2
        pred = model(x, y, t, uin, pin)
        upred = pred[:,:1]
        vpred = pred[:,1:]
        # Compute forward pass
        
        u_t = mygrad(upred, t)
        u_x = mygrad(upred, x, create_graph=True)
        u_y = mygrad(upred, y, create_graph=True)
        u_xx = mygrad(u_x, x)
        u_yy = mygrad(u_y, y)

        v_t = mygrad(vpred, t)
        v_x = mygrad(vpred, x, create_graph=True)
        v_y = mygrad(vpred, y, create_graph=True)
        v_xx = mygrad(v_x, x)
        v_yy = mygrad(v_y, y)

        residual1 = u_t + upred * u_x + vpred * u_y - pin[:,:,0,0] * (u_xx + u_yy)
        residual2 = v_t + upred * v_x + vpred * u_y - pin[:,:,0,0] * (v_xx + v_yy)

        residual = torch.cat([residual1, residual2], dim=1)

        return loss_fn(residual, out)    

    iter = 0
    for _ in range(epoch):
        for (data_in, label), ic, bc, res in dataloader:
            label = label.to(device)
            loss_data = loss_fn(model(*data_in), label)
            loss_ic = icloss(model, *ic)
            loss_bc = bcloss(model, *bc)
            loss_res = resloss(model, *res)
            loss = loss_data + loss_ic + loss_bc + loss_res
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if iter % 100 == 0:
                print(iter, loss.item(), loss_data.item(), loss_ic.item(), loss_bc.item(), loss_res.item())
            iter += 1
    torch.save(model.state_dict(), '/home/xinyang/storage/projects/PDE_structure/baseline/PINN/modelpinn.pt')
            
if __name__ == '__main__':
    main()
        