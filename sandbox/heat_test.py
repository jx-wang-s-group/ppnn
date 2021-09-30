import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data import DataLoader, Dataset



class trunca_error(nn.Module):
    def __init__(self,solver):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(6,12,5,padding=2),
            nn.ReLU(),

        )
        self.mlp = nn.Sequential(
            nn.Linear(1,6),
            nn.ReLU(),
            nn.Linear(6,12),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(12,6,5,padding=2),
            nn.ReLU(),
            nn.Conv2d(6,1,5, padding=2),
        )
        self.solver = solver

    def forward(self, u, dt, phy):
        return self.mlp(dt).unsqueeze(-1).unsqueeze(-1)*self.conv(u) + phy*self.solver(u, dt.unsqueeze(-1).unsqueeze(-1)) 


data0 = torch.tensor(torch.load('heat_dt').unsqueeze(1),dtype=torch.float)
class myd(Dataset):
    def __init__(self):
        super().__init__()
        t = torch.ones([250,1])
        id = torch.randint(0,4800,[5,250])
        u10 = data0[id[0]]
        u11 = data0[id[0]+10]
        u20 = data0[id[1]] 
        u21 = data0[id[1]+5]
        u30 = data0[id[2]]
        u31 = data0[id[2]+2]
        u40 = data0[id[3]]
        u41 = data0[id[3]+1]
        u50 = data0[id[4]]
        u51 = data0[id[4]+20]     
        self.t = torch.cat([t*0.01,t*0.005, t*0.002,t*0.001,t*0.02],dim = 0)
        self.u0 = torch.cat([u10,u20,u30,u40,u50],dim=0)
        self.u1 = torch.cat([u11,u21,u31,u41,u51],dim=0)

    def __getitem__(self, index):
        return self.t[index],self.u0[index],self.u1[index]

    def __len__(self):
        return self.t.shape[0]


if __name__=="__main__":
    from cases.diffusion import diffusion_solver
    import numpy as np
    import os
    os.chdir('/home/lxy/store/dynamic/phy_structure/heat2d')
    for seed in [0,10,20,40,100,1000]:
        for phy in [True, False]:
            name = str(phy) + str(seed)
            torch.manual_seed(seed)
            epoch=100
            dataset = myd()
            loader = DataLoader(dataset=dataset,batch_size=250,shuffle=True)
            solver = diffusion_solver()
            net = trunca_error(solver)
            optimizer = torch.optim.Adam(params=net.parameters(),lr=1e-3)
            loss_fn = nn.MSELoss()
            loss_his=[]
            for i in range(epoch):
                l = 0
                for dt, u0,u1 in loader:
                    u1p = net(u0,dt,phy) + (not phy)*u0
                    loss=loss_fn(u1p,u1)
                    loss.backward()
                    optimizer.step()
                    l+=loss.item()
                loss_his.append(l/5)
                print(loss_his[-1])

            np.save(name,np.array(loss_his))
    




