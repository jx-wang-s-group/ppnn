from numpy import dtype
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.utils.tensorboard import SummaryWriter
from operators import diffusion1D, convection1d




class lblock(nn.Module):
    def __init__(self, hidden_size,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        return self.ln(self.net(x)) + x

class mymlp(nn.Module):
    def __init__(self, size,input_size = 101, output_size = 99, hidden_layers = 2):
        super().__init__()
        self.net = [nn.Linear(input_size, size),]
        for _ in range(hidden_layers):
            self.net.append(lblock(size))
        self.net.append(nn.Linear(size, output_size),)
        self.net = nn.Sequential(*self.net)

    def forward(self, u):
        # return pad(self.net(u),(1,1),'constant',0)  
        # return pad(diffusr(u)*ratio+self.net(u),(1,1),'constant',0)
        # return pad(mu*diffur(u)/dx2-u[:,:,1:-1]*convec(u)/dx)*dt+u[:,:,1:-1],(1,1),'constant',0)
        # return (mu*diffusr(u)/dx2 - 0.5*convector(u*u)/dx)*dt + self.net(u)
        return self.net(u)



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(10)
    torch.manual_seed(10)
    mu = 0.1
    device = torch.device('cuda:0')
    ratio = 20*2e-5/1e-4
    dx = 4/100
    dx2 = dx**2
    dt = 200*1e-4
    diffusr = diffusion1D(accuracy=2,device=device)
    convector = convection1d(accuracy=1,device=device)
    EPOCH = int(4e3)+1
    BATCH_SIZE = int(10)
    writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/burgers/less-PDE-seed10')
    
    plotID = [0,12,24,36,48]
    ID = torch.linspace(1,10000,50,dtype=torch.long).to(device)
    data = torch.load('burgers.pth',map_location=device,)[ID].unsqueeze(1).contiguous().to(torch.float)

    def add_plot(p,l):
        fig,ax = plt.subplots()
        ax.plot(p,label='Predicted')
        ax.plot(l,'--',label='label')
        ax.legend()
        return fig

    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0 = data[:-1]
            self.du = data[1:] - data[:-1]
            self.outmean = self.du.mean()
            self.outstd = self.du.std()
            self.du = (self.du - self.outmean)/self.outstd
        def __getitem__(self, index):
            return self.u0[index], self.du[index]
        def __len__(self):
            return self.u0.shape[0]
            
    dataset=myset()
    outmean, outstd = dataset.outmean, dataset.outstd
    print(dt/dx2*outstd.item(),'\n')
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = mymlp(size=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterier = nn.MSELoss()


    for i in range(EPOCH):
        loshis = 0
        counter= 0
        
        for u0,du in train_loader:

            u_p = model(torch.cat((u0[:,:,-1:],u0,u0[:,:,:1]),dim=2))
            
            loss = criterier(u_p,du)
            optimizer.zero_grad()
            loss.backward()
            loshis += loss.item()
            optimizer.step()
            counter += 1


            # re = []
            # u_p = u0[0:1]
            # for _ in range(4):
            #     u_p = model(u_p) + pad(diffusr(u_p)*ratio,(1,1),'constant',0)
            #     re.append(u_p)
            # loss = criterier(torch.cat(re,dim=0),du)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

        writer.add_scalar('loss', loshis/counter, i)
        
        if i%100 == 0:
            print('loss: {0:4f}\t epoch:{1:d}'.format(loshis/counter, i))

            model.eval()
            test_re = []
            u = data[0:1]
            for _ in range(49):
                
            
                # u = model(u) + pad(diffusr(u)*ratio,(1,1),'constant',0) + u
                u = model(torch.cat((u[:,:,-1:],u,u[:,:,:1]),dim=2))*outstd + outmean + u

                test_re.append(u.detach().cpu())

            model.train()


            test_re = torch.cat(test_re,dim=0)
            for j in plotID:
                writer.add_figure('test_'+str(j),add_plot(test_re[j,0],data[j+1,0].detach().cpu()),i)
            writer.add_scalar('last_T_rel_error', 
                criterier(test_re[j,0],data[j+1,0].detach().cpu())
                /criterier(data[j+1,0].detach().cpu(),torch.zeros_like(data[j+1,0].cpu())),
                 i)
    
    # for i in range(49):
    #     # inp = torch.cat((data[i:i+1,:,-1:],data[i:i+1],data[i:i+1,:,:1]),dim=2)
    #     # tmp = data[i+1:i+2]-data[i:i+1]-((mu*diffusr(inp)/dx2 - 0.5*convector(inp*inp)/dx)*dt*outstd + outmean)
    #     tmp = data[i+1:i+2]-data[i:i+1]-(pad(diffusr(data[i:i+1])*ratio,[1,1],'constant',0)) *outstd + outmean
    #     writer.add_figure('data',add_plot(tmp[0,0,:].cpu().numpy(),
    #         (data[i+1,0]-data[i,0]).detach().cpu()),i)
    #     # u = pad(diffusr(data[i:i+1])*ratio,[1,1],'constant',0)
    #     # writer.add_figure('data',add_plot(u[0,0,:].cpu().numpy(),None),i)


    writer.close()

            

            
    