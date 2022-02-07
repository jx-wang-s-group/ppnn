import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from operators import diffusion1D, convection1d
from src.utility.utils import mesh_convertor, model_count
from demo1x import gen_net, padBC_p_r, padBC_p

def pde_du(u0, mu) -> torch.Tensor:
    u1 = mcvter.down(u0)[:,:,:-1]
    return mcvter.up(padBC_p_r((mu*diffusr(padBC_p(u1))/dx2-0.5*convector(padBC_p(u1*u1))/dx)*dt))

class pmlp(nn.Module):
    def __init__(self, 
                 pdim = 1, 
                 input_size = 101, 
                 hidden_size = 24, 
                 output_size = 99,
                 p_h_layers = 0,
                 h0 = 2,
                 hidden_layers = 2,
                 ) -> None:
        super().__init__()
        self.pnet = gen_net(p_h_layers, pdim, hidden_size, hidden_size)
        self.pdenet = gen_net(h0, input_size, hidden_size, hidden_size)
        self.u0net = gen_net(h0, input_size, hidden_size, hidden_size)
        self.convertnet = gen_net(2, 3, 1, 4)
        self.hnet = gen_net(hidden_layers, hidden_size, output_size, hidden_size)
        # self.smooth = nn.Conv1d(1,1,5,bias=False)

    # def forward(self, u0, pdeu, p):
    #     return self.smooth(self.hnet(
    #                 torch.stack(
    #                     (self.pnet(p), 
    #                      self.pdenet(pdeu), 
    #                      self.u0net(u0)
    #                     ),dim=-1
    #                 ).mean(dim=-1)))

    def forward(self, u0, p):
        return self.hnet(
                    torch.stack(
                        (self.pnet(p), 
                         self.u0net(u0)
                        ),dim=-1
                    ).mean(dim=-1))

if __name__ == '__main__':
        
    import matplotlib.pyplot as plt
    import numpy as np
    np.random.seed(10)
    torch.manual_seed(10)
    device = torch.device('cuda:0')

    mu = torch.linspace(0.03, 0.23, 21,dtype=torch.float)
    mutest = mu.reshape(-1, 1, 1).to(device)
    mutestpde = mutest.repeat(1,1,20)
    munet = mu.to(device).reshape(-1,1,1).repeat(1,50,1).reshape(-1,1,1)
    mus = munet.repeat(1,1,20).contiguous()
    dx = 4/20
    dx2 = dx**2
    dt = 300*1e-4
    diffusr = diffusion1D(accuracy=2,device=device)
    convector = convection1d(accuracy=1,device=device)
    mcvter = mesh_convertor(101,21)
    EPOCH = int(1e4)+1
    BATCH_SIZE = int(10*21)
    writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/Feb1/modelp_PDE-h_2_3-w_48-tanh')
    
    testID0 = torch.tensor([0,10,20],dtype=torch.long)
    testID1 = torch.tensor([0,-1],dtype=torch.long)
    ID = torch.linspace(0,15000,51,dtype=torch.long).to(device)
    data:torch.Tensor = torch.load('para_burgers.pth',map_location=device,)[:,ID].contiguous().to(torch.float)
    data_u0 = data[:,:-1].reshape(-1,1,101).contiguous()
    data_du = (data[:,1:] - data[:,:-1,]).reshape(-1,1,101).contiguous()
    label = data[:,1:,].detach().cpu()

    def add_plot(p,l=None):#
        fig,ax = plt.subplots()
        ax.plot(p,label='pdedu')
        if l is not None:
            ax.plot(l,'--',label='du')
            ax.legend()
        return fig

    class myset(torch.utils.data.Dataset):
        def __init__(self):
            self.u0_normd = (data_u0[:,:,:-1] - data_u0[:,:,:-1].mean())/data_u0[:,:,:-1].std()
            
            self.pdedu = pde_du(data_u0, mus)[:,:,:-1].contiguous()
            self.pdedumean = self.pdedu.mean()
            self.pdedustd = self.pdedu.std()
            self.pdedu = (self.pdedu - self.pdedumean)/self.pdedustd

            self.du = (data_du - pde_du(data_u0, mus))[:,:,:-1].contiguous()
            # self.du = data_du[:,:,:-1].contiguous()
            self.outmean = self.du.mean()
            self.outstd = self.du.std()
            self.du_normd = (self.du - self.outmean)/self.outstd
            self.mumean = munet.mean()
            self.mustd = munet.std()
            self.mu = (munet - self.mumean)/self.mustd

        def __getitem__(self, index):
            return self.u0_normd[index], self.du_normd[index], self.pdedu[index], self.mu[index]

        def __len__(self):
            return self.u0_normd.shape[0]

    dataset = myset()
    inmean, instd = data_u0[:,:,:-1].mean(), data_u0[:,:,:-1].std()
    pdemean, pdestd = dataset.pdedumean, dataset.pdedustd
    outmean, outstd = dataset.outmean, dataset.outstd
    mumean, mustd = dataset.mumean, dataset.mustd
    print(dt/dx2,'\n')
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = pmlp(pdim = 1, 
                 input_size = 100, 
                 hidden_size = 48, 
                 output_size = 100,
                 p_h_layers = 2,
                 h0 = 3,
                 hidden_layers = 3,).to(device)
    
    print('Model parameters: {}\n'.format(model_count(model)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=200, cooldown=200, verbose=True, min_lr=5e-5)
    criterier = nn.MSELoss()

    test_error_best = 1
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
            u = data[:,0:1]
            for _ in range(50):
            
                # u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd, 
                #                     (pde_du(u,mutestpde)[:,:,:-1]-pdemean)/pdestd,
                #                     (mutest-mumean)/mustd)*outstd + outmean) + u + pde_du(u,mutestpde)
                u = padBC_p_r(model((u[:,:,:-1]-inmean)/instd,
                                     (mutest-mumean)/mustd)*outstd + outmean) + u + pde_du(u,mutestpde)
                test_re.append(u.detach())

            model.train()

            test_re = torch.cat(test_re,dim=1).cpu()
            
            for testmu in testID0:
                for testtime in testID1:
                    writer.add_figure('test_mu{}_time{}'.format(testmu,testtime),
                                      add_plot(test_re[testmu,testtime],label[testmu,testtime]),
                                      i)
            test_error = criterier(test_re[:,-1],label[:,-1])/criterier(label[:,-1],torch.zeros_like(label[:,-1]))

            writer.add_scalar('last_T_rel_error', test_error, i)
            if test_error < test_error_best:
                test_error_best = test_error
                torch.save(model.state_dict(), 'modelp_PDE-h_2_3-w_48-tanh.pth')
    writer.close()