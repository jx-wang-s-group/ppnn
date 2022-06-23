"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which uses a recurrent structure to propagates in time.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np

from utilities3 import LpLoss, count_params


from Adam import Adam
from fmodel import *

torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda:1')


savepath='/home/xinyang/storage/projects/PDE_structure/baseline0/FNO'


ntrain = 20

modes = 12
width = 20

batch_size = 250
batch_size2 = batch_size

epochs = 300
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)



u = torch.load('/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/burgers_new.pt', map_location='cpu')
u = u[:,:,:,:-1,:-1]


class myset(torch.utils.data.Dataset):
    def __init__(self, u):
        self.num_t = u.shape[1]-1
        self.num_p = u.shape[0]
        self.u = u[:,1:].reshape(-1,2,256,256).permute(0,2,3,1)
        self.umean, self.ustd = self.u.mean(dim=(0,1,2),keepdim=True), self.u.std(dim=(0,1,2),keepdim=True)
        self.u = (self.u - self.umean)/self.ustd
        self.ic = u[:,0].permute(0,2,3,1).to(device)
        self.icmean, self.icstd = self.ic.mean(dim=(0,1,2),keepdim=True), self.ic.std(dim=(0,1,2),keepdim=True)
        self.ic = (self.ic - self.icmean)/self.icstd
        self.p = torch.linspace(0.02,0.07,6, device=device).reshape(-1,1,1,1).repeat([36,1,1,1])
        self.pmean, self.pstd = self.p.mean(), self.p.std()
        self.p = (self.p - self.pmean)/self.pstd
        self.t = torch.linspace(0,1,101, device=device)[1:].reshape(-1,1,1,1)
        self.tmean, self.tstd = self.t.mean(), self.t.std()
        self.t = (self.t - self.tmean)/self.tstd
        torch.save(self.umean, savepath+'/umean.pt')
        torch.save(self.ustd, savepath+'/ustd.pt')
        torch.save(self.icmean, savepath+'/icmean.pt')
        torch.save(self.icstd, savepath+'/icstd.pt')
        torch.save(self.pmean, savepath+'/pmean.pt')
        torch.save(self.pstd, savepath+'/pstd.pt')
        torch.save(self.tmean, savepath+'/tmean.pt')
        torch.save(self.tstd, savepath+'/tstd.pt')
        self.umean = self.umean.to(device)
        self.ustd = self.ustd.to(device)

    def __getitem__(self, index):
        return (self.ic[index//self.num_t],
                self.p[index//self.num_t], \
                self.t[index%self.num_t],), \
                self.u[index]

    def __len__(self):
        return self.u.shape[0]

datset = myset(u)
train_loader = torch.utils.data.DataLoader(datset, batch_size=batch_size, shuffle=True, num_workers=0)


model = fnot(modes, modes, width).to(device)


print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
loss_fn = torch.nn.MSELoss()


testt = (torch.tensor([1.], device=device).reshape(1,1,1,1) - datset.tmean)/datset.tstd
testp = (torch.tensor([0.02], device=device).reshape(1,1,1,1) - datset.pmean)/datset.pstd
testic = (u[:1,0].permute(0,2,3,1).to(device) - datset.icmean)/datset.icstd
gt = u[:1,-1].to(device)
del u

def test(model):
    model.eval()
    with torch.no_grad():
        u = model(testic, testp, testt)
    u = (u*datset.ustd + datset.umean).permute(0,3,1,2)
    model.train()
    fg, ax = plt.subplots()
    ax.imshow(u[0,0,:,:].detach().cpu().numpy(), cmap='jet')
    return fg, u


writer = SummaryWriter(savepath)
for ep in range(epochs):
    model.train()
    avgl = 0
    i = 0
    for xx, yy in train_loader:
        i += 1
        yy = yy.to(device)

        im = model(*xx)
        loss = myloss(im, yy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avgl += loss.item()
    

    scheduler.step()
    writer.add_scalar('data loss', avgl/i, ep)
    print(ep, avgl/i)
    if ep%10==0:
        fig, u = test(model)
        writer.add_figure('last', fig, ep)
        writer.add_scalar('rel_error', (loss_fn(u, gt)/loss_fn(gt, torch.zeros_like(gt))).item(), ep)

torch.save(model, savepath+'/model.pt')
