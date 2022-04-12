import os
import torch

from utility.pyof import OneStepRunOFCoarse
from utility.utils import mesh_convertor
import models

torch.manual_seed(10)
modelpath = '/home/xinyang/storage/projects/PDE_structure/NS/origin/model_pPDE.pth'
# /home/xinyang/storage/projects/PDE_structure/OpenFoam/longer/pPDE.pth
modeltype = 'cnn2dNSRich'
normpath = '/home/xinyang/storage/projects/PDE_structure/NS/origin/PDE'
savename = '/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/PDEresult1.pt'
pde = True
device = torch.device('cpu')

testlength = 219
para = torch.tensor([[[[0.375]],[[9000]]]])

u0 = torch.load('/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/pipe_gt.pt')[1,7::4][:1]

if pde:
    solver = 'icoFoam' #'/home/xinyang/OpenFOAM/xinyang-8/applications/solvers/incompressible/myicoFoam/myicoFoam'
    coarsesolver = OneStepRunOFCoarse('/home/xinyang/storage/projects/PDE_structure/NS/origin/template',
                                      '/home/xinyang/storage/projects/PDE_structure/NS/origin/tmp_test1',
                                        0.8,(25,100),
                                        para[0,0].squeeze().item(),
                                        1/para[0,1].squeeze().item(),25,
                                        solver)
    mcvter = mesh_convertor((100,400),(25,100),dim=2,align_corners=False)
    begintime = 1.6
    dt = 0.8

inmean = torch.load(os.path.join(normpath,'inmean.pt'),map_location=device)
instd = torch.load(os.path.join(normpath,'instd.pt'),map_location=device)
outmean = torch.load(os.path.join(normpath,'outmean.pt'),map_location=device)
outstd = torch.load(os.path.join(normpath,'outstd.pt'),map_location=device)
parsmean = torch.load(os.path.join(normpath,'parsmean.pt'),map_location=device)
parsstd = torch.load(os.path.join(normpath,'parsstd.pt'),map_location=device)

if pde:
    pdemean = torch.load(os.path.join(normpath,'pdemean.pt'),map_location=device)
    pdestd = torch.load(os.path.join(normpath,'pdestd.pt'),map_location=device)


model = getattr(models,modeltype)([25,100],[100,400])
model.load_state_dict(torch.load(modelpath,map_location=device))
model.eval()

result = []

for i in range(testlength):
    # u = torch.cat((u0,torch.sqrt(u0[:,0:1]**2+u0[:,1:2]**2)),dim=1)
    u = u0
    if pde:
        pdeu,error = coarsesolver(mcvter.down(u0).detach(), begintime + i*dt)
        pdeu = mcvter.up(pdeu)
        # pdeux = torch.cat((pdeu,torch.sqrt(pdeu[:,0:1]**2+pdeu[:,1:2]**2)),dim=1)
        u = model((u-inmean)/instd,(para-parsmean)/parsstd,(pdeu-pdemean)/pdestd)*outstd+outmean
        u0 = u + pdeu
    else:
        u = model((u-inmean)/instd,(para-parsmean)/parsstd)*outstd+outmean
        u0=u+u0
    result.append(u0.detach().cpu())

result = torch.cat(result,dim=0)
torch.save(result, savename)







