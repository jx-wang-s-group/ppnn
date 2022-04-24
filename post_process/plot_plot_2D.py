import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import seaborn as sns
import tikzplotlib as tplt
from utility.utils import mesh_convertor
mcvter = mesh_convertor((257,257), (49,49), dim=2, align_corners=True)

# ID = 5
# ID2 = 0
# fontsize=18
IDs = [1,]
gtpath = '/home/xinyang/storage/projects/PDE_structure/RD/new/RD_gt2.pt'
savename = '/home/xinyang/storage/projects/PDE_structure/RD/new/Test2/error2'
PDEresult, BBresult = [], []
for ID in IDs:
    PDEdata = '/home/xinyang/storage/projects/PDE_structure/RD/new/Test2/fPDEresult{0}.pt'.format(ID)

    BBdata = '/home/xinyang/storage/projects/PDE_structure/RD/new/Test2/noPDEresult{0}.pt'.format(ID)

    # csolver = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/Test/Csolver{0}.pt'.format(ID)

    # Csolver.append(mcvter.up(torch.load(csolver, map_location='cpu')).numpy())
    PDEresult.append(torch.load(PDEdata, map_location='cpu').numpy())
    BBresult.append(torch.load(BBdata, map_location='cpu').numpy())

# Csolver = np.stack(Csolver,axis=0)
PDEresult = np.stack(PDEresult,axis=0)
BBresult = np.stack(BBresult,axis=0)

def error(x,gt):
    return np.sqrt(((x-gt)**2).mean(axis=(0,2,3,4))/(gt**2).mean(axis=(0,2,3,4)))
    


gt = torch.load(gtpath, map_location='cpu')[IDs,11:211].numpy()
# Csolver = mcvter.up(torch.load(csolver, map_location='cpu')).numpy()
# PDEresult = torch.load(PDEdata, map_location='cpu').numpy()
# BBresult = torch.load(BBdata, map_location='cpu').numpy()


x = np.linspace(0,199,200)


# matplotlib.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(10,4))

print(gt.shape,PDEresult.shape)
ax.plot(x,error(PDEresult, gt),label='PPNN')


ax.plot(x,error(BBresult,gt),label='Black-Box NN')


# ax.plot(x,error(Csolver,gt),label='Coarse solver')

ax.set_xlabel('Evolving steps')
ax.set_ylabel('Relative error')
ax.legend()
# ax.legend(ncol=3)
# ax.set_ylim(1e-6,2e-3)
# ax.set_yscale('log')
# ax.set_xlim(-5,205)
fig.tight_layout(pad=0.1)
fig.savefig(savename+'.png',dpi=300)




tikz=tplt.get_tikz_code()
tikz=tikz.replace('addlegendentry','label')
with open(savename+'.tikz','w+') as f:
	f.write(tikz)