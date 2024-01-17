import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import seaborn as sns
import tikzplotlib as tplt
from utility.utils import mesh_convertor

ID = 0
# ID2 = 0
fontsize=18
PDEdata = '/home/xinyang/storage/projects/PDE_structure/NS/magnetic/Test/PDEresult{0}.pt'.format(ID)
# PDEdata2 = '/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/PDEresult{0}.pt'.format(ID2)

# '/home/xinyang/storage/projects/PDE_structure/NS/pipetest.pt'
# '/home/xinyang/storage/projects/PDE_structure/NS/noPDE/testresult.pt'
BBdata = '/home/xinyang/storage/projects/PDE_structure/NS/magnetic/Test/noPDEresult{0}.pt'.format(ID)
# BBdata2 = '/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/BBresult{0}.pt'.format(ID2)
savename = '/home/xinyang/storage/projects/PDE_structure/NS/magnetic/Test/p{0}'.format(ID)
gtpath = '/home/xinyang/storage/projects/PDE_structure/NS/magnetic/pipe_gt_mag.pt'
# csolver = '/home/xinyang/storage/projects/PDE_structure/NS/magnetic/Test/pipe_csolver.pt'


# mcvter = mesh_convertor((100,400), (25,100), dim=2, align_corners=False)


def error(x,gt):
    return np.sqrt(((x-gt)**2).mean(axis=(1,2,3))/(gt**2).mean(axis=(1,2,3)))
    


gt = torch.load(gtpath, map_location='cpu')[ID][1:220]
# coarsegt = mcvter.down(gt).numpy()
gt = gt.numpy()
# gt2 = torch.load(gtpath, map_location='cpu')[ID2,7::4][1:220].numpy()
# Csolver = torch.load(csolver, map_location='cpu')[ID,1:220].numpy()
PDEresult = torch.load(PDEdata, map_location='cpu').numpy()
# PDEresult2 = torch.load(PDEdata2, map_location='cpu').numpy()
BBresult = torch.load(BBdata, map_location='cpu').numpy()
# BBresult2 = torch.load(BBdata2, map_location='cpu').numpy()



x = np.linspace(0,218,219)
xxxx = plt.gca()

matplotlib.rcParams.update({'font.size': fontsize})
fig, ax = plt.subplots(figsize=(10,4))
color = next(xxxx._get_lines.prop_cycler)['color']

ax.plot(x,error(PDEresult, gt),color = color,label='PPNN')
# ax.plot(x,error(PDEresult2,gt2),'--',color = color,alpha=0.5)


color = next(xxxx._get_lines.prop_cycler)['color']
ax.plot(x,error(BBresult,gt),color=color,label='Black-Box NN')
# ax.plot(x,error(BBresult2, gt2),'--',color = color,alpha=0.5)

# ax.plot(x,error(Csolver,coarsegt),label='Coarse solver')
# ax.plot([73,73],[0,2],'k--')
ax.set_xlabel('Evolving steps')
ax.set_ylabel('Relative error')
ax.legend(ncol=3)
ax.set_ylim(0,2)
ax.set_xlim(-5,225)
fig.tight_layout(pad=0.1)
fig.savefig(savename+'x.png',dpi=300)




# tikz=tplt.get_tikz_code()
# tikz=tikz.replace('addlegendentry','label')
# with open(savename+'.tikz','w+') as f:
# 	f.write(tikz)