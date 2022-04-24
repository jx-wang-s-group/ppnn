import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import seaborn as sns

ID = 5
# uv = 0
# plotdata = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/burgers_new_gt.pt'
# plotdata = '/home/xinyang/storage/projects/PDE_structure/RD/Test3/Csolver{0}.pt'.format(ID)
plotdata = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/Test/training_test_BG.pt'.format(ID)
savepath = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/new/Test/p{0}_p_train/'.format(ID)
gtdata = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/burgers_new.pt'

try:
    os.mkdir(savepath)
except:
    pass

plotid = [0, 20, 40, 60, 80, 99]#100, 120, 140, 160, 180, 199]
# steps = 200
cmap = matplotlib.cm.get_cmap("rainbow").copy()
cmap.set_bad('black',1.)

def magnitude(x):
    return np.sqrt(x[0]**2 + x[1]**2)



results = torch.load(plotdata, map_location='cpu')[ID].numpy()
gt = torch.load(gtdata, map_location='cpu')[ID,1:].numpy()

plt.gca().set_aspect('equal')

for num,i in enumerate(plotid):
    fig, ax = plt.subplots(figsize=(5,5))
    p = ax.imshow(magnitude(results[i]), cmap=cmap, 
        vmin=magnitude(gt[i]).min(), vmax=magnitude(gt[i]).max())
    # p = ax.imshow(results[i,uv], cmap=cmap, vmin=gt[i,uv].min(), vmax=gt[i,uv].max())
    ax.axis('off')
    fig.tight_layout(pad=0.0)
    # fig.colorbar(p, ax=ax)
    fig.savefig(savepath+'p{0}_p_'.format(ID)+str(num)+'.png',dpi=300)
    plt.close()
