import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import seaborn as sns

ID = 2
# plotdata = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/Test/test/PDEresult{0}.2pt'.format(ID)
plotdata = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/Test/burgers_gtr.pt'
savepath = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/Test/test/p{0}_t_'.format(ID)
plotid = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 199]

cmap = matplotlib.cm.get_cmap("rainbow").copy()
cmap.set_bad('black',1.)

def magnitude(x):
    return np.sqrt(x[0]**2 + x[1]**2)


results = torch.load(plotdata, map_location='cpu')[ID].numpy()


for num,i in enumerate(plotid):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(magnitude(results[i+1]), cmap=cmap)
    ax.axis('off')
    fig.tight_layout(pad=0.0)
    fig.savefig(savepath+str(num)+'.png',dpi=300)
    plt.close()
