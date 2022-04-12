import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import seaborn as sns

torch.manual_seed(10)
plotdata = '/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/pipe_gt.pt'
# '/home/xinyang/storage/projects/PDE_structure/NS/pipetest.pt'
# '/home/xinyang/storage/projects/PDE_structure/NS/noPDE/testresult.pt'
savepath = '/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/p2_t_'
plotid = [0, 10, 42, 72, 145, 218]

cmap = matplotlib.cm.get_cmap("coolwarm").copy()
cmap.set_bad('yellow',1.)

def magnitude(x):
    return np.sqrt(x[0]**2 + x[1]**2)


results = torch.load(plotdata, map_location='cpu')[2,7::4].numpy()
# [0,,7::4]

for num,i in enumerate(plotid):
    fig, ax = plt.subplots(figsize=(2,8))
    ax.pcolormesh(magnitude(results[i+1]).T,vmin = 0, vmax = 1.45, cmap=cmap)
    ax.axis('off')
    fig.tight_layout(pad=0.0)
    fig.savefig(savepath+str(num)+'.png',dpi=300)
    plt.close()
