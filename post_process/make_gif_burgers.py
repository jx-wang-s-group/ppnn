import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FFMpegWriter
import numpy as np
import os
import seaborn as sns

ID = 1
# plotdata = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/Test2/burgers_gt_test3.pt'
plotdata = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/Test2/noPDEresult{0}.pt'.format(ID)
savepath = '/home/xinyang/storage/projects/PDE_structure/2Dburgers/known/Test2/animation/p{0}_b'.format(ID)


cmap = matplotlib.cm.get_cmap("rainbow").copy()
cmap.set_bad('black',1.)
results = torch.load(plotdata, map_location='cpu').numpy()

def magnitude(x):
    return np.sqrt(x[0]**2 + x[1]**2)




writer = FFMpegWriter(fps=20)

fig,ax = plt.subplots(figsize=(5,5))
l = ax.imshow(magnitude(results[0]),vmin=0.78,vmax=0.88, cmap=cmap)
ax.axis('off')
fig.tight_layout(pad=0.0)



with writer.saving(fig, savepath+".mp4", 200):
    for i in range(200):
        l.set_data(magnitude(results[i]))
        writer.grab_frame()


