import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FFMpegWriter
import numpy as np
import os
import seaborn as sns


ID = 0
plotdata = '/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/pipe_gt.pt'
# plotdata = '/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/PDEresult{0}.pt'.format(ID)
savepath = '/home/xinyang/storage/projects/PDE_structure/NS/origin/Test/animation/p{0}_t'.format(ID)

cmap = matplotlib.cm.get_cmap("coolwarm").copy()
cmap.set_bad('black',1.)
results = torch.load(plotdata, map_location='cpu')[ID,7::4].numpy()
print(results.shape)
def magnitude(x):
    return np.sqrt(x[0]**2 + x[1]**2)




writer = FFMpegWriter(fps=20)

fig,ax = plt.subplots(figsize=(8,2))
l = ax.imshow(magnitude(results[0]) ,vmin = 0, vmax = 1.4, cmap=cmap)
ax.axis('off')
fig.tight_layout(pad=0.0)



with writer.saving(fig, savepath+".mp4", 219):
    for i in range(219):
        l.set_data(magnitude(results[i+1]))
        writer.grab_frame()


