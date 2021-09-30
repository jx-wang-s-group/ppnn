from numpy.lib.function_base import diff
import torch
import torch.nn.functional as F
from cases.diffusion import diffusion_solver, exact_diffusion_2D_

torch.set_default_tensor_type(torch.DoubleTensor)

slvr = torch.jit.script(diffusion_solver(0.001))

x=torch.linspace(0,2,20)
x,y = torch.meshgrid(x,x)
u = exact_diffusion_2D_(x,y,0).unsqueeze(0).unsqueeze(0)

result = []
import time
start=time.time()
for i in range(5000):
    u = slvr(u)
     
    result.append(u[0])
print(time.time()-start)
torch.save(torch.cat(result,dim=0),'heat_dt')
