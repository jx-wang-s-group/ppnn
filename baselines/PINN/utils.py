import torch
import torch.nn as nn
from torch.autograd import grad
from typing import Callable, Tuple

def eq_loss(ones, 
            one1, 
            zero1, 
            zero2, 
            umean,
            ustd,
            vmean,
            vstd,
            tstd,
            xstd,
            ystd,
            batch_size, 
            length, 
            loss_fn, 
            device) -> Tuple[Callable]:
    def mygrad(loss, input, create_graph:bool=False,ones=ones):
        return grad(loss, input, 
            grad_outputs=ones,
            retain_graph=True, 
            create_graph=create_graph)[0]
    
    

    def bcloss(model:nn.Module, uin, pin):
        bc = length[1]*torch.rand((batch_size,1), device=device, requires_grad = True)
        t = length[0]*torch.rand((batch_size,1), device=device, requires_grad = True)
        zero = zero1
        zero.requires_grad = True
        one = one1
        one.requires_grad = True
        out = zero1
        
        bcx = (bc,one,bc,zero)
        bcy = (zero,bc,one,bc)
        pred = []
        for i, j in zip(bcx, bcy):
            i.requires_grad_(True)
            j.requires_grad_(True)
            pred.append(model(i, j, t, uin, pin))
        
        u = [k[:,:1] for k in pred]
        v = [k[:,1:] for k in pred]

        ux = [mygrad(ui, xi) for ui, xi in zip(u, bcx)]
        uy = [mygrad(ui, yi) for ui, yi in zip(u, bcy)]
        vx = [mygrad(vi, xi) for vi, xi in zip(v, bcx)]
        vy = [mygrad(vi, yi) for vi, yi in zip(v, bcy)]
        
        loss_u = loss_fn(u[0]-u[2], out) + loss_fn(u[1]-u[3], out)
        loss_v = loss_fn(v[0]-v[2], out) + loss_fn(v[1]-v[3], out)
        loss_ux = loss_fn(ux[0]-ux[2], out) + loss_fn(ux[1]-ux[3], out)
        loss_uy = loss_fn(uy[0]-uy[2], out) + loss_fn(uy[1]-uy[3], out) 
        loss_vx = loss_fn(vx[0]-vx[2], out) + loss_fn(vx[1]-vx[3], out)
        loss_vy = loss_fn(vy[0]-vy[2], out) + loss_fn(vy[1]-vy[3], out)

        return loss_u + loss_v + loss_ux + loss_uy + loss_vx + loss_vy



    def icloss(model:nn.Module, x, y, uin, pin, out):
        return loss_fn(model(x, y, zero1, uin, pin), out)



    def resloss(model:nn.Module, uin, pin):
        x = length[1]*torch.rand((batch_size, 1), device=device, requires_grad = True)
        y = length[2]*torch.rand((batch_size, 1), device=device, requires_grad = True)
        t = length[0]*torch.rand((batch_size, 1), device=device, requires_grad = True)
        out = zero2
        pred = model(x, y, t, uin, pin)
        upred = pred[:,:1]*ustd + umean
        vpred = pred[:,1:]*vstd + vmean
        # Compute forward pass
        
        u_t = mygrad(upred, t) * ustd/tstd
        u_x = mygrad(upred, x, create_graph=True) * ustd/xstd
        u_y = mygrad(upred, y, create_graph=True) * ustd/ystd
        u_xx = mygrad(u_x, x) / xstd
        u_yy = mygrad(u_y, y) / ystd

        v_t = mygrad(vpred, t) * vstd/tstd
        v_x = mygrad(vpred, x, create_graph=True) * vstd/xstd
        v_y = mygrad(vpred, y, create_graph=True) * vstd/ystd
        v_xx = mygrad(v_x, x) / xstd
        v_yy = mygrad(v_y, y) / ystd

        residual1 = u_t + upred * u_x + vpred * u_y - pin[:,:,0,0] * (u_xx + u_yy)
        residual2 = v_t + upred * v_x + vpred * u_y - pin[:,:,0,0] * (v_xx + v_yy)

        residual = torch.cat([residual1, residual2], dim=1)

        return loss_fn(residual, out)

    return bcloss, icloss, resloss

