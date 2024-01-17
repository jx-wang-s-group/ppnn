def burgers2DFu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    uv = u1*u1 + v1*v1
    return -u1*dudx(ux)/dx - v1*dudy(uy)/dy + mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2) + (1-uv)*u1+uv*v1

def burgers2DFv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    uv = u1*u1 + v1*v1
    return -u1*dudx(vx)/dx - v1*dudy(vy)/dy + mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2) - uv*u1+(1-uv)*v1

def burgers2Dpu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    uv = u1*u1 + v1*v1
    return -u1*dudx(ux)/dx - v1*dudy(uy)/dy + mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2)

def burgers2Dpv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    uv = u1*u1 + v1*v1
    return -u1*dudx(vx)/dx - v1*dudy(vy)/dy + mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2)

def burgers2Ddiffuseu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    # uv = u1*u1 + v1*v1
    return mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2)

def burgers2Ddiffusev(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    # uv = u1*u1 + v1*v1
    return mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2)

def rdFu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return mu*(d2udx2(ux)/dx2 + d2udy2(uy)/dy2) + u1 - u1*u1*u1 - v1 + 0.01

def rdFv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return mu*(d2udx2(vx)/dx2 + d2udy2(vy)/dy2) + 0.25 * (u1 - v1)

def rdpu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return mu*(d2udx2(ux)/dx2 + d2udy2(uy)/dy2)

def rdpv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return mu*(d2udx2(vx)/dx2 + d2udy2(vy)/dy2)

def diffu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2)

def diffv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2)

def convecu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return -u1*dudx(ux)/dx - v1*dudy(uy)/dy

def convecv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return -u1*dudx(vx)/dx - v1*dudy(vy)/dy

def ks1d(ux4,mu,dudx,d2udx2,d4udx4,ux,dx,dx2,dx4):
    return - d2udx2(ux)/dx2 - d4udx4(ux4)/dx4 - 0.5*dudx(ux*ux)/dx

def burgers2DWu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return u1 - u1*u1*u1 - v1 + 0.01

def burgers2DWv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return 0.25 * (u1 - v1)
