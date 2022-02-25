def burgers2Du(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return -u1*dudx(ux)/dx - v1*dudy(uy)/dy + mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2)

def burgers2Dv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return -u1*dudx(vx)/dx - v1*dudy(vy)/dy + mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2)

def rd0u(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return d2udx2(ux)/dx2 + d2udy2(uy)/dy2 + u1 - u1*u1*u1 - v1+0.01

def rd0v(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return d2udx2(vx)/dx2 + d2udy2(vy)/dy2 + mu*(u1 - v1)

def rd0unu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return d2udx2(ux)/dx2 + d2udy2(uy)/dy2 #+ u1 - u1*u1*u1 - v1+0.01

def rd0unv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return d2udx2(vx)/dx2 + d2udy2(vy)/dy2 #+ mu*(u1 - v1)

def diffu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2)

def diffv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2)

def convecu(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2):
    return -u1*dudx(ux)/dx - v1*dudy(uy)/dy

def convecv(u1,v1,mu,dudx,dudy,d2udx2,d2udy2,vx,vy,dx,dy,dx2,dy2):
    return -u1*dudx(vx)/dx - v1*dudy(vy)/dy