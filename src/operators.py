from typing import List
import torch
from torch.nn.functional import conv1d, conv2d, pad


class operator_base1D(object):
    '''
        # The base class of 1D finite difference operator
        ----
        * filter: the derivative operator
    '''
    def __init__(self, accuracy, device='cpu') -> None:
        super().__init__()
        self.mainfilter:torch.Tensor
        self.accuracy:int = accuracy
        self.device:torch.device = torch.device(device)
        self.centralfilters = [None,None,
            torch.tensor(
                [[[1., -2., 1.]]],device=self.device),
            None,
            torch.tensor(
                [[[-1/12, 4/3, -5/2, 4/3, -1/12]]],device=self.device),
            None,
            torch.tensor(
                [[[1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]]],device=self.device),
            None,
            torch.tensor(
                [[[-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]]],device=self.device)
        ]

        self.centralfilters_4th_derivative = [None,None,
            torch.tensor([[[1., -4., 6., -4., 1.]]],device=self.device),
            None,
            torch.tensor([[[-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6]]],device=self.device),
            None,
            torch.tensor([[[7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240]]],device=self.device),
        ]

        self.forwardfilters:List(torch.Tensor) = [None,
            torch.tensor(
                [[[-1., 1.]]],device=self.device),
            torch.tensor(
                [[[-3/2, 2., -1/2]]],device=self.device),
            torch.tensor(
                [[[-11/6, 3, -3/2, 1/3]]],device=self.device),    
        ]

        self.backwardfilters:List(torch.Tensor) = [None,
            torch.tensor(
                [[[-1., 1.]]],device=self.device),
            torch.tensor(
                [[[1/2, -2., 3/2]]],device=self.device),
            torch.tensor(
                [[[-1/3, 3/2, -3., 11/6]]],device=self.device),
        ]
        
        self.schemes = {'Central':self.centralfilters,
                        'Forward':self.forwardfilters,
                        'Backward':self.backwardfilters,
                        'Central_4th':self.centralfilters_4th_derivative,}


        
    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        '''
            # The operator
            ----
            * u: the input tensor
            * return: the output tensor
        '''
        raise NotImplementedError


class diffusion1D(operator_base1D):
    def __init__(self, scheme = 'Central', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)

        assert accuracy%2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central' or scheme == 'Central_4th', 'scheme must be one of the following: Central, Central_4th'
        self.filters = self.schemes[scheme]
        

    def __call__(self, u:torch.Tensor) -> torch.Tensor:

        return conv1d(u, self.filters[self.accuracy])



class advection1d(operator_base1D):
    def __init__(self,scheme='Central', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        
        self.filters = self.schemes[scheme]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class dudx_1D(operator_base1D):
    def __init__(self, scheme='Upwind' ,accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.schemes['Upwind'] = { 
        'forward': [None,
            torch.tensor(
                [[[0, -1., 1.]]],device=self.device),
            torch.tensor(
                [[[0, 0, -3/2, 2., -1/2]]],device=self.device),
            torch.tensor(
                [[[0, 0, 0, -11/6, 3, -3/2, 1/3]]],device=self.device),    
        ],

        'backward': [None,
            torch.tensor(
                [[[-1., 1., 0.]]],device=self.device),
            torch.tensor(
                [[[1/2, -2., 3/2, 0, 0]]],device=self.device),
            torch.tensor(
                [[[-1/3, 3/2, -3., 11/6, 0, 0, 0]]],device=self.device),
        ]}
        self.filter = self.schemes[scheme]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        '''
        only for periodic boundary condition
        '''
        inner = (u[:,:,self.accuracy:-self.accuracy]<=0)*conv1d(u, self.filter['forward'][self.accuracy]) + \
            (u[:,:,self.accuracy:-self.accuracy]>0)*conv1d(u, self.filter['backward'][self.accuracy])
        return inner
        

##################################### 2D #####################################

def permute_y2x(attr_y):
    attr_x = []
    for i in attr_y:
        if i is None:
            attr_x.append(i)
        elif isinstance(i,tuple):
            tmp=(j.permute(0,1,3,2) for j in i)
            attr_x.append(tmp)
        else:
            attr_x.append(i.permute(0,1,3,2))
    return attr_x

class operator_base2D(object):
    def __init__(self, accuracy = 2, device='cpu') -> None:
        self.accuracy = accuracy
        self.device = device

        self.centralfilters_y_1 = [None,None,
            torch.tensor([[[[-.5, 0, 0.5]]]],device=self.device),
            None,
            torch.tensor([[[[1/12, -2/3, 0, 2/3, -1/12]]]],device=self.device),
        ]
        
        self.centralfilters_y_2nd_derivative = [None,None,
            torch.tensor(
                [[[[1., -2., 1.]]]],device=self.device),
            None,
            torch.tensor(
                [[[[-1/12, 4/3, -5/2, 4/3, -1/12]]]],device=self.device),
            None,
            torch.tensor(
                [[[[1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]]]],device=self.device),
            None,
            torch.tensor(
                [[[[-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]]]],device=self.device)
        ]
        self.centralfilters_y_4th_derivative = [None,None,
            torch.tensor([[[[1., -4., 6., -4., 1.]]]],device=self.device),
            None,
            torch.tensor([[[[-1/6, 2, -13/2, 28/3, -13/2, 2, -1/6]]]],device=self.device),
            None,
            torch.tensor([[[[7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240]]]],device=self.device),
        ]
        self.forwardfilters_y:List(torch.Tensor) = [None,
            torch.tensor(
                [[[[-1., 1.]]]],device=self.device),
            torch.tensor(
                [[[[-3/2, 2., -1/2]]]],device=self.device),
            torch.tensor(
                [[[[-11/6, 3, -3/2, 1/3]]]],device=self.device),    
        ]

        self.backwardfilters_y:List(torch.Tensor) = [None,
            torch.tensor(
                [[[[-1., 1.]]]],device=self.device),
            torch.tensor(
                [[[[1/2, -2., 3/2]]]],device=self.device),
            torch.tensor(
                [[[[-1/3, 3/2, -3., 11/6]]]],device=self.device),
        ]

        self.centralfilters_x_1:List(torch.Tensor) = permute_y2x(self.centralfilters_y_1)
        self.centralfilters_x_2nd_derivative:List(torch.Tensor) = permute_y2x(self.centralfilters_y_2nd_derivative)
        self.centralfilters_x_4th_derivative:List(torch.Tensor) = permute_y2x(self.centralfilters_y_4th_derivative)
        self.forwardfilters_x:List(torch.Tensor) = permute_y2x(self.forwardfilters_y)
        self.backwardfilters_x:List(torch.Tensor) = permute_y2x(self.backwardfilters_y)

        self.xschemes = {'Central1':self.centralfilters_x_1,
                        'Central2':self.centralfilters_x_2nd_derivative,
                        'Central4':self.centralfilters_x_4th_derivative,
                        'Forward1':self.forwardfilters_x,
                        'Backward1':self.backwardfilters_x}
        self.yschemes = {'Central1':self.centralfilters_y_1,
                        'Central2':self.centralfilters_y_2nd_derivative,
                        'Central4':self.centralfilters_y_4th_derivative,
                        'Forward1':self.forwardfilters_y,
                        'Backward1':self.backwardfilters_y}

        self.yschemes['Upwind1'] =  [(None,None),
            (torch.tensor([[[[0, -1., 1.]]]],device=self.device),
             torch.tensor([[[[-1., 1., 0.]]]],device=self.device)),
            (torch.tensor([[[[0, 0, -3/2, 2., -1/2]]]],device=self.device),
             torch.tensor([[[[1/2, -2., 3/2, 0, 0]]]],device=self.device)),
            (torch.tensor([[[[0, 0, 0, -11/6, 3, -3/2, 1/3]]]],device=self.device),
             torch.tensor([[[[-1/3, 3/2, -3., 11/6, 0, 0, 0]]]],device=self.device)), 
        ]

        self.xschemes['Upwind1'] = {}
        for i,v in enumerate(self.yschemes['Upwind1']):
            self.xschemes['Upwind1'][i] = permute_y2x(v)

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class d2udx2_2D(operator_base2D):
    def __init__(self, scheme='Central2', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        assert accuracy%2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central2' or scheme == 'Central4' or scheme == 'Forward1' or scheme == 'Backward1', 'scheme must be one of the following: Central2, Central4, Forward1, Backward1'
        
        self.filter  = self.xschemes[scheme][accuracy]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        return conv2d(u, self.filter)

class d2udy2_2D(operator_base2D):
    def __init__(self, scheme='Central2', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        assert accuracy%2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central2' or scheme == 'Central4' or scheme == 'Forward1' or scheme == 'Backward1', 'scheme must be one of the following: Central2, Central4, Forward1, Backward1'
        self.filter  = self.yschemes[scheme][accuracy]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        return conv2d(u, self.filter)

class dudx_2D(operator_base2D):
    def __init__(self, scheme='Upwind1', accuracy = 1, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.xscheme = scheme
        self.filter = self.xschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.xscheme == 'Upwind1':
            return (u[:,:,self.accuracy:-self.accuracy]<=0)*conv2d(u, self.filter[0]) +\
                 (u[:,:,self.accuracy:-self.accuracy]>0)*conv2d(u, self.filter[1])
        else:
            return conv2d(u, self.filter)

class dudy_2D(operator_base2D):
    def __init__(self, scheme='Upwind1', accuracy = 1, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.yscheme = scheme
        self.filter = self.yschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.yscheme == 'Upwind1':
            return (u[:,:,:,self.accuracy:-self.accuracy]<=0)*conv2d(u, self.filter[0]) +\
                 (u[:,:,:,self.accuracy:-self.accuracy]>0)*conv2d(u, self.filter[1])
        else:
            return conv2d(u, self.filter)
        


if __name__=='__main__':
    from math import pi, exp
    import matplotlib.pyplot as plt
    import rhs


    # def smoother(u):
    #     return conv1d(u,torch.tensor([[[0.3,0.4,0.3]]])) 

    # def add_plot(x,u):
    #     fig,ax = plt.subplots()
    #     ax.plot(x,u)
    #     ax.set_ylim([.4,2.6])
    #     return fig

    # region
    ##################### test diffusion operator #####################
    # def truth(x,t):
    #     return exp(-16*pi*pi*t)*torch.sin(4*pi*x)
    
    # def dudt(x,t):
    #     return -16*pi*pi*exp(-pi*pi*t)*torch.sin(4*pi*x)

    # x = torch.linspace(0,1,101)
    # dt = 2e-4
    
    # dx2 = (x[1]-x[0])**2
    # # assert dt<dx2 , "Check CFL"
    
    
    # t=0
    # init = torch.sin(4*pi*x).reshape(1,1,-1)
    # u = init
    # result = []
    # u2 = init
    # re2= []
    # result.append(u.squeeze(1).detach())
    # diffus = diffusion1D(accuracy=2)
    # for _ in range(20):
    #     u = pad(smoother(pad(diffus(u),[1,1],'constant',0)),[1,1],'constant',0)/dx2*dt+u
    #     result.append(u.squeeze(1).detach())
    #     t+=dt
    
    
    #     u2 = pad(diffus(u2),[1,1],'constant',0)/dx2*dt+u2
    #     re2.append(u.squeeze(1).detach())

    #     plt.scatter(x,u.squeeze(),s=0.5,label='FD')
    #     plt.scatter(x,u2.squeeze(),s=0.5,label='smoothed',alpha=0.5)
    #     plt.scatter(x,truth(x,t),s=0.5,label='truth',alpha=0.5)
    #     plt.legend()
    #     plt.show()
    
    # result = torch.cat(result,dim=0)
    # re2 = torch.cat(re2,dim=0)
    # # torch.save(result,'result.pt')
    # plt.plot(x,u.squeeze(),label='FD')
    # plt.plot(x,u2.squeeze())
    # plt.plot(x,truth(x,t),'--',label='truth')
    # plt.legend()
    # plt.show()
    ##################### test advection operator #####################

    ########################## test Burgers  ##########################
    # mu = 0.23
    # # def truth(x,t):
    # #     return ur + 0.5*(1-torch.tanh((x-0.5*t)/4*0.01))

    # def padbc(uinner):
    #     return torch.cat((uinner[:,:,-4:-1],uinner,uinner[:,:,1:4]),dim=2)
    
    # x = torch.linspace(-1,3,101)
    # # x = x[:-1]
    # dt = 1e-4
    # dx = x[1]-x[0]
    # dx2 = dx**2
    # presults = []
    # for mu in [0.045,]:#torch.linspace(0.03,0.23,21):
    #     t=0
    #     init = torch.sin(pi*x).reshape(1,1,-1)+2
    #     # init = torch.exp(-4*(x-1)*(x-1)).reshape(1,1,-1)#truth(x,0).reshape(1,1,-1)
    #     # init = torch.sigmoid(-100*x).reshape(1,1,-1)
    #     u = init
    #     # plt.plot(u[0,0])
    #     # plt.show()
    #     result = []
    #     result.append(u.squeeze(1).detach())
    #     convec = convection1d(accuracy=3)
    #     diffur = diffusion1D(accuracy=6)
    #     for i in range(25000):
    #         # u = padbc((mu*diffur(u)/dx2-u[:,:,1:-1]*convec(u)/dx)*dt+u[:,:,1:-1])
    #         u = (mu*diffur(padbc(u))/dx2-0.5*convec(padbc(u*u))/dx)*dt+u
    #         result.append(u.squeeze(1).detach())
    #         t+=dt
    #         # if i%1000==0:
    #         #     writer.add_figure('u',add_plot(x,u.squeeze()),i)

    #     result = torch.cat(result,dim=0)
    #     presults.append(result)
    #     print('mu: {0:.2f} finished.'.format(mu.item()))
    # presults = torch.stack(presults)
    # # from src.utility.utils import mesh_convertor
    # # mcvter = mesh_convertor(101,21)
    # # x2 = torch.linspace(-1,3,21)
    # # dt = 1e-3
    # # dx = x2[1]-x2[0]
    # # dx2 = dx**2
    # # t=0
    # # init2 = torch.sin(pi*x2).reshape(1,1,-1)+2
    # # u2 = init2
    # # for i in range(1500):
    # #     u2 = mcvter.down(result[i*10].reshape(1,1,-1))
    # #     u2 = (mu*diffur(padbc(u2))/dx2-0.5*convec(padbc(u2*u2))/dx)*dt+u2
    # #     t+=dt

    
    # # torch.save(result,'burgers_test.pth')
    # plt.plot(x,u.squeeze(),label='FD0')
    # # plt.plot(x2,u2.squeeze(),label='FD1')
    # # plt.plot(x,truth(x,t),'--',label='truth')
    # plt.plot(x,init[0,0],'--',label='init')
    # plt.legend()
    # plt.show()
    # endregion


    # region #################### 2D Burgers ###########################
    torch.manual_seed(10)
    
    device = torch.device('cuda:0')
    # beta = 1
    para = 5#8
    repeat = 20#32
    mu = torch.linspace(0.025,0.065,para,device=device).reshape(-1,1)
    mu = mu.repeat(repeat,1)
    num_para = para*repeat
    mu = mu.reshape(-1,1,1,1)
    # mu = torch.tensor([[[[0.025]]]],device=device)

    def padbcx(uinner):
        return torch.cat((uinner[:,:,-4:-1],uinner,uinner[:,:,1:4]),dim=2)

    def padbcy(uinner):
        return torch.cat((uinner[:,:,:,-4:-1],uinner,uinner[:,:,:,1:4]),dim=3)
        

    x = torch.linspace(0,2*pi,257,device=device)
    y = torch.linspace(0,2*pi,257,device=device)
    x,y = torch.meshgrid(x,y,indexing='ij')
    x = x.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    y = y.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    dt = 1e-4
    dx = 0.0125
    dy = 0.0125
    dx2 = dx**2
    dy2 = dy**2

    t=0
    
    initu = torch.zeros_like(x)
    initv = torch.zeros_like(y)
    for k in range(-4,5):
        for l in range(-4,5):
            initu += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)
            initv += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)
    initu = (initu-initu.amin(dim=(1,2,3),keepdim=True))/(initu.amax(dim=(1,2,3),keepdim=True)-initu.amin(dim=(1,2,3),keepdim=True))  + 0.1
    initv = (initv-initv.amin(dim=(1,2,3),keepdim=True))/(initv.amax(dim=(1,2,3),keepdim=True)-initv.amin(dim=(1,2,3),keepdim=True))  + 0.1
    u = initu
    v = initv
    
    resultu = []
    resultv = []

    dudx = dudx_2D('Upwind1',accuracy=3,device=device)
    dudy = dudy_2D('Upwind1',accuracy=3,device=device)
    # dudxc = dudx_2D('Central',accuracy=6)
    # dudyc = dudy_2D('Central',accuracy=6)
    d2udx2 = d2udx2_2D('Central2',accuracy=6,device=device)
    d2udy2 = d2udy2_2D('Central2',accuracy=6,device=device)

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/2D/burgers/solver/0.02')
    
    # def addplot(u,v):
    #     fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     ax.plot_surface(x.squeeze(),y.squeeze(),torch.sqrt(u.squeeze() + v.squeeze())**2,cmap=cm.coolwarm)
    #     ax.set_zlim(1,4)
    #     return fig

    for i in range(20001):
        
        ux = padbcx(u)
        uy = padbcy(u)
        vx = padbcx(v)
        vy = padbcy(v)
        
        uv = u*u+v*v
        u = u + dt*rhs.burgers2Dpu(u,v,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2)#(-u*dudx(ux)/dx - v*dudy(uy)/dy + mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2) + beta*((1-uv)*u+uv*v))
        v = v + dt*rhs.burgers2Dpv(u,v,mu,dudx,dudy,d2udx2,d2udy2,ux,uy,dx,dy,dx2,dy2)#(-u*dudx(vx)/dx - v*dudy(vy)/dy + mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2) + beta*(-uv*u+(1-uv)*v))
        
        # if i%100==0:
        #     writer.add_figure('velocity',addplot(u,v),i)
        if i%200==0:
            resultu.append(u.detach().cpu())
            resultv.append(v.detach().cpu())
            print(i)

        t+=dt
        

    resultu = torch.cat(resultu,dim=1)
    resultv = torch.cat(resultv,dim=1)
    torch.save(torch.stack((resultu,resultv),dim=2),'/storage/xinyang/projects/PDE_structure/burgers_test.pt')
    # torch.save(resultu,'burgers_p_2Du.pth')
    # torch.save(resultv,'burgers_p_2Dv.pth')
    # endregion

    # region ################## RD Equation ##############################
    # torch.manual_seed(20)
    
    # device = torch.device('cuda:1')
    # beta = 0.1
    # para = 8
    # repeat = 16
    # num_para = para*repeat
    # beta = torch.linspace(0.2,0.8,para,device=device).reshape(-1,1)
    # beta = beta.repeat([repeat,1]).unsqueeze(-1).unsqueeze(-1)

    # def padbcx(uinner):
    #     return torch.cat((uinner[:,:,-4:-1],uinner,uinner[:,:,1:4]),dim=2)

    # def padbcy(uinner):
    #     return torch.cat((uinner[:,:,:,-4:-1],uinner,uinner[:,:,:,1:4]),dim=3)
        

    # x = torch.linspace(0,2*pi,257,device=device)
    # y = torch.linspace(0,2*pi,257,device=device)
    # x,y = torch.meshgrid(x,y,indexing='ij')
    # x = x.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    # y = y.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    # dt = 1e-5
    # dx = 0.025
    # dy = 0.025
    # dx2 = dx**2
    # dy2 = dy**2
    # t=0
    
    # # initu = torch.zeros_like(x)
    # # initv = torch.zeros_like(y)
    # # for k in range(-8,9):
    # #     for l in range(-8,9):
    # #         initu += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)
    # #         initv += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)

    # initu = torch.randn_like(x)
    # initv = torch.randn_like(y)
    # initu = (initu-initu.amin(dim=(1,2,3),keepdim=True))/(initu.amax(dim=(1,2,3),keepdim=True)-initu.amin(dim=(1,2,3),keepdim=True))  + 0.1
    # initv = (initv-initv.amin(dim=(1,2,3),keepdim=True))/(initv.amax(dim=(1,2,3),keepdim=True)-initv.amin(dim=(1,2,3),keepdim=True))  + 0.1
    # u = initu
    # v = initv
    
    # resultu = []
    # resultv = []

    # d2udx2 = d2udx2_2D('Central',accuracy=6,device=device)
    # d2udy2 = d2udy2_2D('Central',accuracy=6,device=device)

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('/home/xinyang/store/dynamic/PDE_structure/2D/RD/solver/2')

    # def addplot(u,v):
    #     fig,ax = plt.subplots(1,2,figsize=(10,5))
    #     ax[0].pcolormesh(u)
    #     ax[1].pcolormesh(v)
    #     return fig

    # for i in range(20001):
        
    #     ux = padbcx(u)
    #     uy = padbcy(u)
    #     vx = padbcx(v)
    #     vy = padbcy(v)
        
    #     u = u + dt*(d2udx2(ux)/dx2 + d2udy2(uy)/dy2 + u - u*u*u - v+0.01)
    #     v = v + dt*(d2udx2(vx)/dx2 + d2udy2(vy)/dy2 + beta*(u - v))
        
    #     if i%100==0:
    #         writer.add_figure('velocity0',addplot(u[0,0].detach().cpu(),v[0,0].detach().cpu()),i)
    #         writer.add_figure('velocity1',addplot(u[-1,0].detach().cpu(),v[-1,0].detach().cpu()),i)
    #     if i%200==0:
    #         resultu.append(u.detach().cpu())
    #         resultv.append(v.detach().cpu())
    #         print(i)

    #     t+=dt
        

    # resultu = torch.cat(resultu,dim=1)
    # resultv = torch.cat(resultv,dim=1)
    # torch.save(torch.stack((resultu,resultv),dim=2),'burgers_2D_test_un_mu0.025_uninit.pth')
    # torch.save(resultu,'burgers_p_2Du.pth')
    # torch.save(resultv,'burgers_p_2Dv.pth')
    # endregion ################## RD Equation ##############################

    # region ################# 2D Diffusion ##############################
    # torch.manual_seed(10)
    
    # device = torch.device('cuda:0')
    # para = 8
    # repeat = 32
    # mu = torch.linspace(0.02,0.09,para,device=device).reshape(-1,1,1,1)
    # mu = mu.repeat(repeat,1,1,1)
    # num_para = para*repeat

    # def padbcx(uinner):
    #     return torch.cat((uinner[:,:,-4:-1],uinner,uinner[:,:,1:4]),dim=2)

    # def padbcy(uinner):
    #     return torch.cat((uinner[:,:,:,-4:-1],uinner,uinner[:,:,:,1:4]),dim=3)
        

    # x = torch.linspace(0,2*pi,257,device=device)
    # y = torch.linspace(0,2*pi,257,device=device)
    # x,y = torch.meshgrid(x,y,indexing='ij')
    # x = x.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    # y = y.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    # dt = 1e-4
    # dx = 0.0125
    # dy = 0.0125
    # dx2 = dx**2
    # dy2 = dy**2

    # t=0
    
    # initu = torch.zeros_like(x)
    # initv = torch.zeros_like(y)
    # for k in range(-4,5):
    #     for l in range(-4,5):
    #         initu += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)
    #         initv += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)
    # initu = (initu-initu.amin(dim=(1,2,3),keepdim=True))/(initu.amax(dim=(1,2,3),keepdim=True)-initu.amin(dim=(1,2,3),keepdim=True))  + 0.1
    # initv = (initv-initv.amin(dim=(1,2,3),keepdim=True))/(initv.amax(dim=(1,2,3),keepdim=True)-initv.amin(dim=(1,2,3),keepdim=True))  + 0.1
    # u = initu
    # v = initv
    
    # resultu = []
    # resultv = []

    # dudx = dudx_2D('Upwind',accuracy=3,device=device)
    # dudy = dudy_2D('Upwind',accuracy=3,device=device)
    # d2udx2 = d2udx2_2D('Central',accuracy=6,device=device)
    # d2udy2 = d2udy2_2D('Central',accuracy=6,device=device)

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('/home/xinyang/store/dynamic/PDE_structure/2D/transfer/solver/diffusion')
    
    # def addplot(u,v):
    #     fig,ax = plt.subplots(1,2,figsize=(10,5))
    #     ax[0].pcolormesh(u[0,0])
    #     ax[1].pcolormesh(v[0,0])
    #     return fig

    # for i in range(3201):
        
    #     ux = padbcx(u)
    #     uy = padbcy(u)
    #     vx = padbcx(v)
    #     vy = padbcy(v)
        
    #     uv = u*u+v*v
    #     u = u + dt*(mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2))
    #     v = v + dt*(mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2))
        
    #     if i%100==0:
    #         writer.add_figure('velocity',addplot(u.detach().cpu(),v.detach().cpu()),i)
    #     if i%200==0:
    #         resultu.append(u.detach().cpu())
    #         resultv.append(v.detach().cpu())
    #         print(i)

    #     t+=dt
        

    # resultu = torch.cat(resultu,dim=1)
    # resultv = torch.cat(resultv,dim=1)
    # torch.save(torch.stack((resultu,resultv),dim=2),'diffusion_2D_p.pth')
    # endregion

    # region #################### 2D Convection ##############################
    
    # torch.manual_seed(10)
    
    # device = torch.device('cuda:0')
    # para = 1
    # repeat = 32 * 8
    # mu = torch.linspace(0.02,0.9,para,device=device).reshape(-1,1,1,1)
    # mu = mu.repeat(repeat,1,1,1)
    # num_para = para*repeat

    # def padbcx(uinner):
    #     return torch.cat((uinner[:,:,-4:-1],uinner,uinner[:,:,1:4]),dim=2)

    # def padbcy(uinner):
    #     return torch.cat((uinner[:,:,:,-4:-1],uinner,uinner[:,:,:,1:4]),dim=3)
        

    # x = torch.linspace(0,2*pi,257,device=device)
    # y = torch.linspace(0,2*pi,257,device=device)
    # x,y = torch.meshgrid(x,y,indexing='ij')
    # x = x.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    # y = y.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    # dt = 1e-4
    # dx = 0.0125
    # dy = 0.0125
    # dx2 = dx**2
    # dy2 = dy**2

    # t=0
    
    # initu = torch.zeros_like(x)
    # initv = torch.zeros_like(y)
    # for k in range(-4,5):
    #     for l in range(-4,5):
    #         initu += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)
    #         initv += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)
    # initu = (initu-initu.amin(dim=(1,2,3),keepdim=True))/(initu.amax(dim=(1,2,3),keepdim=True)-initu.amin(dim=(1,2,3),keepdim=True))  + 0.1
    # initv = (initv-initv.amin(dim=(1,2,3),keepdim=True))/(initv.amax(dim=(1,2,3),keepdim=True)-initv.amin(dim=(1,2,3),keepdim=True))  + 0.1
    # u = initu
    # v = initv
    
    # resultu = []
    # resultv = []

    # dudx = dudx_2D('Upwind',accuracy=3,device=device)
    # dudy = dudy_2D('Upwind',accuracy=3,device=device)
    # d2udx2 = d2udx2_2D('Central',accuracy=6,device=device)
    # d2udy2 = d2udy2_2D('Central',accuracy=6,device=device)

    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('/home/xinyang/store/dynamic/PDE_structure/2D/transfer/solver/convection')
    
    # def addplot(u,v):
    #     fig,ax = plt.subplots(1,2,figsize=(10,5))
    #     ax[0].pcolormesh(u[0,0])
    #     ax[1].pcolormesh(v[0,0])
    #     return fig

    # for i in range(3201):
        
    #     ux = padbcx(u)
    #     uy = padbcy(u)
    #     vx = padbcx(v)
    #     vy = padbcy(v)
        
    #     u = u + dt*(-u*dudx(ux)/dx - v*dudy(uy)/dy)
    #     v = v + dt*(-u*dudx(vx)/dx - v*dudy(vy)/dy)
        
    #     if i%100==0:
    #         writer.add_figure('velocity',addplot(u.detach().cpu(),v.detach().cpu()),i)
    #     if i%200==0:
    #         resultu.append(u.detach().cpu())
    #         resultv.append(v.detach().cpu())
    #         print(i)

    #     t+=dt
        

    # resultu = torch.cat(resultu,dim=1)
    # resultv = torch.cat(resultv,dim=1)
    # torch.save(torch.stack((resultu,resultv),dim=2),'convection_2D.pth')
    
    # endregion ####################

    # region ##################### 2D KS ##############################
    # torch.manual_seed(42)
    
    # device = torch.device('cuda:0')
    # para = 1
    # repeat = 1
    # mu = torch.linspace(0.02,0.9,para,device=device).reshape(-1,1,1,1)
    # mu = mu.repeat(repeat,1,1,1)
    # num_para = para*repeat

    # def padbcx(uinner):
    #     return torch.cat((uinner[:,:,-4:-1],uinner,uinner[:,:,1:4]),dim=2)

    # def padbcy(uinner):
    #     return torch.cat((uinner[:,:,:,-4:-1],uinner,uinner[:,:,:,1:4]),dim=3)
        

    # x = torch.linspace(0,8*pi,65,device=device)
    # y = torch.linspace(0,8*pi,65,device=device)
    # x,y = torch.meshgrid(x,y,indexing='ij')
    # x = x.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    # y = y.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1,1])
    # dt = 1e-3
    # dx = pi/8
    # dy = dx
    # dx2 = dx**2
    # dy2 = dy**2
    # dx4 = dx2**2
    # dy4 = dy2**2
    # print(dt/dx4)
    # t=0
    
    # initu = torch.zeros_like(x)
    
    # # for k in range(-2,3):
    # #     for l in range(-2,3):
    # #         initu += torch.randn_like(mu)*torch.sin(k*x + l*y) + torch.randn_like(mu)*torch.cos(k*x + l*y)
    # initu = torch.cos((x)/4) * (1+torch.sin((y)/4)) #+ torch.exp(-1*((x-4*pi)**2 + (y-4*pi)**2))
    # # initu = (initu-initu.amin(dim=(1,2,3),keepdim=True))/(initu.amax(dim=(1,2,3),keepdim=True)-initu.amin(dim=(1,2,3),keepdim=True))  + 0.1
    
    # u = initu
    
    # resultu = []

    # dudx = dudx_2D('Upwind1',accuracy=3,device=device)
    # dudy = dudy_2D('Upwind1',accuracy=3,device=device)
    # d2udx2 = d2udx2_2D('Central2',accuracy=6,device=device)
    # d2udy2 = d2udy2_2D('Central2',accuracy=6,device=device)
    # d4udx4 = d2udx2_2D('Central4',accuracy=4,device=device)
    # d4udy4 = d2udy2_2D('Central4',accuracy=4,device=device)

    # def rk4(u):
    #     def f(u):
    #         return - .1*d2udx2(padbcx(u))/dx2 - .1*d2udy2(padbcy(u))/dy2 - d4udx4(padbcx(u))/dx4 \
    #             - d4udy4(padbcy(u))/dy4 - d2udy2(padbcy(d2udx2(padbcx(u)))) - d2udx2(padbcx(d2udy2(padbcy(u)))) \
    #                 - 0.5*dudx(padbcx(u*u))/dx - 0.5*dudy(padbcy(u*u))/dy
        
    #     k1 = f(u)
    #     k2 = f(u + dt*k1/2)
    #     k3 = f(u + dt*k2/2)
    #     k4 = f(u + dt*k3)
    #     return u + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        

    
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/2D/transfer/solver/KS10')
    
    # def addplot(u):
    #     fig,ax = plt.subplots()
    #     p=ax.pcolormesh(u[0,0])
    #     fig.colorbar(p)
    #     return fig

    # for i in range(20001):
        
    #     # u = u + dt*(- d2udx2(padbcx(u))/dx2 - d2udy2(padbcy(u))/dy2 - d4udx4(padbcx(u))/dx4 \
    #     #         - d4udy4(padbcy(u))/dy4 - u*dudx(padbcx(u))/dx - u*dudy(padbcy(u))/dy)
    #     u = rk4(u)
        
        
    #     if i%1000==0:
    #         writer.add_figure('velocity',addplot(u.detach().cpu()),i)
    #     if i%200==0:
    #         # resultu.append(u.detach().cpu())
    #         print(i)

    #     t+=dt
        

    # # resultu = torch.cat(resultu,dim=1)
    # # torch.save(resultu,'KS_2D.pth')

    # endregion ####################

    # region ##################### 1D KS ##############################
    # torch.manual_seed(42)
    
    # device = torch.device('cuda:0')
    # repeat = 256
    
    # num_para = 1*repeat

    # def padbc(uinner):
    #     return torch.cat((uinner[:,:,-4:-1],uinner,uinner[:,:,1:4]),dim=2)

        
    # x = torch.linspace(0,8*pi,129,device=device)
    
    # x = x.unsqueeze(0).unsqueeze(0).repeat([num_para,1,1])
    # dt = 1e-4
    # dx = pi/8
    # dx2 = dx**2
    # dx4 = dx2**2
    # print(dt/dx4)
    # t=0
    
    # phi0 = torch.randn((num_para,1,1),device=device)
    # phi1 = torch.randn((num_para,1,1),device=device)
    # initu = torch.cos(x/16 + phi0)*(1+torch.sin(x/16 + phi1)) + torch.exp(-25*(x/8/pi-0.5)**2)
    # # initu = (initu-initu.amin(dim=(1,2,3),keepdim=True))/(initu.amax(dim=(1,2,3),keepdim=True)-initu.amin(dim=(1,2,3),keepdim=True))  + 0.1
    
    # u = initu
    
    # resultu = []

    # dudx = dudx_1D('Upwind',accuracy=3,device=device)
    # d2udx2 = diffusion1D('Central',accuracy=6,device=device)
    # d4udx4 = diffusion1D('Central_4th',accuracy=4,device=device)
    

    # def rk4(u):
    #     def f(u):
    #         return - d2udx2(padbc(u))/dx2 - d4udx4(padbc(u))/dx4 - 0.5*dudx(padbc(u*u))/dx + 0.1*torch.exp(-(x - 4*pi)**2/2)
        
    #     k1 = f(u)
    #     k2 = f(u + dt*k1/2)
    #     k3 = f(u + dt*k2/2)
    #     k4 = f(u + dt*k3)
    #     return u + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    
    
    # def addplot(u):
    #     fig,ax = plt.subplots()
    #     p=ax.pcolormesh(u)
    #     fig.colorbar(p)
    #     return fig

    # for i in range(200001):
        
    #     u = rk4(u)
        
    #     if i%500==0:
    #         resultu.append(u.detach())
    #         print(i)

    #     t+=dt
        

    # resultu = torch.stack(resultu,dim=1)
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter('/home/lxy/store/projects/dynamic/PDE_structure/1D/KS/solver/11')
    # for i in range(256):
    #     writer.add_figure('velocity',addplot(resultu[i].cpu().squeeze()),i)
    # writer.close()
    # torch.save(resultu,'KS_1D.pth')
    # plt.pcolormesh(resultu[0,:,0].cpu())
    # plt.show()
    # endregion ####################