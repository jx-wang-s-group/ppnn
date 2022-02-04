from typing import List
from matplotlib.pyplot import cla
from numpy import isin
import torch
from torch.nn.functional import conv1d, conv2d, pad

def attach1d1():
    pass

def attach1dp():
    pass


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
                        'Backward':self.backwardfilters,}


        
    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        '''
            # The operator
            ----
            * u: the input tensor
            * return: the output tensor
        '''
        raise NotImplementedError


class diffusion1D(operator_base1D):
    def __init__(self, accuracy = 2, scheme = 'Central',device='cpu') -> None:
        super().__init__(accuracy, device)

        assert accuracy%2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central'
        self.filters = self.schemes[scheme]
        

    def __call__(self, u:torch.Tensor) -> torch.Tensor:

        if self.accuracy == 2:
            return conv1d(u, self.filters[self.accuracy])
        elif self.accuracy == 4:
            inner = conv1d(u, self.filters[self.accuracy])
            bc = conv1d(u, self.filters[self.accuracy-2],stride=u.shape[-1]-3)
            return torch.cat((bc[:,:,0:1],inner,bc[:,:,1:]),dim=-1)

        elif self.accuracy == 6:
            inner = conv1d(u, self.filters[self.accuracy])
            # bc1 = conv1d(u, self.filters[self.accuracy-2],stride=u.shape[-1]-5)
            # bc2 = conv1d(u, self.filters[self.accuracy-2],stride=u.shape[-1]-3)
            # return torch.cat((bc2[:,:,0:1],bc1[:,:,0:1],inner,bc1[:,:,1:],bc2[:,:,1:]),dim=-1)
            return inner


class advection1d(operator_base1D):
    def __init__(self, accuracy = 2, scheme='Central',device='cpu') -> None:
        super().__init__(accuracy, device)
        
        self.filters = self.schemes[scheme]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class convection1d(operator_base1D):
    def __init__(self, accuracy = 2, scheme='Upwind',device='cpu') -> None:
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
        
        if self.accuracy == 1:
            return (u[:,:,1:-1]<=0)*conv1d(u, self.filter['forward'][self.accuracy]) + \
                (u[:,:,1:-1]>0)*conv1d(u, self.filter['backward'][self.accuracy])
        elif self.accuracy == 2:
            inner = (u[:,:,2:-2]<=0)*conv1d(u, self.filter['forward'][self.accuracy]) + \
                (u[:,:,2:-2]>0)*conv1d(u, self.filter['backward'][self.accuracy])
            # inner = (u[:,:,2:-2]<=0)*conv1d(u, self.filter['forward'][self.accuracy]) + \
            #     (u[:,:,2:-2]>0)*conv1d(u, self.filter['backward'][self.accuracy])
            # bc1 = (u[:,:,1:2]<=0)*conv1d(u, self.filter['forward'][self.accuracy-1],stride=u.shape[-1]) + \
            #     (u[:,:,1:2]>0)*conv1d(u, self.filter['backward'][self.accuracy-1],stride=u.shape[-1])
            # bc2 = (u[:,:,-2:-1]<=0)*conv1d(u, self.filter['forward'][self.accuracy-1],stride=u.shape[-1]) + \
            #     (u[:,:,1:2]>0)*conv1d(u, self.filter['backward'][self.accuracy-1],stride=u.shape[-1])
            # return torch.cat((bc1,inner,bc2),dim=-1)
            return inner
        elif self.accuracy == 3:
            '''
            only for periodic boundary condition
            '''
            inner = (u[:,:,3:-3]<=0)*conv1d(u, self.filter['forward'][self.accuracy]) + \
                (u[:,:,3:-3]>0)*conv1d(u, self.filter['backward'][self.accuracy])
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
        
        self.centralfilters_y = [None,None,
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

        self.centralfilters_x:List(torch.Tensor) = permute_y2x(self.centralfilters_y)
        self.forwardfilters_x:List(torch.Tensor) = permute_y2x(self.forwardfilters_y)
        self.backwardfilters_x:List(torch.Tensor) = permute_y2x(self.backwardfilters_y)

        self.xschemes = {'Central':self.centralfilters_x,
                        'Forward':self.forwardfilters_x,
                        'Backward':self.backwardfilters_x}
        self.yschemes = {'Central':self.centralfilters_y,
                        'Forward':self.forwardfilters_y,
                        'Backward':self.backwardfilters_y}

        self.yschemes['Upwind'] =  [(None,None),
            (torch.tensor([[[[0, -1., 1.]]]],device=self.device),
             torch.tensor([[[[-1., 1., 0.]]]],device=self.device)),
            (torch.tensor([[[[0, 0, -3/2, 2., -1/2]]]],device=self.device),
             torch.tensor([[[[1/2, -2., 3/2, 0, 0]]]],device=self.device)),
            (torch.tensor([[[[0, 0, 0, -11/6, 3, -3/2, 1/3]]]],device=self.device),
             torch.tensor([[[[-1/3, 3/2, -3., 11/6, 0, 0, 0]]]],device=self.device)), 
        ]

        self.xschemes['Upwind'] = {}
        for i,v in enumerate(self.yschemes['Upwind']):
            self.xschemes['Upwind'][i] = permute_y2x(v)

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class d2udx2_2D(operator_base2D):
    def __init__(self, scheme='Central', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        assert accuracy%2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central'
        
        self.filter  = self.xschemes[scheme][accuracy]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        return conv2d(u, self.filter)

class d2udy2_2D(operator_base2D):
    def __init__(self, scheme='Central', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        assert accuracy%2 == 0, 'diffusion operator precision must be even number'
        assert scheme == 'Central'
        self.filter  = self.yschemes[scheme][accuracy]

    def __call__(self, u:torch.Tensor) -> torch.Tensor:
        return conv2d(u, self.filter)

class dudx_2D(operator_base2D):
    def __init__(self, scheme='Upwind', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.xscheme = scheme
        self.filter = self.xschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.xscheme == 'Upwind':
            return (u[:,:,self.accuracy:-self.accuracy]<=0)*conv2d(u, self.filter[0]) +\
                 (u[:,:,self.accuracy:-self.accuracy]>0)*conv2d(u, self.filter[1])
        else:
            return conv2d(u, self.filter)

class dudy_2D(operator_base2D):
    def __init__(self, scheme='Upwind', accuracy = 2, device='cpu') -> None:
        super().__init__(accuracy, device)
        self.yscheme = scheme
        self.filter = self.yschemes[scheme][accuracy]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        if self.yscheme == 'Upwind':
            return (u[:,:,:,self.accuracy:-self.accuracy]<=0)*conv2d(u, self.filter[0]) +\
                 (u[:,:,:,self.accuracy:-self.accuracy]>0)*conv2d(u, self.filter[1])
        else:
            return conv2d(u, self.filter)
        



if __name__=='__main__':
    from math import pi, exp
    import matplotlib.pyplot as plt


    def smoother(u):
        return conv1d(u,torch.tensor([[[0.3,0.4,0.3]]])) 

    def add_plot(x,u):
        fig,ax = plt.subplots()
        ax.plot(x,u)
        ax.set_ylim([.4,2.6])
        return fig

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

    ############################ 2D Burgers ###########################
    from matplotlib import cm
    mu = 0.04
    def padbcx(uinner):
        return torch.cat((uinner[:,:,-4:-1],uinner,uinner[:,:,1:4]),dim=2)

    def padbcy(uinner):
        return torch.cat((uinner[:,:,:,-4:-1],uinner,uinner[:,:,:,1:4]),dim=3)
        

    x = torch.linspace(0,4,161)
    y = torch.linspace(0,4,161)
    x,y = torch.meshgrid(x,y)
    x = x.unsqueeze(0).unsqueeze(0)
    y = y.unsqueeze(0).unsqueeze(0)
    dt = 1e-4
    dx = 0.025
    dy = 0.025
    dx2 = dx**2
    dy2 = dy**2

    t=0
    initu = torch.sin(pi*x)+1.2
    initv = 0.2*torch.sin(pi*y) + 1.2
    u = initu
    v = initv
    
    resultu = []
    resultv = []
    resultu.append(u.detach())
    resultv.append(v.detach())

    dudx = dudx_2D('Upwind',accuracy=3)
    dudy = dudy_2D('Upwind',accuracy=3)
    # dudxc = dudx_2D('Central',accuracy=6)
    # dudyc = dudy_2D('Central',accuracy=6)
    d2udx2 = d2udx2_2D('Central',accuracy=6)
    d2udy2 = d2udy2_2D('Central',accuracy=6)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    
    def addplot(u,v):
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(x.squeeze(),y.squeeze(),torch.sqrt(u.squeeze() + v.squeeze())**2,cmap=cm.coolwarm)
        ax.set_zlim(1,4)
        # ax1=ax[1].pcolormesh(initu[0,0])
        # surf1.set_facecolor((0.1,0.1,0.9,0.5))
        return fig

    for i in range(150000):
        
        ux = padbcx(u)
        uy = padbcy(u)
        vx = padbcx(v)
        vy = padbcy(v)
        
        u = u + dt*(-u*dudx(ux)/dx - v*dudy(uy)/dy + mu*(d2udx2(ux)/dx2+d2udy2(uy)/dy2))
        v = v + dt*(-u*dudx(vx)/dx - v*dudy(vy)/dy + mu*(d2udx2(vx)/dx2+d2udy2(vy)/dy2))
        
        if i%100==0:
            writer.add_figure('velocity',addplot(u,v),i)
        resultu.append(u.detach())
        resultv.append(v.detach())
        t+=dt
        

    resultu = torch.cat(resultu,dim=0)
    resultv = torch.cat(resultv,dim=0)




