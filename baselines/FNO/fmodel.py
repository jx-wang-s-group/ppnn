import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class f_layer(nn.Module):
    def __init__(self, width, modes1, modes2):
        super(f_layer, self).__init__()
        self.conv = SpectralConv2d_fast(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
    
    def forward(self, x):
        return F.gelu(self.conv(x) + self.w(x))


class FNO(nn.Module):
    def __init__(self, modes1, modes2, width, inchannels=10, outchannels=1):
        super(FNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(inchannels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.flayers = nn.Sequential(
            f_layer(self.width, self.modes1, self.modes2),
            f_layer(self.width, self.modes1, self.modes2),
            f_layer(self.width, self.modes1, self.modes2),
            f_layer(self.width, self.modes1, self.modes2),
        )

        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, outchannels)


    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        x = self.flayers(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class fnot(FNO):
    def __init__(self, modes1, modes2, width, inchannels=4, outchannels=2):
        '''
        input: the solution of the IC + time + parameter
        output: the solution of the next timestep
        output shape: (batchsize, x=256, y=256, c=2)
        '''
        super(fnot, self).__init__(modes1, modes2, width, inchannels, outchannels)
        self.cw = nn.Parameter(torch.randn(1,1,1,256)) 
        self.rw = nn.Parameter(torch.randn(1,1,256,1))
        self.cwt = nn.Parameter(torch.randn(1,1,1,256)) 
        self.rwt = nn.Parameter(torch.randn(1,1,256,1))
        
        gridx = torch.linspace(0, 1, 256, dtype=torch.float)
        grid = torch.meshgrid(gridx, gridx, indexing='ij')
        grid = torch.stack(grid, dim=-1).unsqueeze(0)
        self.grid = grid.repeat([1,1,1,1])


    def forward(self, u, mu, t):
        mu = mu*self.rw@self.cw
        t = t*self.rwt@self.cwt
        x = torch.cat((u, mu.permute(0,2,3,1), t.permute(0,2,3,1)), dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        
        x = self.flayers(x)

        x = x.permute(0, 2, 3, 1)
        return self.fc2(F.gelu(self.fc1(x)))
