import torch 
import torch.nn as nn


def gen_net(layers,insize,outsize,hsize):
    l = [nn.Linear(insize,hsize),] 
    for _ in range(layers):
        l.append(lblock(hsize))
    l.append(nn.Linear(hsize, outsize))
    return nn.Sequential(*l)


class lblock(nn.Module):
    def __init__(self, hidden_size,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        return self.ln(self.net(x)) + x


class pmlp(nn.Module):
    def __init__(self, 
                 input_size = 101, 
                 hidden_size = 24, 
                 hidden_layers = 2,
                 ) -> None:
        super().__init__()
        self.pnet = gen_net(0, 1, hidden_size, hidden_size)
        # self.pdenet = gen_net(h0, input_size, hidden_size, hidden_size)
        self.u0net = gen_net(hidden_layers, input_size, hidden_size, hidden_size)
        self.convertnet = gen_net(2, 3, 1, 4)
        self.hnet = gen_net(hidden_layers, hidden_size, input_size, hidden_size)

    def forward(self, u0, p):
        return self.hnet(
                    torch.stack(
                        (self.pnet(p), 
                         self.u0net(u0)
                        ),dim=-1
                    ).mean(dim=-1))


class mlpnop(nn.Module):
    def __init__(self, 
                 input_size = 101, 
                 hidden_size = 24, 
                 p_h_layers = 0,
                 hidden_layers = 3,) -> None:
        super().__init__()
        self.u0net = gen_net(hidden_layers, input_size, hidden_size, hidden_size)
        self.hnet = gen_net(hidden_layers, hidden_size, input_size, hidden_size)

    def forward(self, u0, mu):
        return self.hnet(self.u0net(u0))

class cblock(nn.Module):
    def __init__(self,hc,ksize,feature_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hc,hc,ksize,padding=ksize//2),
            nn.ReLU(),
            nn.Conv2d(hc,hc,ksize,padding=ksize//2),
        )
        self.ln = nn.LayerNorm([hc]+feature_size)
        
    def forward(self,x):
        return self.ln(self.net(x)) + x


class cnn2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(3,12,6,stride=2,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(12,48,6,stride=2,padding=2),
            nn.ReLU(),
            cblock(48,5,[64,64]),
            cblock(48,5,[64,64]),
            cblock(48,5,[64,64]),
            nn.PixelShuffle(4),#185
            nn.Conv2d(3,2,5,padding=2),
        )
        self.cw = nn.Parameter(torch.randn(1,1,1,256)) 
        self.rw = nn.Parameter(torch.randn(1,1,256,1))

    def forward(self,u0,mu,pdeu=None):
        return self.net(torch.cat((u0,mu*self.rw@self.cw),dim=1))



class cnn2dRich(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(5,12,6,stride=2,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(12,48,6,stride=2,padding=2),
            nn.ReLU(),
            cblock(48,5,[64,64]),
            cblock(48,5,[64,64]),
            cblock(48,5,[64,64]),
            nn.PixelShuffle(4),#185
            nn.Conv2d(3,2,5,padding=2),
        )

        self.cw = nn.Parameter(torch.randn(1,1,1,256)) 
        self.rw = nn.Parameter(torch.randn(1,1,256,1))

        # self.cb = nn.Parameter(torch.randn(1,1,1,256)) 
        # self.rb = nn.Parameter(torch.randn(1,1,256,1))

    def forward(self,u0,mu,pdeu):
        return self.net(torch.cat((u0,mu*self.rw@self.cw,pdeu),dim=1))
        # return self.net(torch.cat((u0,mu*self.rw@self.cw + self.cb@self.rb,pdeu),dim=1))


class cnn2dns(nn.Module):
    def __init__(self,cmesh,mesh_size, inchannels = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(inchannels, 24,6,stride=2,padding=2),#50,200
            nn.ReLU(),
            
            nn.Conv2d(24,96,6,stride=2,padding=2),#25,100
            nn.ReLU(),
            cblock(96,7,cmesh),
            cblock(96,7,cmesh),
            cblock(96,7,cmesh),
            cblock(96,7,cmesh),
            nn.PixelShuffle(4),
            nn.Conv2d(6,3,5,padding=2),
        )
        self.cw = nn.Parameter(torch.randn(1,2,1,mesh_size[1])) 
        self.rw = nn.Parameter(torch.randn(1,2,mesh_size[0],1))

    def forward(self,u0,mu):
        return self.net(torch.cat((u0, mu*self.rw@self.cw,),dim=1))


class cnn2dNSRich(nn.Module):
    def __init__(self,cmesh,mesh_size, inchannels = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(

            nn.Conv2d(inchannels, 24,6,stride=2,padding=2),#50,200
            nn.ReLU(),
            
            nn.Conv2d(24,96,6,stride=2,padding=2),#25,100
            nn.ReLU(),
            cblock(96,7,cmesh),
            cblock(96,7,cmesh),
            cblock(96,7,cmesh),
            cblock(96,7,cmesh),
            nn.PixelShuffle(4),
            nn.Conv2d(6,3,5,padding=2),
        )
        self.cw = nn.Parameter(torch.randn(1,2,1,mesh_size[1])) 
        self.rw = nn.Parameter(torch.randn(1,2,mesh_size[0],1))


    def forward(self,u0,mu,pdeu):
        return self.net(torch.cat((u0, mu*self.rw@self.cw, pdeu),dim=1))



from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    def __init__(self, 
                 image_size=256, 
                 patch_size=32, 
                 dim=1024, 
                 depth=6, 
                 heads=16, 
                 mlp_dim=2048, 
                 channels = 5, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.cw = nn.Parameter(torch.randn(1,1,1,256)) 
        self.rw = nn.Parameter(torch.randn(1,1,256,1))

        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()
        
        self.decoder = Transformer(dim, 1, heads, dim_head, mlp_dim)

        self.out = nn.Linear(dim, 2048)

        # self.outconv = nn.Sequential(
        #     nn.Conv2d(2, 2, 5, padding=2, padding_mode='circular'),
        #     nn.ReLU(),
        #     nn.Conv2d(2, 2, 5, groups=2, padding=2, padding_mode='circular'),
        # )

    def forward(self, u0, mu, pdeu):
        u0 = torch.cat((u0,mu*self.rw@self.cw, pdeu),dim=1) 
    # def forward(self, u0, mu):
    #     u0 = torch.cat((u0,mu*self.rw@self.cw),dim=1) 

        *_, h, w, dtype = *u0.shape, u0.dtype
        
        x = self.to_patch_embedding(u0)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.encoder(x)
        # x = x.mean(dim = 1)

        x = self.to_latent(x)

        x = self.decoder(x + pe[None])

        x = self.out(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', c=2, h=8, w=8, p1=32, p2=32)


        return x
    
class ViTsmall(SimpleViT):
    def __init__(self, image_size=256, patch_size=32, dim=512, depth=6, heads=8, mlp_dim=1024, channels = 5, dim_head = 64):
        super().__init__(image_size=image_size, patch_size=patch_size, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, channels = channels, dim_head = dim_head)

from torch.nn.functional import interpolate 
interp = lambda x: interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

class Unet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 12, 6, stride=2, padding=2),#128
            nn.ReLU(),
            cblock(12, 5, [128, 128]),
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(12, 24, 6, stride=2, padding=2),
            nn.ReLU(),
            cblock(24, 5, [64, 64])
        )
        self.encoder3 = nn.Sequential(
            nn.Conv2d(24, 36, 6, stride=2, padding=2),
            nn.ReLU(),
            cblock(36, 5, [32, 32]), # 32
        )
        # self.encoder4 = nn.Sequential(
        #     nn.Conv2d(48, 64, 6, stride=2, padding=2),
        #     nn.ReLU(),
        #     cblock(64, 5, [16, 16]), # 16
        # )
        # self.encoder5 = nn.Sequential(
        #     nn.Conv2d(64, 96, 6, stride=2, padding=2),
        #     nn.ReLU(),
        #     cblock(96, 5, [8, 8]), # 8
        # )
        self.encoder = [self.encoder1, self.encoder2, self.encoder3,]

        # self.decoder1 = nn.Sequential(
        #     nn.Conv2d(192, 64, 5, stride=1, padding=2),
        #     nn.ReLU(),
        #     cblock(64, 5, [16, 16]),
        # )
        # self.decoder2 = nn.Sequential(
        #     nn.Conv2d(128, 48, 5, stride=1, padding=2),
        #     nn.ReLU(),
        #     cblock(48, 5, [32, 32]),
        # )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(72, 24, 5, stride=1, padding=2),
            nn.ReLU(),
            cblock(24, 5, [64, 64]),
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(48, 12, 5, stride=1, padding=2),
            nn.ReLU(),
            cblock(12, 5, [128, 128]),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(24, 6, 5, stride=1, padding=2),
            nn.ReLU(),
            cblock(6, 5, [256, 256]),
        )
        self.decoder = [self.decoder3, self.decoder4, self.decoder5]
        
        self.out = nn.Conv2d(6, 2, 5, padding=2)

        self.cw = nn.Parameter(torch.randn(1,1,1,256)) 
        self.rw = nn.Parameter(torch.randn(1,1,256,1))

    def forward(self, u0, mu):
        u0 = torch.cat((u0, self.rw@self.cw*mu),dim=1)
    # def forward(self, u0, mu, pdeu):
    #     u0 = torch.cat((u0, self.rw@self.cw*mu, pdeu),dim=1)
        encoded = []
        for encode in self.encoder:
            u0 = encode(u0)
            encoded.append(u0)
        for decode in self.decoder:
            inp = interp(torch.cat((u0, encoded.pop()), dim=1))
            u0 = decode(inp)
        return self.out(u0)

    
        

