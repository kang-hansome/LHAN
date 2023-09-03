import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from basicsr.archs.arch_util import default_init_weights, make_layer, Upsample
from basicsr.utils.registry import ARCH_REGISTRY
class ESA(nn.Module):
    def __init__(self, num_feat=64) -> None:
        super().__init__()
        num_grow_ch     = num_feat // 4
        self.conv1      = nn.Conv2d(num_feat, num_grow_ch, kernel_size=1)
        self.s_conv     = nn.Conv2d(num_grow_ch, num_grow_ch, kernel_size=3, stride=2, padding=0)
        self.pool       = nn.MaxPool2d(kernel_size=7, stride=3)
        self.g_conv     = nn.Conv2d(num_grow_ch, num_grow_ch, kernel_size=3, padding=1)
        self.conv2      = nn.Conv2d(num_grow_ch, num_feat, kernel_size=1)
        self.sigmod     = nn.Sigmoid()
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.g_conv(self.pool(self.s_conv(x1)))
        x3 = F.interpolate(x2, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        x4 = self.conv2(x1 + x3)
        k = self.sigmod(x4)
        return x * k
class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class GMSA(nn.Module):
    def __init__(self, channels, shifts=4, window_sizes=[4, 8, 12], calc_attn=True):
        super(GMSA, self).__init__()    
        self.channels = channels
        self.shifts   = shifts
        self.window_sizes = window_sizes
        self.calc_attn = calc_attn

        if self.calc_attn:
            self.split_chns  = [channels*2//3, channels*2//3, channels*2//3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels*2, kernel_size=1), 
                nn.BatchNorm2d(self.channels*2)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        else:
            self.split_chns  = [channels//3, channels//3,channels//3]
            self.project_inp = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=1), 
                nn.BatchNorm2d(self.channels)
            )
            self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, prev_atns = None):
        b,c,h,w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []
        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c', 
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1)) 
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)            
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize//2, -wsize//2), dims=(2,3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c', 
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)', 
                    h=h//wsize, w=w//wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize//2, wsize//2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)            
            y = self.project_out(y)
            return y, prev_atns
    
class LHAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, shifts=0, window_sizes=[4, 8, 12], shared_depth=1):
        super(LHAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shared_depth = shared_depth

        modules_lfe = {}
        modules_gmsa = {}
        # modules_lfe['lfe_0'] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
        modules_lfe['lfe_0'] = nn.Sequential(
            nn.Conv2d(inp_channels, inp_channels * exp_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(inp_channels * exp_ratio, inp_channels * exp_ratio, 3, 1, 1, groups=inp_channels * exp_ratio),
            nn.Conv2d(inp_channels * exp_ratio, out_channels, 1)
        )
        modules_gmsa['gmsa_0'] = GMSA(channels=inp_channels, shifts=shifts, window_sizes=window_sizes, calc_attn=True)
        for i in range(shared_depth):
            # modules_lfe['lfe_{}'.format(i+1)] = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio)
            modules_lfe['lfe_{}'.format(i+1)] = nn.Sequential(
            nn.Conv2d(inp_channels, inp_channels * exp_ratio, 1),
            nn.Conv2d(inp_channels * exp_ratio, inp_channels * exp_ratio, 3, 1, 1, groups=inp_channels * exp_ratio),
            nn.ReLU(),
            nn.Conv2d(inp_channels * exp_ratio, out_channels, 1)
        )
            modules_gmsa['gmsa_{}'.format(i+1)] = LKAB(inp_channels)
        self.modules_lfe = nn.ModuleDict(modules_lfe)
        self.modules_gmsa = nn.ModuleDict(modules_gmsa)


    def forward(self, x):
        atn = None
        for i in range(1 + self.shared_depth):
            if i == 0: ## only calculate attention for the 1-st module
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y, _ = self.modules_gmsa['gmsa_{}'.format(i)](x, None)
                x = y + x
            else:
                x = self.modules_lfe['lfe_{}'.format(i)](x) + x
                y = self.modules_gmsa['gmsa_{}'.format(i)](x)
                x = y + x
        return x

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.depthwise = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.depthwise_dilated = nn.Conv2d(dim, dim, 5, stride=1, padding=6, groups=dim, dilation=3)

    def forward(self, x):
        u = x.clone()
        attn = self.pointwise(x)
        attn = self.depthwise(attn)
        attn = self.depthwise_dilated(attn)
        return u * attn

class LKAB(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        num_group = num_feat // 2
        self.proj_1 = nn.Conv2d(num_feat, num_group, 1)
        self.activation = nn.GELU()
        self.atten_branch = Attention(num_group)
        self.proj_2 = nn.Conv2d(num_group, num_feat, 1)
        self.pixel_norm = nn.LayerNorm(num_feat)
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.atten_branch(x)
        x = self.proj_2(x)
        x = x + shorcut

        x = x.permute(0, 2, 3, 1) #(B, H, W, C)
        x = self.pixel_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous() #(B, C, H, W)

        return x
       

class LHAN(nn.Module):
    def __init__(self, args):
        super(LHAN, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.window_sizes = args.window_sizes
        self.m_elan  = args.m_elan
        self.c_elan  = args.c_elan
        self.n_share = args.n_share
        self.r_expand = args.r_expand
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)


        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1+self.n_share)):
            if (i+1) % 2 == 1: 
                m_body.append(
                    LHAB(
                        self.c_elan, self.c_elan, self.r_expand, 0, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_body.append(
                    LHAB(
                        self.c_elan, self.c_elan, self.r_expand, 1, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            m_body.append(ESA(self.c_elan))

            
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        x = self.sub_mean(x)
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


def create_model(args):
    return LHAN(args)

