"""by lyuwenyu
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation

from .yaml_utils import register

import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import einops

__all__ = ['HybridEncoder']


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    #print ('wp', x.shape)  #torch.Size([32, 20, 20, 64])
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class BiA_Attention(nn.Module):
    '''
    :param
        dim(int): Number of input channels.
        input_resolution(tuple[int]): Resolution of input features (H, W)
        stride(int): The sample rate of reference_points. (1 or 2)
        postype(str): Type of position encoding. ("conv" or "rel")
        dynamic_factor(int): The transform scale of Dynamic Attention.
        num_heads (int): Number of attention heads.
        k(int): The center points number. (2 or 3)
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    '''

    def __init__(self, dim, input_resolution, num_heads, stride, dynamic_factor=2, window_size=7, Attention_Group=2,
                 postype="rel", k=None, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.heads = num_heads
        self.channels = dim
        self.head_channels = self.channels // self.heads
        self.head_groups = Attention_Group
        self.head_per_group = self.heads // self.head_groups
        self.groups_channel = self.channels // self.head_groups
        self.img_size = input_resolution

        # if self.img_size[0] in [36]:
        #     self.offset_stride = 4
        # else:
        #     self.offset_stride = 2

        self.window_size = window_size
        self.window_num = (int(input_resolution[0] // window_size), int(input_resolution[1] / window_size))
        #print ('wh,ww',int(input_resolution[0] // window_size), int(input_resolution[1] / window_size))

        self.sample_size = (math.ceil(self.img_size[0] / stride), math.ceil(self.img_size[1] / stride))

        #print ('self.sample_size', self.sample_size)

        self.factor = dynamic_factor
        self.stride = stride
        self.pos = postype
        self.scale = qk_scale or dim ** -0.5

        self.k = k

        self.use_center = False
        if self.k is not None:
            self.use_center = True

        self.offset_mats = None
        self.center_offset = None

        if self.use_center:
            if k == 2:
                self.offset_mats = torch.tensor([(0, -1), (-1, 0), (0, 1), (1, 0)])
            elif k == 3:
                self.offset_mats = torch.tensor(
                    [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, 0)])
            # compute the final offset.
            self.register_buffer("offset_mat", self.offset_mats)
            self.center_channels = self.channels // 4
            self.center_downsample = nn.Conv2d(self.channels, self.center_channels, 1, 1, 0)
            self.center_offset = nn.Sequential(
                nn.Conv2d(self.groups_channel // 4, self.groups_channel // 4, 3, 1, 1),
                nn.BatchNorm2d(self.groups_channel // 4),
                nn.GELU(),
                nn.Conv2d(self.groups_channel // 4, 2 * k * k, 1, 1, 0)
            )
            self.center_q = nn.Conv2d(self.center_channels, self.center_channels, 1, 1, 0, bias=qkv_bias)
            self.center_k = nn.Conv2d(self.center_channels, self.center_channels, 1, 1, 0, bias=qkv_bias)
            self.center_v = nn.Conv2d(self.center_channels, self.center_channels, 1, 1, 0, bias=qkv_bias)

            # compute the relative position embedding
            center_refX = torch.arange(0, self.img_size[0], 1).view(-1, 1).repeat(1, self.img_size[1])
            center_refY = torch.arange(0, self.img_size[1], 1).view(1, -1).repeat(self.img_size[0], 1)
            center = torch.stack((center_refX, center_refY), dim=-1)
            self.register_buffer("center", center)

        self.offset = nn.Sequential(
            nn.Conv2d(self.groups_channel, self.groups_channel, kernel_size=3, stride=1, padding=0, groups=self.groups_channel, bias=False),
            nn.BatchNorm2d(self.groups_channel),
            nn.GELU(),
            
            nn.Conv2d(self.groups_channel, self.groups_channel, kernel_size=3, stride=1, padding=1, groups=self.groups_channel, bias=False),
            nn.BatchNorm2d(self.groups_channel),
            nn.GELU(),
            
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            
            nn.Conv2d(self.groups_channel, self.groups_channel, kernel_size=3, stride=2, padding=1, groups=self.groups_channel, bias=False),
            nn.BatchNorm2d(self.groups_channel),
            nn.GELU(),
            
            nn.Conv2d(self.groups_channel, 2 * self.sample_size[0] * self.sample_size[1], kernel_size=1, stride=1, padding=1, bias=False)
        )

        self.q_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        self.k_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        self.v_conv = nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=qkv_bias)
        if self.use_center:
            self.proj = nn.Conv2d(self.channels + self.channels//4, self.channels, 1, 1, 0)
        else:
            self.proj = nn.Conv2d(self.channels, self.channels, 1, 1, 0)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        if self.pos == "conv":
            # depth-wise
            self.posembed = nn.Conv2d(self.channels, self.channels, 3, 1, 1, groups=self.channels)
        elif self.pos == "rel":  # relative pos embedding
            self.posembed = nn.Parameter(torch.zeros(self.heads, 2 * self.img_size[0] - 1, 2 * self.img_size[1] - 1))
            trunc_normal_(self.posembed, std=0.01)

            ref_pos = (torch.arange(0, self.img_size[0], self.stride).view(1, -1)
                       - torch.arange(0, self.img_size[1], 1).view(-1, 1))

            ref_pos_x = ref_pos.repeat(self.img_size[0], self.sample_size[0])
            ref_pos_y = ref_pos.repeat_interleave(self.img_size[1], dim=0).repeat_interleave(self.sample_size[1], dim=1)

            ref_pos_point = torch.stack((ref_pos_y, ref_pos_x), dim=-1)
            self.register_buffer("ref_pos_point", ref_pos_point)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch.no_grad()
    def reference_points(self, batch, H, W, rH, rW, stride, dtype=torch.float32, device=None):
        refX = torch.arange(0, H, stride, dtype=dtype, device=device).view(-1, 1).repeat(1, rW)
        # print(refX.shape)
        refY = torch.arange(0, W, stride, dtype=dtype, device=device).view(1, -1).repeat(rH, 1)
        # print(refY)

        center = None
        if self.use_center:
            center = self.center
            center = center.type(dtype)
            center = center.reshape(H * W, 2).unsqueeze(dim=0)
            center = center.expand(self.k * self.k, H * W, 2).permute(1, 0, 2)

            offset_mat = self.offset_mat.unsqueeze(dim=0).expand(H * W, self.k * self.k, 2)
            center = center + offset_mat
            center[..., 0] = torch.clip(center[..., 0], 0, H - 1).div(H - 1).mul(2).sub(1)
            center[..., 1] = torch.clip(center[..., 1], 0, W - 1).div(W - 1).mul(2).sub(1)
            center = center.unsqueeze(dim=0).expand(batch * self.head_groups, H * W, self.k * self.k, 2)

        grid = torch.stack((refX, refY), dim=-1)
        grid[..., 0] = grid[..., 0].div(H - 1).mul(2).sub(1)
        #使用公式 grid[..., 0].div(H - 1).mul(2).sub(1) 将网格坐标缩放到 [-1, 1] 范围内。这种缩放在深度学习中常用于规范化网格坐标，以便进行像网格采样这样的操作。
        grid[..., 1] = grid[..., 1].div(W - 1).mul(2).sub(1)
        grid = grid.unsqueeze(dim=0).expand(batch * self.head_groups * self.window_num[0] * self.window_num[1], rH, rW, 2)

        return grid, center

    def forward(self, x):

        B, C, H, W = x.shape
        #print (x.shape)
        Wh, Ww = self.window_num[0], self.window_num[1]
        #print('Wh, Ww', Wh, Ww)

        dtype = x.dtype
        device = x.device

        rH = math.ceil(H / self.stride)
        rW = math.ceil(W / self.stride)

        
        #print ('rH,rW',rH,rW)

        h = self.heads
        h_c = self.head_channels

        hg = self.head_groups
        gc = self.groups_channel
        cgc = gc // 4
        hpg = self.head_per_group

        # ref_points:（B*hg*Wh*Ww, rH, rW, 2） center_points: (B*hg, H*W, k*k, 2)
        ref_points, center_points = self.reference_points(B, H, W, rH, rW, self.stride, dtype=dtype, device=device)

        # (B*hg, Wh*Ww, rH*rW, 2)
        ref_points = einops.rearrange(ref_points, '(b h Wh Ww) rh rw g -> (b h) (Wh Ww) (rh rw) g',
                                      b=B, h=hg, Wh=Wh, Ww=Ww, rh=rH, rw=rW)

        # query: (B, C, H, W)
        query = self.q_conv(x)

        # offset:(B*hg, 2*rh*rw, Wh, Ww)
        offset = self.offset(query.reshape(B * hg, gc, H, W))
        #8*2, 2*12*12, 8*8
        #print ('offset shape:',offset.shape)
        # offset:(B, 2, hg, Wh*Ww, rH*rW)
        offset = offset.reshape(B * hg, 2, rH * rW, Wh * Ww).permute(0, 3, 2, 1)
        #print("reshaped offset shape:", offset.shape)
        ranges = torch.tensor([1.0 / H, 1.0 / W], device=device).view([1, 1, 1, 2])
        offset = self.tanh(offset).mul(ranges)

        if self.factor > 1:
            offset = offset.mul(self.factor)

        sample_grid = offset + ref_points

        #print (sample_grid.shape)

        # value（B, h, h_c, H, W)
        value = x.reshape(B, hg, gc, H, W).reshape(B * hg, gc, H, W)
        # grid_sample: (B*hg, gc, Wh*Ww, rH*rW)
        grid_sample = F.grid_sample(
            input=value,
            grid=sample_grid[..., (1, 0)],   
            mode="bilinear",
            #mode="nearest",
            align_corners=True
        )

        grid_sample = grid_sample.reshape(B, C, Wh * Wh, rH * rW)

        # k,v: (B*h, h_c, Wh*Ww, rH*rW)   #h_c = 128
        k = self.k_conv(grid_sample).reshape(B * h, h_c, Wh * Wh, rH * rW).permute(0, 2, 1, 3)
        v = self.v_conv(grid_sample).reshape(B * h, h_c, Wh * Wh, rH * rW).permute(0, 2, 1, 3)

        # (B*h, H, W, h_c)
        window_query = query.reshape(B * h, h_c, H, W).permute(0, 2, 3, 1)
        # B*h*Wh*Ww, W_s, W_s, C
        window_query = window_partition(window_query, self.window_size).reshape(B * h, Wh * Ww,
                                                                                self.window_size * self.window_size,h_c)

        if self.scale:
            window_query = window_query * self.scale
        attn = window_query @ k    

        center_output = None
        if self.use_center:
            kc = self.k
            # (batch, cen_c, H, W)
            center_x = self.center_downsample(x)

            # (batch, cen_c, H, W)
            center_q = self.center_q(center_x)

            center_offset = self.center_offset(center_q.reshape(B, hg, cgc, H, W).reshape(B * hg, cgc, H, W))\
                .reshape(B * hg, 2, kc * kc, H * W).permute(0, 3, 2, 1)

            center_offset = self.tanh(center_offset).mul(
                torch.tensor([1.0 / H, 1.0 / W], device=device).view(1, 1, 1, 2))
            # (B*hg, H*W, k*k ,2)
            center_grid = center_offset + center_points

            center_value = center_x.reshape(B, hg, cgc, H, W).reshape(B * hg, cgc, H, W)

            # (B*hg, gc, H*W, k*k)
            center_sample = F.grid_sample(
                input=center_value,
                grid=center_grid[..., (1, 0)],
                #mode="nearest",
                mode="bilinear",
                align_corners=True
            )

            center_sample = einops.rearrange(center_sample, '(b h) hc A B->b (h hc) A B', b=B, h=hg, hc=cgc)
            center_k = self.center_k(center_sample).reshape(B * h, h_c // 4, H * W, kc * kc)
            center_v = self.center_v(center_sample).reshape(B * h, h_c // 4, H * W, kc * kc)

            center_q = center_q.reshape(B * h, h_c // 4, H * W).unsqueeze(dim=2).permute(0, 3, 2, 1)
            # (B*h, H*W, 1, h_c)

            if self.scale:
                center_q = center_q * self.scale
            # (B*h, H*W, 1, k*k)
            center_attn = center_q @ (center_k.permute(0, 2, 1, 3))

            if self.pos == "rel":
                center = self.center
                center = center.type(dtype)
                center = center.reshape(H * W, 2).unsqueeze(dim=0)
                # (H*W, k*k, 2)
                center = center.expand(self.k * self.k, H * W, 2).permute(1, 0, 2)
                center = center[None, ...].expand(B * hg, H * W, self.k * self.k, 2)
                center_rel_pos = center_offset - center

                rel_pos_tab = self.posembed[None, ...].expand(B, h, 2 * H - 1, 2 * W - 1).reshape(B * hg, hpg,
                                                                                                  2 * H - 1, 2 * W - 1)
                # (B*hg, hpg, H*W, k*k)
                d_center_rel_pos = F.grid_sample(input=rel_pos_tab,
                                                 grid=center_rel_pos[..., (1, 0)],
                                                 #mode="nearest",
                                                 mode="bilinear",
                                                 align_corners=True
                                                 )
                # (B*h, 1, HW, k*k)
                d_center_rel_pos = d_center_rel_pos.reshape(B * h, 1, H * W, self.k * self.k).permute(0, 2, 1, 3)
                center_attn = center_attn + d_center_rel_pos

            center_attn = self.softmax(center_attn)

            center_attn = self.attn_drop(center_attn)
            #(B/h, H*W, 1, h_c)
            center_output = center_attn @ (center_v.permute(0, 2, 3, 1))
            center_output = center_output.permute(0, 3, 1, 2).squeeze(dim=-1).reshape(B, C // 4, H, W)

        dwpos = None
        if self.pos == "conv":
            dwpos = self.posembed(query)
        else:
            # (H*W, rH*rW, 2)
            relative_dynamic_point = self.ref_pos_point
            relative_dynamic_point = relative_dynamic_point.type(dtype)
            relative_dynamic_point[..., 0] = relative_dynamic_point[..., 0].mul(1.0).div(self.img_size[0] - 1)
            relative_dynamic_point[..., 1] = relative_dynamic_point[..., 1].mul(1.0).div(self.img_size[1] - 1)
            relative_dynamic_point = relative_dynamic_point[None, ...].expand(B * hg, H * W, rH * rW, 2).clone()

            # (B*hg, Wh*Ww, rH*rW, 2）->(B*hg, H*W, rH*rW, 2)
            offset = offset.reshape(B * hg, Wh, Ww, rH * rW, 2).repeat_interleave(self.window_size,
                                                                                  dim=1).repeat_interleave(
                self.window_size, dim=2)
            offset = offset.reshape(B * hg, H * W, rH * rW, 2)
            relative_dynamic_point -= offset

            rel_pos_tab = self.posembed[None, ...].expand(B, h, 2 * H - 1, 2 * W - 1).reshape(B * hg, hpg, 2 * H - 1,
                                                                                              2 * W - 1)  #
            # （B*hg，hpg，H*W，rH*rW）
            d_rel_pos = F.grid_sample(input=rel_pos_tab,
                                      grid=relative_dynamic_point[..., (1, 0)],
                                      #mode="nearest",
                                      mode="bilinear",
                                      align_corners=True
                                      )
            # （B*h，H*W，rH*rW）
            d_rel_pos = d_rel_pos.reshape(B, h, H, W, rH * rW).reshape(B * h, H, W, rH * rW)
            d_rel_pos = d_rel_pos.view(B * h, H // self.window_size, self.window_size, W // self.window_size,
                                       self.window_size, rH * rW)
            # (B*h, Wh*Ww, Ws*Ws, rH*rW)
            d_rel_pos = d_rel_pos.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B * h, Wh * Ww,
                                                                                 self.window_size * self.window_size,
                                                                                 rH * rW)

            # attn = attn + d_rel_pos
            attn = attn + d_rel_pos

        # attn: (batch*head, Wh*Ww,Ws*Ws,rH*rW)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # x: (batch*head, Wh*Ww, Ws*Ws, head_Channels)
        x = attn @ (v.permute(0, 1, 3, 2))
        x = x.reshape(B, h, Wh * Ww, self.window_size * self.window_size, h_c).permute(0, 2, 3, 1, 4).reshape(
            B * Wh * Ww, -1, C)

        x = window_reverse(x, self.window_size, H, W).permute(0, 3, 1, 2)

        if self.pos == "conv":
            x = x + dwpos

        if self.use_center:
            x = torch.cat((x, center_output), dim=1)
        # x: (batch, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x#, sample_grid

class asyConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(asyConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
            self.initialize()
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)
            self.initialize()


    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

    def initialize(self):
        weight_init(self)

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class DFA(nn.Module):
    """ Enhance the feature diversity.
    """
    def __init__(self, x, y):
        super(DFA, self).__init__()
        self.asyConv = asyConv(in_channels=x, out_channels=y, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.oriConv = nn.Conv2d(x, y, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(x, y, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(y), nn.PReLU()
        )           
        self.conv2d = nn.Conv2d(y*3, y, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(y)
        self.initialize()

    def forward(self, f):
        p1 = self.oriConv(f)
        p2 = self.asyConv(f)
        p3 = self.atrConv(f)
        p  = torch.cat((p1, p2, p3), 1)
        p  = F.relu(self.bn2d(self.conv2d(p)), inplace=True)

        return p

    def initialize(self):
        #pass
        weight_init(self)


class SEA(nn.Module):
    def __init__(self, channels=64, r=4):
        super(SEA, self).__init__()
        out_channels = int(channels // r)

        # local_att
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # global_att
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

        # channel_att
        self.channel_att = nn.Sequential(
            nn.Conv2d(channels, channels // r, kernel_size=1),
            nn.BatchNorm2d(channels // r),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # local_att
        xl = self.local_att(x)
        # global_att
        xg = self.global_att(x)
        # channel_att
        xc = self.channel_att(x)
        # weighted sum of local and global attention features
        xlg = xl + xg
        # apply channel attention
        xla = xc * xlg
        # sigmoid activation
        wei = self.sig(xla)

        return wei

def cus_sample(feat, **kwargs):
    """
    :param feat:
    :param kwargs: size or scale_factor
    """
    assert len(kwargs.keys()) == 1 and list(kwargs.keys())[0] in ["size", "scale_factor"]
    return F.interpolate(feat, **kwargs, mode="bilinear", align_corners=False)

class SFF(nn.Module):
    def __init__(self, channels=64):
        super(SFF, self).__init__()

        self.conv_gate = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_gate = nn.BatchNorm2d(channels * 2)
        self.sigmoid = nn.Sigmoid()

        self.relu = nn.PReLU()

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=1)

        gate_conv = self.conv_gate(xy)
        gate_bn = self.bn_gate(gate_conv)
        gate_sigmoid = self.sigmoid(gate_bn)

        feat = xy * gate_sigmoid
        feat = self.relu(feat)
        return feat

@register
class FusionModule(nn.Module):
    def __init__(self, feature_dim, n_features):
        super(FusionModule, self).__init__()
        self.feature_dim = feature_dim
        self.n_features = n_features
        self.weighting = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * n_features, n_features, bias=True),
        )
        self.weighting.apply(self.init)

    def init(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(0.)
            m.bias.data.fill_(1.)

    def forward(self, x):
        weight = nn.functional.softmax(self.weighting(x), dim=1)
        out = None
        for idx in range(self.n_features):
            start, end = idx * self.feature_dim, (idx + 1) * self.feature_dim
            if out is None:
                out = torch.einsum('b, bchw-> bchw', weight[:, idx], x[:, start:end, :, :])
            else:
                out = out + torch.einsum('b, bchw-> bchw', weight[:, idx], x[:, start:end, :, :])
        return out

class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        self.lateral_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))

        self.downsample_convs = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )

        self.bia_attention = BiA_Attention(
            dim=256, 
            input_resolution=None,
            num_heads=2, 
            stride=2, 
            k=3,
            window_size=3, 
            dynamic_factor=10,
            postype='rel', 
            Attention_Group=2,
            qkv_bias=True, 
            qk_scale=None, 
            attn_drop=0., 
            proj_drop=0.0
        )

        self.FUSION = FusionModule(feature_dim=256, n_features=2)
        self.DFA = DFA(x=256, y=256)
        self.SFF = SFF(channels=256)

    def forward(self, feats1):
        proj_feats = []
        for l, feat in enumerate(feats1):
            src, _ = feat.decompose()
            proj_feats.append(self.input_proj[l](src))

        proj_feats_5_0 = self.DFA(proj_feats[-1])
        height, width = map(int, proj_feats[-1].shape[2:])

        self.bia_attention.input_resolution = (height, width)
        self.bia_attention = self.bia_attention.to(proj_feats[0].device)
        
        proj_feats_5_1 = self.bia_attention(proj_feats[-1])
        proj_feats[-1] = self.FUSION(self.SFF(proj_feats_5_0, proj_feats_5_1))

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]] 
        for idx in range(len(self.in_channels) - 1, 0, -1):   
            feat_high = inner_outs[0] 
            feat_low = proj_feats[idx - 1] 
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high

            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.FUSION(torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]] 
        for idx in range(len(self.in_channels) - 1):  
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1] 
            downsample_feat = self.downsample_convs[idx](feat_low) 
            out = self.FUSION(torch.concat([downsample_feat, feat_high], dim=1))
            outs.append(out)
        
        return outs

def build_hybird_encoder(args):
    return HybridEncoder(
        in_channels=[64, 256, 512, 1024, 2048],
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        enc_act="gelu",
        act='silu',
        use_encoder_idx=[2],
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        eval_spatial_size=None  #[672,672]
    )
