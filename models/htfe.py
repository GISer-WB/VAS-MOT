import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F
from timm.models.layers import DropPath
import lightning as L
from .mamba_simple import Mamba
from einops import rearrange

def random_masking_3D(xb, mask_ratio):
    # xb: [bs x num_patch x dim]
    bs, L, D = xb.shape
    x = xb.clone()

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(bs, L, device=xb.device)  # noise in [0, 1], bs x L

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep]
    x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # x_kept: [bs x len_keep x dim]

    # removed x
    x_removed = torch.zeros(bs, L - len_keep, D, device=xb.device)  # x_removed: [bs x (L-len_keep) x dim]
    x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x dim]

    # combine the kept part and the removed one
    x_masked = torch.gather(x_, dim=1,
                            index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x num_patch x dim]

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([bs, L], device=x.device)  # mask: [bs x num_patch]
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)  # [bs x num_patch]
    return x_masked, x_kept, mask, ids_restore

class IDMamba_Block(L.LightningModule):
    def __init__(self, dim, drop=0., d_state=16, d_conv_1=2, d_conv_2=4, e_fact=1):
        super().__init__()
        #self.configs = configs
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)
        self.conv = nn.Conv1d(dim, dim, 1)
        self.mamba = Mamba(dim, d_state=d_state, d_conv_1=d_conv_1, d_conv_2=d_conv_2, expand=e_fact)
        
    def forward(self, x):
        x_act = self.act(x)
        x1, x2 = self.mamba(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2 * x_act
        out2 = x2 * x1_2 * x_act

        x = out1 + out2
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x

def complex_relu(x):
    real = F.relu(x.real)
    imag = F.relu(x.imag)
    return torch.complex(real, imag)

class LearnableFilterLayer(nn.Module):
    def __init__(self, dim):
        super(LearnableFilterLayer, self).__init__()
        self.complex_weight_1 = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_2 = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_1, std=.02)
        trunc_normal_(self.complex_weight_2, std=.02)

    def forward(self, x_fft):
        weight_1 = torch.view_as_complex(self.complex_weight_1)
        weight_2 = torch.view_as_complex(self.complex_weight_2)
        x_weighted = x_fft * weight_1
        x_weighted = complex_relu(x_weighted)
        x_weighted = x_weighted * weight_2
        return x_weighted

class Adaptive_Fourier_Filter_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.learnable_filter_layer_1 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_2 = LearnableFilterLayer(dim)
        self.learnable_filter_layer_3 = LearnableFilterLayer(dim)
        
        self.threshold_param = nn.Parameter(torch.rand(1) * 0.5)
        self.low_pass_cut_freq_param = nn.Parameter(dim // 2 - torch.rand(1) * 0.5)
        self.high_pass_cut_freq_param = nn.Parameter(dim // 4 - torch.rand(1) * 0.5)
    
    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)

        threshold = torch.quantile(normalized_energy, self.threshold_param_high)
        dominant_frequencies = normalized_energy > threshold

        # Initialize adaptive mask
        adaptive_mask = torch.zeros_like(x_fft, device=x_fft.device)
        adaptive_mask[dominant_frequencies] = 1

        return adaptive_mask
        
    def adaptive_freq_pass(self, x_fft, flag="high"): 
        B, H, W_half = x_fft.shape  # W_half is the reduced dimension for real FFT
        W = (W_half - 1) * 2  # Calculate the full width assuming the input was real
        
        # Generate the non-negative frequency values along one dimension
        freq = torch.fft.rfftfreq(W, d=1/W).to(x_fft.device)
        
        if flag == "high": 
            freq_mask = torch.abs(freq) >= self.high_pass_cut_freq_param.to(x_fft.device)#
            freq_mask = torch.abs(freq) <= self.low_pass_cut_freq_param.to(x_fft.device)#
        return x_fft * freq_mask#
        
    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
         
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')

        #if args.adaptive_filter:
        # freq_mask = self.create_adaptive_high_freq_mask(x_fft)
        x_low_pass = self.adaptive_freq_pass(x_fft, flag="low")#低通
        
        x_high_pass = self.adaptive_freq_pass(x_fft, flag="high")#高通

        x_weighted = self.learnable_filter_layer_1(x_fft) + self.learnable_filter_layer_2(x_low_pass) + self.learnable_filter_layer_3(x_high_pass)

        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x


class Affirm_layer(L.LightningModule):
    def __init__(self, dim, mlp_ratio=3., drop=0., bias=True, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        #self.configs = configs
        self.affb = Adaptive_Fourier_Filter_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.idmamba = IDMamba_Block(dim=dim)

    def forward(self, x_in):
      
        x = self.norm1(x_in)
        x = self.affb(x)
        x = self.norm2(x)
        x = self.idmamba(x)
        x = x_in + self.drop_path(x)

        return x

class Affirm(nn.Module):

    def __init__(self, patch_size=1, seq_len=9, emb_dim=256, dropout=0.1, depth=2, mask_ratio=0.4, pred_len=9):
        super(Affirm, self).__init__()
        
        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]

        self.affirm_blocks = nn.ModuleList([
            Affirm_layer(dim=emb_dim, drop=dropout, drop_path=dpr[i])
            for i in range(depth)]
        )

    def forward(self, x):
        for affirm_block in self.affirm_blocks:
            x = affirm_block(x)
        return x 