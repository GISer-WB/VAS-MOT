"""by lyuwenyu
"""

import math
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .denoising import get_contrastive_denoising_training_group
from .utils import deformable_attention_core_func, get_activation, inverse_sigmoid
from .utils import bias_init_with_prob

from .post_trans import MSA_yolov

from .yaml_utils import register

__all__ = ['RTDETRTransformer']


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, act='relu'):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MSDeformableAttention(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, ):
        """
        Multi-Scale Deformable Attention Module
        """
        super(MSDeformableAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.total_points = num_heads * num_levels * num_points

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2, )
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.ms_deformable_attn_core = deformable_attention_core_func

        self._reset_parameters()

    def _reset_parameters(self):
        # sampling_offsets
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 1, 2).tile([1, self.num_levels, self.num_points, 1])
        scaling = torch.arange(1, self.num_points + 1, dtype=torch.float32).reshape(1, 1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()

        # attention_weights
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

        # proj
        init.xavier_uniform_(self.value_proj.weight)
        init.constant_(self.value_proj.bias, 0)
        init.xavier_uniform_(self.output_proj.weight)
        init.constant_(self.output_proj.bias, 0)

    def forward(self,
                query,
                reference_points,
                value,
                value_spatial_shapes,
                value_mask=None):
        """
        Args:
            query (Tensor): [bs, query_length, C]
            reference_points (Tensor): [bs, query_length, n_levels, 2], range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area
            value (Tensor): [bs, value_length, C]
            value_spatial_shapes (List): [n_levels, 2], [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
            value_level_start_index (List): [n_levels], [0, H_0*W_0, H_0*W_0+H_1*W_1, ...]
            value_mask (Tensor): [bs, value_length], True for non-padding elements, False for padding elements

        Returns:
            output (Tensor): [bs, Length_{query}, C]
        """
        bs, Len_q = query.shape[:2]
        Len_v = value.shape[1]

        value = self.value_proj(value)
        if value_mask is not None:
            value_mask = value_mask.astype(value.dtype).unsqueeze(-1)
            value *= value_mask
        value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).reshape(
            bs, Len_q, self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1).reshape(
            bs, Len_q, self.num_heads, self.num_levels, self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2)
            sampling_locations = reference_points.reshape(
                bs, Len_q, 1, self.num_levels, 1, 2
            ) + sampling_offsets / offset_normalizer
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                    reference_points[:, :, None, :, None, :2] + sampling_offsets /
                    self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5)
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".
                format(reference_points.shape[-1]))

        output = self.ms_deformable_attn_core(value, value_spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 n_head=8,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 n_levels=4,
                 n_points=4, ):
        super(TransformerDecoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = getattr(F, activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # self._reset_parameters()

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        return self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

    def forward(self,
                tgt,
                reference_points,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                attn_mask=None,
                memory_mask=None,
                query_pos_embed=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos_embed)

        tgt2, _ = self.self_attn(q, k, value=tgt, attn_mask=attn_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # cross attention
        tgt2 = self.cross_attn( \
            self.with_pos_embed(tgt, query_pos_embed),
            reference_points,
            memory,
            memory_spatial_shapes,
            memory_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # ffn
        tgt2 = self.forward_ffn(tgt)
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_layer, num_layers, eval_idx=-1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.trans_cls = MSA_yolov(dim=256, out_dim=256, num_heads=4, attn_drop=0.0)
        self.trans_box = MSA_yolov(dim=256, out_dim=256, num_heads=4, attn_drop=0.0) 

    def forward(self,
                tgt,
                ref_points_unact,
                memory,
                memory_spatial_shapes,
                memory_level_start_index,
                bbox_head,
                score_head,
                query_pos_head,
                attn_mask=None,
                memory_mask=None):
        output = tgt
        dec_out_bboxes = []
        dec_out_logits = []
        ref_points_detach = F.sigmoid(ref_points_unact) #(b, 500, 4)
        intermediate = []
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            query_pos_embed = query_pos_head(ref_points_detach)

            output = layer(output, ref_points_input, memory,
                           memory_spatial_shapes, memory_level_start_index,
                           attn_mask, memory_mask, query_pos_embed)  
            
            output_cls = self.trans_cls(output, None, None, sim_thresh=0.75, ave=True)
            output_box = self.trans_box(output, None, None, sim_thresh=0.75, ave=True)

            inter_ref_bbox = F.sigmoid(bbox_head[i](output_box) + inverse_sigmoid(ref_points_detach))

            if self.training:
                dec_out_logits.append(score_head[i](output_cls))
                if i == 0:
                    dec_out_bboxes.append(inter_ref_bbox)
                else:
                    dec_out_bboxes.append(F.sigmoid(bbox_head[i](output_box) + inverse_sigmoid(ref_points)))

            elif i == self.eval_idx:
                dec_out_logits.append(score_head[i](output_cls))
                dec_out_bboxes.append(inter_ref_bbox)
                break

            ref_points = inter_ref_bbox
            intermediate.append(output)
            ref_points_detach = inter_ref_bbox.detach(
            ) if self.training else inter_ref_bbox

        return torch.stack(dec_out_bboxes), torch.stack(dec_out_logits), torch.stack(intermediate)

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

class DEPTHWISECONV(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DEPTHWISECONV, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=in_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_ch)
        self.point_conv = nn.Conv2d(in_channels=in_ch,
                                    out_channels=out_ch,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class AdaptiveAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, 1, bias=False),
        )
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(out_channels)
        self.conv_1x1= nn.Conv2d(in_channels*3, out_channels, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(in_channels,out_channels, kernel_size=3,  bias=False)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        query = self.channel_attention(F.adaptive_avg_pool2d(self.conv_3x3(x),[height,1]))
        key = self.channel_attention(F.adaptive_avg_pool2d(x,[1,width]))

        value1 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//2, width//2]))))     #2,6,8
        value1 = F.interpolate(value1, size=x.size()[2:], mode='bilinear', align_corners=True)

        value2 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//6, width//6]))))
        value2 = F.interpolate(value2, size=x.size()[2:], mode='bilinear', align_corners=True)

        value3 = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(F.adaptive_avg_pool2d(x, [height//8, width//8]))))
        value3 = F.interpolate(value3, size=x.size()[2:], mode='bilinear', align_corners=True)

        value = torch.cat([value1,value2,value3],dim=1)
        value = self.conv_1x1(value)

        attention_scores = torch.matmul(query, key)
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.out_channels//2, dtype=torch.float32))

        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        attended_value = torch.matmul(attention_weights, value)

        output = x + self.gamma * attended_value

        return output

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class Transformer_Decoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, number_class):
        super(Transformer_Decoder, self).__init__()

        self.number_classes = number_class
        self.FusionModule = FusionModule(256, 3)
        self.cov1 = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
        )

        self.cov2 = nn.Sequential(
                        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.pre_1_16 = nn.Linear(256, number_class)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('Linear') != -1:
                nn.init.xavier_uniform_(m.weight),
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif classname.find('BatchNorm') != -1:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, z):
        x5 = x
        y4 = self.cov1(y)
        z3 = self.cov2(z)
        feat_t = torch.cat([x5, y4, z3], 1)
        feat_t = self.FusionModule(feat_t)
        B, Ct, Ht, Wt = feat_t.shape
        feat_t = feat_t.view(B, Ct, -1).transpose(1, 2)
        mask_x = self.pre_1_16(feat_t)
        mask_x = mask_x.transpose(1, 2).reshape(B, self.number_classes, Ht, Wt)
        return mask_x

class RCU(nn.Module):
    def __init__(self, channel, subchannel, number_class):
        super(RCU, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + 4*number_class, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, number_class, 3, padding=1)
        self.conv1_32 = nn.Conv2d(number_class, 32, 3, padding=1)
        self.query_conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.key_conv1 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.query_conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.key_conv2 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.query_conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.key_conv3 = nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)

        self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        y1 = y
        if self.group == 1:
            x_cat = torch.cat((x, y1), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1, xs[4], y1, xs[5], y1, xs[6], y1, xs[7], y1),
                              1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1, xs[4], y1, xs[5], y1, xs[6], y1, xs[7], y1,
                               xs[8], y1, xs[9], y1, xs[10], y1, xs[11], y1, xs[12], y1, xs[13], y1, xs[14], y1, xs[15],
                               y1), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1, xs[4], y1, xs[5], y1, xs[6], y1, xs[7], y1,
                               xs[8], y1, xs[9], y1, xs[10], y1, xs[11], y1, xs[12], y1, xs[13], y1, xs[14], y1, xs[15],
                               y1,
                               xs[16], y1, xs[17], y1, xs[18], y1, xs[19], y1, xs[20], y1, xs[21], y1, xs[22], y1,
                               xs[23], y1,
                               xs[24], y1, xs[25], y1, xs[26], y1, xs[27], y1, xs[28], y1, xs[29], y1, xs[30], y1,
                               xs[31], y1),
                              1)
        else:
            xs = torch.chunk(x, 64, dim=1)
            x_cat = torch.cat((xs[0], y1, xs[1], y1, xs[2], y1, xs[3], y1, xs[4], y1, xs[5], y1, xs[6], y1, xs[7], y1,
                               xs[8], y1, xs[9], y1, xs[10], y1, xs[11], y1, xs[12], y1, xs[13], y1, xs[14], y1, xs[15],
                               y1,
                               xs[16], y1, xs[17], y1, xs[18], y1, xs[19], y1, xs[20], y1, xs[21], y1, xs[22], y1,
                               xs[23], y1,
                               xs[24], y1, xs[25], y1, xs[26], y1, xs[27], y1, xs[28], y1, xs[29], y1, xs[30], y1,
                               xs[31], y1,
                               xs[32], y1, xs[33], y1, xs[34], y1, xs[35], y1, xs[36], y1, xs[37], y1, xs[38], y1,
                               xs[39], y1,
                               xs[40], y1, xs[41], y1, xs[42], y1, xs[43], y1, xs[44], y1, xs[45], y1, xs[46], y1,
                               xs[47], y1,
                               xs[48], y1, xs[49], y1, xs[50], y1, xs[51], y1, xs[52], y1, xs[53], y1, xs[54], y1,
                               xs[55], y1,
                               xs[56], y1, xs[57], y1, xs[58], y1, xs[59], y1, xs[60], y1, xs[61], y1, xs[62], y1,
                               xs[63], y1),
                              1)

        x_cat = self.conv(x_cat)
        x1_co = x_cat
        x2_co = y
        x2_co = self.conv1_32(x2_co)

        B1, C1, H1, W1 = x1_co.size()

        x_query1 = self.query_conv1(x1_co).view(B1, -1, W1 * H1)  # [b, c, hw]
        x_key1 = self.key_conv1(x1_co).view(B1, -1, W1 * H1)  # [b, c, hw]

        x_query2 = self.query_conv2(x1_co).view(B1, -1, W1 * H1)
        x_key2 = self.key_conv2(x1_co).view(B1, -1, W1 * H1)  # [b, c, hw]

        B2, C2, H2, W2 = x2_co.size()
        y_query = self.query_conv3(x2_co).view(B2, -1, W2 * H2)  # [b, c, hw]
        y_key = self.key_conv3(x2_co).view(B2, -1, W2 * H2)  # [b, c, hw]

        x_hw = torch.bmm(x_query1.permute(0, 2, 1), x_key2)  # [b, h1w1, h1w1]
        x_hw = F.softmax(x_hw, dim=-1)  # [b, h1w1, h1w1]

        x_c = torch.bmm(x_key1, x_query2.permute(0, 2, 1))  # [b, c1, c1]
        x_c = F.softmax(x_c, dim=-1)  # [b, c1, c1]

        xy_hw = torch.bmm(y_query, x_hw)  # [b, c2, h1w1]
        xy_c = torch.bmm(x_c.permute(0, 2, 1), y_key)  # [b, c1, h2w2]

        xy_hw = xy_hw.view(B1, 32, H1, W1)  # [b, 32, h, w]
        xy_c = xy_c.view(B2, 32, H2, W2)  # [b, 32, h, w]

        out_final = torch.cat([xy_hw, xy_c], 1)  # [b, 64, h, w]
        out_final = self.conv6(out_final)

        x = x + out_final
        y = y + self.score(x)
        return x, y

class ReverseStage(nn.Module):
    def __init__(self, channel, number_class):
        super(ReverseStage, self).__init__()
        self.weak_gra = RCU(channel, 8, number_class)

    def forward(self, x, y):
        y= -1 * torch.sigmoid(y) + 1
        x, y = self.weak_gra(x, y)
        return y

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class SIEM(nn.Module):
    def __init__(self, channel, number_class):
        super(SIEM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_atten = nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False)
        self.conv_1 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv_4 = BasicConv2d(channel, 256, kernel_size=3, stride=1, padding=1)
        self.conv_5 = BasicConv2d(2 * channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv1_32 = nn.Conv2d(number_class, 32, 1)
        self.conv256_32 = nn.Conv2d(256, 32, 1)
        self.conv64_32 = nn.Conv2d(64, 32, 1)

    def forward(self, x1, x2, y):
        x1 = self.max_pool(x1)

        x1 = self.conv64_32(x1)
        x2 = self.conv256_32(x2)
        x = torch.cat([x1, x2], 1)

        x = self.conv_5(x)  # 88 88

        left1 = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=True)  # 88
        right1 = F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True)  # 44

        left = self.conv_1(left1 * x)
        right = self.conv_2(right1 * y)
        right = F.interpolate(right, size=left.size()[2:], mode='bilinear', align_corners=True)

        x_co = left + right

        atten = self.avgpool(x_co)
        atten = torch.sigmoid(self.conv_atten(atten))

        out = torch.mul(x_co, atten) + x_co

        out = self.conv_3(out)
        out = self.conv_4(out)

        return out
    
@register
class RTDETRTransformer(nn.Module):
    __share__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 hidden_dim=256,
                 num_queries=300,
                 position_embed_type='sine',
                 feat_channels=[256, 256, 256],
                 feat_strides=[8, 16, 32],
                 num_levels=4,
                 num_decoder_points=4,
                 nhead=8,
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 dropout=0.,
                 activation="relu",
                 num_denoising=100,
                 label_noise_ratio=0.5,
                 box_noise_scale=1.0,
                 learnt_init_query=False,
                 eval_spatial_size=None,
                 eval_idx=-1,
                 eps=1e-2,
                 aux_loss=True):

        super(RTDETRTransformer, self).__init__()
        assert position_embed_type in ['sine', 'learned'], \
            f'ValueError: position_embed_type not supported {position_embed_type}!'
        assert len(feat_channels) <= num_levels
        assert len(feat_strides) == len(feat_channels)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_decoder_layers = num_decoder_layers
        self.eval_spatial_size = eval_spatial_size
        self.aux_loss = aux_loss

        # backbone feature projection
        self._build_input_proj_layer(feat_channels)

        # Transformer module
        decoder_layer = TransformerDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, activation, num_levels,
                                                num_decoder_points)
        self.decoder = TransformerDecoder(hidden_dim, decoder_layer, num_decoder_layers, eval_idx)

        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        # denoising part
        if num_denoising > 0:
            # self.denoising_class_embed = nn.Embedding(num_classes, hidden_dim, padding_idx=num_classes-1) # TODO for load paddle weights
            self.denoising_class_embed = nn.Embedding(num_classes + 1, hidden_dim, padding_idx=num_classes)

        # decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)

        # encoder head
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, )
        )
        self.enc_score_head = nn.Linear(hidden_dim, num_classes)
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

        # decoder head
        self.dec_score_head = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for _ in range(num_decoder_layers)
        ])
        self.dec_bbox_head = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, num_layers=3)
            for _ in range(num_decoder_layers)
        ])

        # init encoder output anchors and valid_mask
        if self.eval_spatial_size:
            self.anchors, self.valid_mask = self._generate_anchors()

        self.TD = Transformer_Decoder(256, 4, 6, self.num_classes)
        self.RFB_modified = RFB_modified(in_channel=256, out_channel=32)
        self.conv4x32 = nn.Conv2d(self.num_classes, 32, kernel_size=1, stride=1, padding=0)
        self.RS5 = ReverseStage(channel=32, number_class=self.num_classes)
        self.RS4 = FusionModule(32, 2)
        self.siem = SIEM(channel=32,number_class=self.num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        bias = bias_init_with_prob(0.01)

        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            init.constant_(reg_.layers[-1].weight, 0)
            init.constant_(reg_.layers[-1].bias, 0)

        # linear_init_(self.enc_output[0])
        init.xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            init.xavier_uniform_(self.tgt_embed.weight)
        init.xavier_uniform_(self.query_pos_head.layers[0].weight)
        init.xavier_uniform_(self.query_pos_head.layers[1].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim, ))])
                )
            )

        in_channels = feat_channels[-1]

        for _ in range(self.num_levels - len(feat_channels)):
            self.input_proj.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channels, self.hidden_dim, 3, 2, padding=1, bias=False)),
                    ('norm', nn.BatchNorm2d(self.hidden_dim))])
                )
            )
            in_channels = self.hidden_dim

    def _get_encoder_input(self, encoder_feats, feats):

        S_g = self.TD(feats[2],feats[1],feats[0])
        feats1 = self.RFB_modified(feats[1])
        feats0 = self.RFB_modified(feats[0])
        # ----stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=2, mode='bilinear') 

        ra5_feat = self.RS5(feats1, guidance_g)
        S_5 = ra5_feat + guidance_g

        S_5 = self.conv4x32(S_5)
        # ----stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra4_feat = self.RS4(torch.cat((feats0, guidance_5),1))
        S_4 = ra4_feat + guidance_5

        S_4= F.interpolate(S_4, scale_factor=2, mode='bilinear')  # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)
        
        Sg = self.siem(encoder_feats[0], encoder_feats[1], S_4)

        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        proj_feats.insert(0, Sg)

        if self.num_levels > len(proj_feats): 
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        # get encoder inputs
        feat_flatten = []
        spatial_shapes = []
        level_start_index = [0, ]
        for i, feat in enumerate(proj_feats):
            _, _, h, w = feat.shape
            # [b, c, h, w] -> [b, h*w, c]
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            # [num_levels, 2]
            spatial_shapes.append([h, w])
            # [l], start index of each level
            level_start_index.append(h * w + level_start_index[-1])

        # [b, l, c]
        feat_flatten = torch.concat(feat_flatten, 1)
        level_start_index.pop() 
        return (feat_flatten, spatial_shapes, level_start_index, S_g)

    def _generate_anchors(self,
                          spatial_shapes=None,
                          grid_size=0.05,
                          dtype=torch.float32,
                          device='cpu'):
        if spatial_shapes is None:
            spatial_shapes = [[int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                              for s in self.feat_strides
                              ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid( \
                torch.arange(end=h, dtype=dtype), \
                torch.arange(end=w, dtype=dtype), indexing='ij')
            grid_xy = torch.stack([grid_x, grid_y], -1)
            valid_WH = torch.tensor([w, h]).to(dtype)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))

        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        # anchors = torch.where(valid_mask, anchors, float('inf'))
        # anchors[valid_mask] = torch.inf # valid_mask [1, 8400, 1]
        anchors = torch.where(valid_mask, anchors, torch.inf)

        return anchors, valid_mask

    def _get_decoder_input(self,
                           memory,
                           spatial_shapes,
                           denoising_class=None,
                           denoising_bbox_unact=None):
        bs, _, _ = memory.shape
        # prepare input for decoder
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors, valid_mask = self.anchors.to(memory.device), self.valid_mask.to(memory.device)

        # memory = torch.where(valid_mask, memory, 0)
        memory = valid_mask.to(memory.dtype) * memory  # TODO fix type error for onnx export 

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_unact = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_unact.gather(dim=1, \
                                                                index=topk_ind.unsqueeze(-1).repeat(1, 1,
                                                                                                    enc_outputs_coord_unact.shape[
                                                                                                        -1]))

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat(
                [denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(dim=1, \
                                                   index=topk_ind.unsqueeze(-1).repeat(1, 1,
                                                                                       enc_outputs_class.shape[-1]))

        # extract region features
        if self.learnt_init_query:
            target = self.tgt_embed.weight.unsqueeze(0).tile([bs, 1, 1])
        else:
            target = output_memory.gather(dim=1, \
                                          index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        return target, reference_points_unact.detach(), enc_topk_bboxes, enc_topk_logits

    def forward(self, encoder_feats, feats, targets=None):

        (memory, spatial_shapes, level_start_index, S_g_category) = self._get_encoder_input(encoder_feats, feats[2:])

        if self.training and self.num_denoising > 0 and targets is not None:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = \
                get_contrastive_denoising_training_group(targets, \
                                                         self.num_classes,
                                                         self.num_queries,
                                                         self.denoising_class_embed,
                                                         num_denoising=self.num_denoising,
                                                         label_noise_ratio=self.label_noise_ratio,
                                                         box_noise_scale=self.box_noise_scale)
        else:
            denoising_class, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        target, init_ref_points_unact, enc_topk_bboxes, enc_topk_logits = \
            self._get_decoder_input(memory, spatial_shapes, denoising_class, denoising_bbox_unact)

        # decoder
        out_bboxes, out_logits, hs = self.decoder(
            target,
            init_ref_points_unact,
            memory,
            spatial_shapes,
            level_start_index,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask)

        if self.training and dn_meta is not None:
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta['dn_num_split'], dim=2)
            dn_out_hs, out_hs = torch.split(hs[-1], dn_meta['dn_num_split'], dim=1) 

        out = {'pred_logits': out_logits[-1], 'pred_boxes': out_bboxes[-1], 'outputs': hs[-1], }
        
        out['hw_outputs'] = S_g_category

        if self.training and self.aux_loss:
            
            out['aux_outputs'] = self._set_aux_loss(out_logits[:-1], out_bboxes[:-1])
            out['aux_outputs'].extend(self._set_aux_loss([enc_topk_logits], [enc_topk_bboxes]))

            if self.training and dn_meta is not None:
                out['dn_aux_outputs'] = self._set_aux_loss(dn_out_logits, dn_out_bboxes)
                out['dn_meta'] = dn_meta
                out['outputs'] = out_hs

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class, outputs_coord)]

def build_rt_transformer(args):
    return RTDETRTransformer(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation ='gelu',
        num_decoder_layers=args.dec_layers,
        num_denoising=100,
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True
    )