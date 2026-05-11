__all__ = ['PatchTST_backbone']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

#from collections import OrderedDict
from layers.PatchTST_layers import *
from layers.RevIN import RevIN
import torch
import torch.nn as nn
import torch.nn.functional as F

class TrendFeatureExtractor(nn.Module):
    """
    提取稳定段的趋势特征，包括：
    - 均值
    - 标准差
    - 首末差（近似斜率）
    - 一阶差分的均值
    """
    def __init__(self):
        super(TrendFeatureExtractor, self).__init__()

    def forward(self, x):
        """
        x: [bs, 5, 1] — 5个稳定段带宽点
        return: [bs, 4] — 提取的统计特征
        """
        mean = x.mean(dim=1)                    # [bs, 1]
        std = x.std(dim=1)                      # [bs, 1]
        slope = (x[:, -1] - x[:, 0])            # [bs, 1]
        diff_mean = x[:, 1:] - x[:, :-1]        # [bs, 4, 1]
        diff_mean = diff_mean.mean(dim=1)       # [bs, 1]

        features = torch.cat([mean, std, slope, diff_mean], dim=-1)  # [bs, 4]
        return features

class ResidualPredictor(nn.Module):
    """
    预测 avg - label（残差），输入为：
    - 稳定段特征 [bs, 5, 1]
    - buffer duration Δt: [bs, 1]
    - 当前 label（带宽的最后一帧）: [bs, 1]
    """
    def __init__(self, hidden_dim=64):
        super(ResidualPredictor, self).__init__()
        self.feature_extractor = TrendFeatureExtractor()
        self.fusion = nn.Sequential(
            nn.Linear(4 + 1 + 1, hidden_dim),  # trend_feat + delta_t + label
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, delta_t):
        """
        x: [bs, 5, 1]
        delta_t: [bs, 1]
        """
        trend_feat = self.feature_extractor(x).to(x.device)   # [bs, 4]
        label = x[:, -1, :].to(x.device)                      # [bs, 1]
        fusion_input = torch.cat([trend_feat, delta_t, label], dim=-1).to(x.device)  # [bs, 6]
        residual = self.fusion(fusion_input).to(x.device)     # [bs, 1]
       # print(label.shape,residual.shape)
        avg_pred = label + residual             # 最终预测 avg

        return avg_pred, residual

# Cell


class ResidualPredictor(nn.Module):
    """
    输入:
    - x: [B * nvars, patch_num, d_model]
    - delta_t: [B, 1]
    输出:
    - avg_pred: [B * nvars, 1]
    """

    def __init__(self, d_model=128, hidden_dim=64):
        super(ResidualPredictor, self).__init__()
        self.trend_pool = nn.AdaptiveAvgPool1d(1)  # 将 patch_num 降维
        self.fusion = nn.Sequential(
            nn.Linear(6 + 1 + 1, hidden_dim),#6
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, delta_t):
        """
        x: [B * nvars, patch_num, d_model]
        delta_t: [B, 1]
        """
        Bn, P, D = x.shape
        nvars = 6  # 若你的 nvars 是 6，则 B = Bn // 6

        B = delta_t.shape[0]
        assert Bn % B == 0, "delta_t 不能正确广播到 x 的维度"

        # trend 特征 = 平均池化 patch
        trend_feat = x.mean(dim=1)  # [B * nvars, d_model]

        # 取 label：最后一帧的 embedding，假设即 x[:, -1, :]（也可以是原始 label 直接输入）
        label_feat = x#[:, -1, :]  # [B * nvars, d_model] 或可用独立输入

        # 扩展 delta_t
        delta_t_expanded = delta_t#.repeat_interleave(nvars, dim=0)  # [B * nvars, 1]

        # 简化：也可以只用 label scalar 而非 d_model
        label_scalar = label_feat.squeeze(-1)#.mean(dim=1, keepdim=True)  # [B * nvars, 1]
       # print(trend_feat.shape, delta_t_expanded.shape, label_scalar.shape)
        # 拼接融合向量
        fusion_input = torch.cat([trend_feat, delta_t_expanded, label_scalar], dim=-1)  # [B * nvars, d_model+2]
       # print(fusion_input.shape)
        # 预测残差
        residual = self.fusion(fusion_input)  # [B * nvars, 1]
        avg_pred = label_scalar + residual  # 可加回真实 label 值

        return avg_pred, residual
class PatchTST_backbone(nn.Module):
    def __init__(self, c_in:int, context_window:int, target_window:int, patch_len:int, stride:int, max_seq_len:Optional[int]=1024,
                 n_layers:int=3, d_model=128, n_heads=16, d_k:Optional[int]=None, d_v:Optional[int]=None,
                 d_ff:int=256, norm:str='BatchNorm', attn_dropout:float=0., dropout:float=0., act:str="gelu", key_padding_mask:bool='auto',
                 padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, pre_norm:bool=False, store_attn:bool=False,
                 pe:str='zeros', learn_pe:bool=True, fc_dropout:float=0., head_dropout = 0, padding_patch = None,
                 pretrain_head:bool=False, head_type = 'flatten', individual = False, revin = True, affine = True, subtract_last = False,
                 verbose:bool=False, **kwargs):

        super().__init__()

        # RevIn
        self.revin = revin
        if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)
        self.fea=TrendFeatureExtractor()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((context_window - patch_len)/stride + 1)
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        #print("patch_num",patch_num)
        #print("max_seq_len",max_seq_len)
        # Backbone
        self.backbone = TSTiEncoder(c_in, patch_num=patch_num, patch_len=patch_len, max_seq_len=max_seq_len,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var,
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, verbose=verbose, **kwargs)

        # Head
        self.head_nf = d_model * 3#patch_num
        self.n_vars = c_in
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self. d_model= d_model
        self.patch_num=patch_num
        self.individual = individual
        """
        self.weight_generator = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 6),  # 生成每个输出位置的权重
        )
        """
        self.weight_generator = nn.Sequential(
            nn.Linear(1, 32),
            nn.PReLU(),
           # nn.Dropout(0.1),
            nn.Linear(32, 6),  # 对应 patch_num6
        )
        #self.weight_generator = nn.Sequential(
         #   nn.Linear(1, 6),  # 明确输出 patch_num 维度
        #    nn.Softmax(dim=-1)  # 保证归一化为注意力权重
        #)
        self.res=ResidualPredictor()
        if self.pretrain_head:
            self.head = self.create_pretrain_head(self.head_nf, c_in, fc_dropout) # custom head passed as a partial func with all its kwargs
        elif head_type == 'flatten':
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def buffer_ratio_to_patch(self,buffer_ratio):
        patches = []
        for i in range(0, buffer_ratio.shape[1] - self.patch_len + 1, self.stride):
            patch = buffer_ratio[:, i:i + self.patch_len].mean(dim=1, keepdim=True)  # mean pooling or central frame
            patches.append(patch)
        return torch.cat(patches, dim=1)

    def forward(self, z,buffer_ratio):                                                                   # z: [bs x nvars x seq_len]
        # norm
       # print("buffer_shape",buffer_ratio.shape)
        #print("z", z.shape)
        #buffer_ratio=buffer_ratio.permute(0,2,1)
        # do patching
        #if self.padding_patch == 'end':
           # z = self.padding_patch_layer(z)
            #buffer_ratio=self.padding_patch_layer(buffer_ratio)
        #print("z0",z.shape)
        #z = torch.cat([z, buffer_ratio.to(z.device)], dim=1,).to(z.device)
        z=z.float()
        #model = DTAwareWeightedAveraging(input_dim=1)
        #w=z.permute(0,2,1).unsqueeze(-1)
        #t=torch.reshape(w, (w.shape[0] * w.shape[1], w.shape[2], w.shape[3]))
        #fe=torch.reshape(self.fea(t),(w.shape[0] ,w.shape[1], -1))
       # print("tt",t.shape)
        #print("z",fe.shape)
        #avg_pred = model(z, buffer_ratio)
        #print(avg_pred.shape)
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'norm')
            z = z.permute(0,2,1)
        #print("zf", z.shape)
       # z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # z: [bs x nvars x patch_num x patch_len]
        #print("z1",z.shape)
        #print("bufsh", buffer_ratio.shape)
        #buffer_ratio=buffer_ratio.unfold(dimension=-1, size=self.patch_len, step=self.stride)#.mean(dim=-1)
        z=z.unsqueeze(1)
       # print(z.shape)
        z = z.permute(0,1,3,2)
        #buffer_ratio=buffer_ratio.permute(0,1,3,2)# z: [bs x nvars x patch_len x patch_num]
        #print("z2", z.shape)
        # model
        #print("bufsh1",buffer_ratio.shape)
        #buffer_ratio=self.buffer_ratio_to_patch(buffer_ratio)

       # print("zf",z.shape)
        z = self.backbone(z,buffer_ratio)
       # print("z.s",z.shape)# z: [bs x nvars x d_model x patch_num]
       # print(self.n_vars,self.patch_num,self.d_model)
        #patch_num = z.shape[-1]
        #patch_num = z.shape[1]
        #B = z.shape[0]

       # 输入 [B, 1] → 输出 [B, patch_num]

       # weights = self.weight_generator(buffer_ratio)  # [B, patch_num]
       # print(weights.shape)
        #weights = weights.view(B, 6, 1, 8)
       # print(weights.shape)
        # reshape 权重为 [B, patch_num, 1, 1]
       # weights = weights.view(B, 6, 1, 1)
       # print(z.shape)
        # z: [B, patch_num, n   vars, d_model]
        #weighted_z = (z * weights)#.sum(dim=1)  # 按 patch 聚合 → [B, nvars, d_model]

        #z = weighted_z#.view(B, -1)

        #print("zzz", z.shape)
        z = self.head(z)      # z: [bs x nvars x target_window]

        #
        #z=torch.reshape(z,(16,-1,1))
       # z=z[:,:,]
        if self.revin:
            z = z.permute(0,2,1)
            z = self.revin_layer(z, 'denorm')
            z = z.permute(0,2,1)
        #print(buffer_ratio.shape,"b0 ",buffer_ratio)
        weights = self.weight_generator(buffer_ratio.to(z.device).float()).to(z.device).float()  # [bs, seq_len]
        #print(weights.shape,"w0 ",weights)
        weights = F.softmax(weights, dim=-1)  # [bs, seq_len]
        #print(weights.shape,"w1",weights)
        weights = weights.unsqueeze(-1)  # [bs, seq_len, 1]
        #print("w1", weights.shape)
       # print("ccc",weights.shape,z.shape)
        #print(weights.shape, "w2", weights)
        weighted_avg = (z * weights).sum(dim=1)  #        return z

        #print(z.shape,buffer_ratio.shape)
        #avg_pred,reds=self.res(z,buffer_ratio)
       # print("se",self.res(z,buffer_ratio))
        #print(weighted_avg.shape,"wf", weighted_avg)
        return weighted_avg
        #return avg_pred
        return z
        #return weighted_avg
    def create_pretrain_head(self, head_nf, vars, dropout):
        return nn.Sequential(nn.Dropout(dropout),
                    nn.Conv1d(head_nf, vars, 1)
                    )


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):                                 # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs x d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x




class TSTiEncoder(nn.Module):  #i means channel-independent
    def __init__(self, c_in, patch_num, patch_len, max_seq_len=1024,
                 n_layers=3, d_model=128, n_heads=16, d_k=None, d_v=None,
                 d_ff=256, norm='BatchNorm', attn_dropout=0., dropout=0., act="gelu", store_attn=False,
                 key_padding_mask='auto', padding_var=None, attn_mask=None, res_attention=True, pre_norm=False,
                 pe='zeros', learn_pe=True, verbose=False, **kwargs):


        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(6, d_model)   # patch_len, d_model    # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len
        self.res = ResidualPredictor()
        # Positional encoding
        self.W_pos = positional_encoding(pe, learn_pe, 3, d_model)#pe, learn_pe, q_len, d_model
        #print("d_model",d_model)
        self.buffer_embed = nn.Linear(patch_len, d_model).to("cuda:0")
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.show= nn.Linear(256,128)

        # Encoder
        self.encoder = TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout, dropout=dropout,
                                   pre_norm=pre_norm, activation=act, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x,buffer_ratio) -> Tensor:                                              # x: [bs x nvars x patch_len x patch_num]

        n_vars = x.shape[1]
        # Input encoding
        #x = x.permute(0,1,3,2)                                                   # x: [bs x nvars x patch_num x patch_len]
        #buffer_ratio = buffer_ratio.permute(0,1,3,2)
        #x = x.permute(0, 2, 3, 1)
        #print("x",x.shape)
        x = self.W_P(x)                                                          # x: [bs x nvars x patch_num x d_model]

        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))      # u: [bs * nvars x patch_num x d_model]
        #print("u.shape", u.shape)
        if buffer_ratio is not None:
            # [bs, patch_num] -> [bs, patch_num, 1] -> [bs, patch_num, d_model]
          #  print("buffer_ratio",buffer_ratio.shape)
           # print(buffer_ratio.device)
#            print(self.buffer_embed.device)
            device = x.device
            #buffer_ratio = buffer_ratio.to(device).float()
            #buffer_embed= self.W_P(buffer_ratio)
           # buffer_embed = self.buffer_embed(buffer_ratio)
            #print("buffer_embed.shape1", buffer_embed.shape)
            #buffer_embed = buffer_embed.repeat_interleave(5, dim=1)  # 需要与 unfold 后 patch_num 对齐
            #print("buffer_embed.shape2",buffer_embed.shape)
            # 由于 u 是 [bs * nvars, patch_num, d_model]，我们需要调整 buffer_embed 的维度
            #buffer_embed = buffer_embed.unsqueeze(1).repeat(1,  n_vars, 1, 1)  # [bs, nvars, patch_num, d_model]
            #print("buffer_embed.shape3", buffer_embed.shape)
            #buffer_embed = buffer_embed.repeat(1, n_vars, 1, 1)
           # print("buffer_embed.shape3", buffer_embed.shape)

#            buffer_embed = buffer_embed.view(u.shape)  # [bs * nvars, patch_num, d_model]
            #u = self.dropout(u + self.W_pos)
            #x = torch.cat([time_token, patch_embedding], dim=1)
            #buffer_embed = buffer_embed.unsqueeze(1)
           # print("buffer",buffer_embed.shape,u.shape)
            #print(self.W_pos.shape)
            u = self.dropout(u + self.W_pos)
           # u = self.dropout(self.norm(u + self.W_pos + buffer_embed))

        else:
            u = self.dropout(u + self.W_pos)


        #u = self.dropout(u + self.W_pos)                                         # u: [bs * nvars x patch_num x d_model]
        #print("encoder",u.shape)
        # Encoder
        #print(buffer_ratio)
        #uz,reds=self.res(u,buffer_ratio)
        #print("uz",uz.shape)
        z = self.encoder(u)                                                      # z: [bs * nvars x patch_num x d_model]
        #print("zz",z.shape)
        #z=torch.reshape(z, (16, -1, 1))

        #z = torch.cat([z, buffer_embed], dim=-1)
        #z = self.show(z)
        #print("zz", z.shape)
        z = torch.reshape(z, (-1,n_vars,z.shape[-2],z.shape[-1]))                # z: [bs x nvars x patch_num x d_model]
        z = z.permute(0,1,3,2)                                                   # z: [bs x nvars x d_model x patch_num]

        return z



# Cell
class TSTEncoder(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                        norm='BatchNorm', attn_dropout=0., dropout=0., activation='gelu',
                        res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()

        self.layers = nn.ModuleList([TSTEncoderLayer(q_len, d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention

    def forward(self, src:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers: output, scores = mod(output, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output
        else:
            for mod in self.layers: output = mod(output, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            return output



class TSTEncoderLayer(nn.Module):
    def __init__(self, q_len, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='BatchNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                get_activation_fn(activation),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, src:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(src, src, src, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            src2, attn = self.self_attn(src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(src2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores
        else:
            return src




class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        """Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x d_model]
            K, V:    [batch_size (bs) x q_len x d_model]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)       # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)     # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)       # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    r"""Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)"""

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = torch.matmul(q, k) * self.scale      # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:                                     # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:                              # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)                 # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)                        # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

