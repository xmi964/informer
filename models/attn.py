import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import pad as F_pad

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask, peak_mask=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        
        scale = self.scale or 1./np.sqrt(D)
        
        scores = torch.einsum("blhd,bshd->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask==True, -np.inf)
            scores = self.dropout(torch.softmax(scale * scores, dim=-1))
        else:
            scores = self.dropout(torch.softmax(scale * scores, dim=-1))
            
        V = torch.einsum("bhls,bshd->blhd", scores, values)
        
        if self.output_attention:
            return V.contiguous(), scores
        else:
            return V.contiguous(), None

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, 
                 output_attention=False, custom_attention='prob_sparse', d_model=512, n_heads=8,
                 peak_threshold=3.0):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.custom_attention = custom_attention
        self.peak_threshold = peak_threshold
        self.n_heads = n_heads
        self.d_model = d_model

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_Q, D = Q.shape
        B, H, L_K, D = K.shape
        
        # 计算采样索引
        K_expand = K.unsqueeze(2).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # [L_Q, sample_k]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(3), K_sample.transpose(3, 4)).squeeze(3)
        
        # 筛选 Top-k
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]  # [B, H, u]
        
        # 修正 Q_reduce 的维度：[B, H, u, D]
        Q_reduce = Q[
            torch.arange(B)[:, None, None],  # [B, 1, 1]
            torch.arange(H)[None, :, None],   # [1, H, 1]
            M_top,                           # [B, H, u]
            :                                # [D]
        ]
        
        # 计算 scores_top：[B, H, u, L_K]
        scores_top = torch.matmul(Q_reduce, K.transpose(2, 3))
        
        return scores_top, M_top

    def _update_context(self, context_in, values, scores, index, L_Q):
        B, H, L_V, D = values.shape
        
        # 确保 index 是 [B, H, u]
        if index.dim() == 2:  # 如果 index 是 [B, u]，扩展成 [B, H, u]
            index = index.unsqueeze(1).expand(-1, H, -1)
        
        # 修正 values_selected 的维度：[B, H, u, D]
        values_selected = values[
            torch.arange(B)[:, None, None],  # [B, 1, 1]
            torch.arange(H)[None, :, None],  # [1, H, 1]
            index,                           # [B, H, u]
            :                                # [D]
        ]
        
        # 修正 attn 的维度：[B, H, u, L_K]
        attn = torch.softmax(scores, dim=-1)
        
        # 矩阵乘法：[B, H, u, L_K] @ [B, H, L_K, D] → [B, H, u, D]
        context_update = torch.matmul(attn, values)
        
        # 赋值到 context_in
        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index.sort(dim=-1)[0]
        ] = context_update
        
        return context_in, attn if self.output_attention else None

    def forward(self, queries, keys, values, attn_mask=None, peak_mask=None):
        B, L_Q, H, D = queries.shape
        B, L_K, H, D = keys.shape

        # 添加维度检查
        assert queries.size(3) == keys.size(3), "Query and Key feature dimensions must match"

        # 维度调整：[B, L, H, D] → [B, H, L, D]
        queries = queries.transpose(1, 2).contiguous()
        keys = keys.transpose(1, 2).contiguous()
        values = values.transpose(1, 2).contiguous()

        # 异常检测逻辑（可选）
        if self.custom_attention == 'anomaly' and peak_mask is None:
            seq_mean = queries.mean(dim=2, keepdim=True)
            seq_std = queries.std(dim=2, keepdim=True)
            peak_mask = ((queries - seq_mean) > self.peak_threshold * seq_std).float()

        # 计算采样数
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # 概率注意力核心逻辑
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        # 输出维度调整：[B, H, L_Q, D] → [B, L_Q, H, D]
        return context.transpose(1, 2).contiguous(), attn

    def _get_initial_context(self, values, L_Q):
        B, H, L_V, D = values.shape
        if not self.mask_flag:
            values_sum = values.sum(dim=2, keepdim=True) / L_V
            return values_sum.expand(B, H, L_Q, D).clone()
        else:
            return torch.zeros(B, H, L_Q, D).to(values.device)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None, peak_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            peak_mask  # 传递peak_mask到内部注意力
        )
        
        if self.mix:
            out = out.transpose(2,1).contiguous()
        
        out = self.out_projection(out.view(B, L, -1))
        
        return out, attn