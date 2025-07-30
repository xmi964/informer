import math
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
    def __init__(self, mask_flag=True, factor=5, scale=None,
                attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)  # 修正：使用小括号

    def _prob_QK(self, Q, K, sample_k, n_top):
        """优化后的概率QK计算"""
        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # 采样K
        K_expand = K.unsqueeze(2).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # [L_Q, sample_k]
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(3), K_sample.transpose(3, 4)).squeeze(3)

        # Top-k筛选
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]  # [B, H, u]
        return Q[torch.arange(B)[:, None, None], 
                torch.arange(H)[None, :, None],
                M_top], K.transpose(2, 3)  # scores_top: [B, H, u, L_K]

    def _update_context(self, context_in, values, scores, index, L_Q):
        # ==================== 1. 输入验证 ====================
        B, H, L_V, D = values.shape
        
        # 类型检查
        if not isinstance(index, torch.Tensor):
            raise TypeError(f"index必须是Tensor, 实际是{type(index)}")
        
        # 维度验证
        assert scores.dim() == 4, f"scores应为4D Tensor, 实际是{scores.dim()}D"
        assert index.dim() in (3,4), f"index应为3D/4D Tensor, 实际是{index.dim()}D"
        
        # ==================== 2. 预处理（核心修改：克隆context_in） ====================
        # 关键修复：克隆context_in，避免内存共享导致的修改冲突
        context_in = context_in.clone()  # ← 新增克隆操作，切断内存共享
        
        # 统一设备
        device = values.device
        context_in = context_in.to(device)
        scores = scores.to(device)
        index = index.to(device)
        
        # 索引处理
        if index.dim() == 4:
            index = index[:, :, :, 0]  # 降维到[B, H, u]
        index = index.long()  # 确保是整数索引
        
        # 获取实际要更新的位置数
        u = index.shape[-1] if index.dim() > 0 else 0
        
        # ==================== 3. 注意力计算 ====================
        # 强制对齐维度
        min_len = min(scores.shape[-1], L_V)
        scores = scores[..., :min_len]
        values = values[..., :min_len, :]
        
        # 计算注意力权重
        with torch.no_grad():
            attn = torch.softmax(scores, dim=-1)  # [B, H, u, L_V]
        
        # ==================== 4. 安全的矩阵乘法 ====================
        try:
            # 使用einsum确保维度正确
            context_update = torch.einsum('bhuv,bhvd->bhud', attn, values)  # [B, H, u, D]
            
            # 维度修复 (处理u不匹配的情况)
            current_u = context_update.shape[2]
            if current_u != u:
                if current_u < u:  # 不足时补零
                    pad_size = u - current_u
                    pad = torch.zeros(B, H, pad_size, D, device=device)
                    context_update = torch.cat([context_update, pad], dim=2)
                else:  # 超长时截断
                    context_update = context_update[:, :, :u, :]
        except RuntimeError as e:
            print(f"❌ 矩阵乘法失败 - 形状: attn{attn.shape} * values{values.shape}")
            print(f"❌ 目标输出形状: [B={B}, H={H}, u={u}, D={D}]")
            raise RuntimeError("注意力计算失败，请检查输入维度") from e
        
        # ==================== 5. 安全的上下文更新（优化索引处理） ====================
        # 创建广播友好的索引（兼容不同批次和头数）
        b_idx = torch.arange(B, device=device).view(B, 1, 1)  # [B,1,1]
        h_idx = torch.arange(H, device=device).view(1, H, 1)  # [1,H,1]
        
        # 确保索引不越界
        max_idx = context_in.shape[2] - 1  # context_in的时间维度最大索引
        index = torch.clamp(index, 0, max_idx)  # 截断到合法范围
        
        try:
            # 安全赋值：使用索引广播更新
            context_in[b_idx, h_idx, index] = context_update
        except RuntimeError as e:
            print(f"❌ 更新失败 - 维度信息:")
            print(f"  context_in形状: {context_in.shape}, 索引范围: [0, {max_idx}]")
            print(f"  index形状: {index.shape}, 内容: {index[:2, :2, :2]}（前2个元素）")
            print(f"  context_update形状: {context_update.shape}")
            raise RuntimeError("上下文更新失败，请检查索引和维度") from e
        
        return context_in, (attn if self.output_attention else None)

    def forward(self, queries, keys, values, attn_mask=None):
        """优化后的前向传播"""
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # 维度检查
        if queries.size(-1) != keys.size(-1):
            raise ValueError(
                f"Query dim {queries.size(-1)} != Key dim {keys.size(-1)}")

        # 维度调整 [B, L, H, D] -> [B, H, L, D]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 概率注意力计算（使用math.ceil）
        U_part = min(self.factor * math.ceil(math.log(L_K)), L_K)
        u = min(self.factor * math.ceil(math.log(L_Q)), L_Q)
        
        scores_top, index = self._prob_QK(queries, keys, U_part, u)
        context = self._get_initial_context(values, L_Q)  # 使用初始化方法
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.transpose(1, 2).contiguous(), attn

    def _get_initial_context(self, values, L_Q):
        """已验证正确的上下文初始化方法"""
        B, H, L_V, D = values.shape
        if not self.mask_flag:
            # 非掩码模式：用values的均值初始化
            return values.mean(dim=2, keepdim=True).expand(B, H, L_Q, D)
        else:
            # 掩码模式：零初始化
            return torch.zeros(B, H, L_Q, D).to(values.device)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # 1. 线性投影 + 头部分解
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        # 2. 注意力计算（简化接口）
        out, attn = self.inner_attention(
            queries=queries,
            keys=keys,
            values=values,
            attn_mask=attn_mask
        )
        
        # 3. 头合并 + 输出投影
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = self.out_projection(out.view(B, L, -1))
        
        return out, attn