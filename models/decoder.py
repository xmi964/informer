import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, peak_mask=None):
        # 1. 自注意力层（处理 peak_mask）
        x1, _ = self.self_attention(
            x, x, x,  # q, k, v
            attn_mask=x_mask,
            peak_mask=peak_mask  # 传递异常标记
        )
        x = self.norm1(x + self.dropout(x1))  # 残差连接和层归一化

        # 2. 交叉注意力层（不处理 peak_mask，只关注编码器输出）
        x2, _ = self.cross_attention(
            x, cross, cross,  # q, k, v
            attn_mask=cross_mask
        )
        x = self.norm2(x + self.dropout(x2))  # 残差连接和层归一化

        # 3. 前馈网络
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))
        x = self.norm3(x + y)  # 残差连接和层归一化

        return x

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, custom_decoder=False, d_model=512):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.custom_decoder = custom_decoder
        
        if self.custom_decoder:
            # 增加峰值专用前馈层，用于增强异常点的解码能力
            self.peak_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(0.1)
            )

    def forward(self, x, cross, x_mask=None, cross_mask=None, peak_mask=None):
        # 1. 长度对齐处理（确保 peak_mask 与输入序列长度匹配）
        if peak_mask is not None:
            if peak_mask.size(1) != x.size(1):
                print(f"[Decoder] 自动调整peak_mask: {peak_mask.shape} -> x_len={x.size(1)}")
                # 截断或填充 peak_mask 以匹配输入长度
                if peak_mask.size(1) > x.size(1):
                    peak_mask = peak_mask[:, :x.size(1)]  # 截断
                else:
                    # 填充（通常不会发生，因为解码器输入长度 <= 编码器输入长度）
                    pad_len = x.size(1) - peak_mask.size(1)
                    peak_mask = torch.cat([peak_mask, torch.zeros(peak_mask.size(0), pad_len).to(peak_mask.device)], dim=1)

            # 保存原始输入用于后续异常增强
            x_original = x

        # 2. 逐层处理
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, peak_mask=peak_mask)

        # 3. 应用层归一化
        if self.norm is not None:
            x = self.norm(x)

        # 4. 自定义解码器：对异常点进行额外处理
        if self.custom_decoder and peak_mask is not None:
            # 扩展 peak_mask 维度以匹配特征维度
            peak_mask_expanded = peak_mask.unsqueeze(-1)  # [B,L] -> [B,L,1]
            
            # 对异常点应用专用前馈网络
            peak_features = self.peak_ffn(x_original)
            
            # 合并正常特征和异常增强特征
            x = x + peak_mask_expanded * peak_features

        return x