import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                 out_channels=c_in,
                                 kernel_size=3,
                                 padding=padding,
                                 padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # 输入x形状应为[B, L, C]
        if x.dim() == 4:  # 如果输入是[B, L, H, D]
            B, L, H, D = x.shape
            x = x.reshape(B, L, -1)  # 合并最后两个维度
        
        # 维度转换 [B, L, C] -> [B, C, L]
        x = x.permute(0, 2, 1)
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # 转换回 [B, L, C]
        return x.permute(0, 2, 1)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # 输入维度检查
        original_shape = x.shape
        if x.dim() == 4:  # [B, L, H, D]
            B, L, H, D = x.shape
            x = x.reshape(B, L, -1)  # 合并最后两个维度
        
        # Attention处理
        new_x, attn = self.attention(
            queries=x,
            keys=x,
            values=x,
            attn_mask=attn_mask
        )
        
        # 残差连接
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        
        # 卷积处理
        y = y.transpose(-1, 1)  # [B, L, D] -> [B, D, L]
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(-1, 1)  # [B, D, L] -> [B, L, D]
        
        # 恢复原始形状
        if len(original_shape) == 4:
            y = y.reshape(original_shape)
        
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # 输入维度预处理
        if x.dim() == 4:  # [B, L, H, D]
            B, L, H, D = x.shape
            need_reshape = True
        else:
            need_reshape = False
            
        attns = []
        for i, attn_layer in enumerate(self.attn_layers):
            if need_reshape:
                x = x.reshape(B, L, -1)  # 合并最后两个维度
                
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
            
            if self.conv_layers is not None and i < len(self.conv_layers):
                if need_reshape:
                    x = x.reshape(B, -1, H, D)  # 恢复多头形状
                x = self.conv_layers[i](x)
                L = x.shape[1]  # 更新序列长度
                
        if self.norm is not None:
            x = self.norm(x)
            
        return (x, attns)

class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super(EncoderStack, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        x_stack = []
        attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1] // (2 ** i_len)
            x_s, attn = encoder(
                x[:, -inp_len:, :],
                attn_mask=attn_mask
            )
            x_stack.append(x_s)
            attns.append(attn)
        return torch.cat(x_stack, -2), attns