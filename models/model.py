import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention=False, distil=True, mix=True,
                device=torch.device('cuda:0'), custom_attention=None,
                peak_threshold=3.0):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.custom_attention = custom_attention
        self.peak_threshold = peak_threshold

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        if custom_attention == 'anomaly':
            self._configure_anomaly_attention(Attn)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                            custom_attention=self.custom_attention,
                            d_model=d_model,
                            n_heads=n_heads
                        ), 
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(
                            mask_flag=True,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=False,
                            custom_attention=self.custom_attention,
                            d_model=d_model,
                            n_heads=n_heads
                        ),
                        d_model, n_heads, mix=mix
                    ),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)
    
    def _configure_anomaly_attention(self, Attn):
        if isinstance(Attn, ProbAttention):
            for layer in self.encoder.attn_layers:
                if hasattr(layer, 'inner_attention') and isinstance(layer.inner_attention, ProbAttention):
                    layer.inner_attention.custom_attention = 'anomaly'
                    layer.inner_attention.peak_threshold = self.peak_threshold

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None,
                enc_peak_mask=None, dec_peak_mask=None):
        
        # Encoder
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        
        # Handle peak mask for encoder
        if enc_peak_mask is not None:
            # Ensure peak_mask matches the sequence length after embedding
            if enc_peak_mask.size(1) != enc_out.size(1):
                enc_peak_mask = F.interpolate(
                    enc_peak_mask.unsqueeze(1).float(),
                    size=enc_out.size(1),
                    mode='linear',
                    align_corners=False
                ).squeeze(1).to(enc_out.dtype)
        
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask, peak_mask=enc_peak_mask)

        # Decoder
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        
        # Handle peak mask for decoder
        if dec_peak_mask is not None:
            if dec_peak_mask.size(1) != dec_out.size(1):
                dec_peak_mask = F.interpolate(
                    dec_peak_mask.unsqueeze(1).float(),
                    size=dec_out.size(1),
                    mode='linear',
                    align_corners=False
                ).squeeze(1).to(dec_out.dtype)
        
        dec_out = self.decoder(
            dec_out, enc_out, 
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            peak_mask=dec_peak_mask
        )
        
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

class InformerStack(Informer):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention=False, distil=True, mix=True,
                device=torch.device('cuda:0'), custom_attention=None,
                peak_threshold=3.0):
        super(InformerStack, self).__init__(
            enc_in, dec_in, c_out, seq_len, label_len, out_len, 
            factor, d_model, n_heads, e_layers[0], d_layers, d_ff, 
            dropout, attn, embed, freq, activation,
            output_attention, distil, mix,
            device, custom_attention, peak_threshold
        )
        
        # Stacked Encoder
        inp_lens = list(range(len(e_layers)))
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            ProbAttention(
                                mask_flag=False,
                                factor=factor,
                                attention_dropout=dropout,
                                output_attention=output_attention,
                                custom_attention=self.custom_attention,
                                d_model=d_model,
                                n_heads=n_heads
                            ), 
                            d_model, n_heads, mix=False
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)