import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                dropout=0.1, attn='prob', embed='timeF', freq='h', activation='gelu',
                output_attention=False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.attn = attn
        self.output_attention = output_attention

        # === 1. Enhanced Embedding ===
        self.enc_embedding = DataEmbedding(
            enc_in, d_model, embed, freq, dropout
        )
        self.dec_embedding = DataEmbedding(
            dec_in, d_model, embed, freq, dropout
        )

        # === 2. Optimized Attention ===
        Attn = ProbAttention if attn=='prob' else FullAttention
        
        # === 3. Encoder with Simplified Structure ===
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        Attn(False, factor, dropout, output_attention),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [ConvLayer(d_model) for l in range(e_layers-1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model)
        )

        # === 4. Decoder with Output Activation ===
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        Attn(True, factor, dropout, False),
                        d_model, n_heads, mix=mix
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, dropout, False),
                        d_model, n_heads, mix=False
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        # === 5. Enhanced Projection ===
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, c_out))
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        # === Encoder Processing ===
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        # === Decoder Processing ===
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask
        )
        
        # === Final Projection ===
        dec_out = self.projection(dec_out)
        
        return dec_out[:, -self.pred_len:, :] if not self.output_attention else (
            dec_out[:, -self.pred_len:, :], attns)

class InformerStack(Informer):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2,
                d_ff=512, dropout=0.1, attn='prob', embed='timeF', freq='h',
                activation='gelu', output_attention=False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        
        super(InformerStack, self).__init__(
            enc_in, dec_in, c_out, seq_len, label_len, out_len,
            factor, d_model, n_heads, e_layers[0], d_layers, d_ff,
            dropout, attn, embed, freq, activation,
            output_attention, distil, mix, device
        )
        
        # === Stacked Encoder Modification ===
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            ProbAttention(False, factor, dropout, output_attention),
                            d_model, n_heads, mix=False
                        ),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for _ in range(el)
                ],
                [ConvLayer(d_model) for _ in range(el-1)] if distil else None,
                norm_layer=nn.LayerNorm(d_model)
            )for el in e_layers]
        
        self.encoder = EncoderStack(encoders, list(range(len(e_layers))))