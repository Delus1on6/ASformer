import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer, CrossAttentionLayer
from layers.Embed import DataEmbedding_inverted
import math
import numpy as np
from collections import Counter
import sys

class Model(nn.Module):

    def __init__(self, configs, seg_num):
        super(Model, self).__init__()
        self.flag = configs.flag
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, seg_num, configs.embed,
                                                    configs.freq, configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    CrossAttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads,
                        seg_num),
                    configs.d_model,
                    seg_num,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model * seg_num)
        )
        self.projector = nn.Linear(configs.d_model * seg_num, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        if self.flag == 0:
            seg_num = self.calculate_gcd_period_mean(x_enc)
            with open('seg_num.txt', 'w') as file:
                file.write(f"{seg_num}\n")
            sys.exit()

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def calculate_gcd_period_mean(self, x):  
        batch_size, time_steps, num_variates = x.shape
        # all_batch_gcds = []

        for batch in range(batch_size):
            batch_periods = []
            for variate in range(num_variates):
                variate_data = x[batch, :, variate]
                variate_data_without_dc = variate_data - torch.mean(variate_data)
               
                X = torch.fft.fft(variate_data_without_dc)
                magnitude_spectrum = torch.abs(X)
                peak_indices = torch.argsort(magnitude_spectrum, dim=0)[-3:] 
                valid_periods = []
                for peak_idx in peak_indices:
                    if peak_idx != 0:
                        period_estimate = time_steps / peak_idx.item()
                        valid_periods.append(period_estimate)
                if valid_periods:
                    average_period = sum(valid_periods) / len(valid_periods)
                    batch_periods.append(average_period)

            batch_periods = Counter(batch_periods)
            counts = 3
            batch_periods = [num for num in batch_periods if batch_periods[num] >= counts]
            
            if batch_periods:
                current_batch_gcd = int(math.floor(batch_periods[0]))
                for period in batch_periods[1:]:
                    current_batch_gcd = math.gcd(current_batch_gcd, int(math.floor(period)))
                all_batch_gcds = current_batch_gcd
                break
        return all_batch_gcds

        

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
