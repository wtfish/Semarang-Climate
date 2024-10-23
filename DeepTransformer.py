import torch
from torch import nn, einsum
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

def full_attention(query, key, value, causal=False, dropout=0.0):
    device = key.device
    B_k, h_k, n_k, d_k = key.shape
    B_q, h_q, n_q, d_q = query.shape

    scale = einsum("bhqd,bhkd->bhqk", query, key)/math.sqrt(d_k)

    if causal:
        ones = torch.ones(B_k, h_k, n_q, n_k).to(device)
        mask = torch.tril(ones)
        scale = scale.masked_fill(mask == 0, -1e9)
    atn = F.softmax(scale, dim=-1)
    if dropout is not None:
        atn = F.dropout(atn, p=dropout)
    out = einsum("bhqk,bhkd->bhqd", atn, value)
    return out
def to_eachhead(x, head_num, split_num=3):
    B, n, pre_d = x.shape
    new_d = pre_d//split_num
    assert pre_d%split_num == 0, f"have to be multiple of {split_num}"
    assert new_d%head_num == 0, "dim must be divided by head_num"

    tpl = torch.chunk(x, split_num, dim=2)
    out = []
    for t in tpl:
        out.append(t.reshape(B, n, head_num, new_d//head_num).transpose(1,2))
    return out
def concat_head(x):
    B, h, n, _d = x.shape
    out = x.transpose(1,2).reshape(B, n, _d*h)
    return out
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, #dropout: float = 0.1,
                 max_len: int = 100):
        super().__init__()
        # self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.squeeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        # return self.dropout(x)
        return x

class PreLayer(nn.Module):
    def __init__(self, hid, d_model, drop_out=0.0, in_dim=3):
        super().__init__()
        self.linear = nn.Linear(in_dim, d_model)
        self.linear1 = nn.Linear(in_dim, hid, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hid, d_model, bias=True)


    def forward(self, x):
        out = self.linear(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
class PostLayer(nn.Module):
    def __init__(self, dim, vocab_num, hid_dim, dropout_ratio):
        super().__init__()
        # self.linear = nn.Linear(dim, vocab_num)
        self.linear1 = nn.Linear(dim, hid_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, vocab_num, bias=True)
    def forward(self,x):
        # out = self.linear(x)
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, head_num):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim*3)
        self.make_head = partial(to_eachhead, head_num=head_num, split_num=3)
        self.mhsa = full_attention

    def forward(self, x):
        qvk = self.to_qvk(x)
        q, v, k = self.make_head(qvk)
        out = self.mhsa(q, k, v)
        out = concat_head(out)
        return out
class MultiHeadCausalAttention(nn.Module):
    def __init__(self, dim, head_num):
        super().__init__()
        self.to_qvk = nn.Linear(dim, dim*3)
        self.make_head = partial(to_eachhead, head_num=head_num, split_num=3)
        self.mhca = partial(full_attention, causal=True)

    def forward(self, x):
        qvk = self.to_qvk(x)
        q, v, k = self.make_head(qvk)
        out = self.mhca(q, k, v)
        out = concat_head(out)
        return out
class MultiHeadSourceAttention(nn.Module):
    def __init__(self, dim, head_num):
        super().__init__()
        self.to_kv = nn.Linear(dim, dim*2)
        self.to_q = nn.Linear(dim, dim)
        self.make_head_kv = partial(to_eachhead, head_num=head_num, split_num=2)
        self.make_head_q = partial(to_eachhead, head_num=head_num, split_num=1)
        self.mhsa = full_attention

    def forward(self, x, memory):
        mem = self.to_kv(memory)
        x = self.to_q(x)
        k, v = self.make_head_kv(mem)
        q = self.make_head_q(x)[0]
        out = self.mhsa(q, k, v)
        out = concat_head(out)
        return out
class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hid_dim, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hid_dim, dim, bias=True)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
class EncoderLayer(nn.Module):
    def __init__(self, dim, head_num, ff_hidnum, dropout_ratio, norm_first=False):
        super().__init__()
        self.dor = dropout_ratio
        self.mhsa = MultiHeadSelfAttention(dim, head_num)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidnum)
        self.ln2 = nn.LayerNorm(dim)
        self.norm_first = norm_first

    def forward(self, x):
        res = torch.clone(x)

        if self.norm_first:
          out = self.ln1(x)
          out = self.mhsa(out)
          out = F.dropout(out, p=self.dor) + res

          res = torch.clone(out)
          out = self.ln2(out)
          out = self.ff(out)
          out = F.dropout(out, p=self.dor) + res
        else:
          out = self.mhsa(x)
          out = F.dropout(out, p=self.dor) + res
          out = self.ln1(out)

          res = torch.clone(out)
          out = self.ff(out)
          out = F.dropout(out, p=self.dor) + res
          out = self.ln2(out)

        return out
class Encoder(nn.Module):
    def __init__(self, depth, dim, head_num, ff_hidnum=2048, dropout_ratio=0.2, norm_first=False):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim, head_num, ff_hidnum, dropout_ratio, norm_first) for i in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, dim, head_num, ff_hidnum, dropout_ratio, norm_first=False):
        super().__init__()
        self.dor = dropout_ratio
        self.mhca = MultiHeadCausalAttention(dim, head_num)
        self.ln1 = nn.LayerNorm(dim)
        self.mhsa = MultiHeadSourceAttention(dim, head_num)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_hidnum)
        self.ln3 = nn.LayerNorm(dim)
        self.norm_first = norm_first

    def forward(self, x, memory):
        res = torch.clone(x)

        if self.norm_first:
          out = self.ln1(x)
          out = self.mhca(out)
          out = F.dropout(out, p=self.dor) + res

          res = torch.clone(out)
          out = self.ln2(out)
          out = self.mhsa(out, memory)
          out = F.dropout(out, p=self.dor) + res

          res = torch.clone(out)
          out = self.ln3(out)
          out = self.ff(out)
          out = F.dropout(out, p=self.dor) + res

        else:
          out = self.mhca(x)
          out = F.dropout(out, p=self.dor) + res
          out = self.ln1(out)

          res = torch.clone(out)
          out = self.mhsa(out, memory)
          out = F.dropout(out, p=self.dor) + res
          out = self.ln2(out)

          res = torch.clone(out)
          out = self.ff(out)
          out = F.dropout(out, p=self.dor) + res
          out = self.ln3(out)

        return out
class Decoder(nn.Module):
    def __init__(self, depth, dim, head_num, ff_hidnum, dropout_ratio=0.2, norm_first=False):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(dim, head_num, ff_hidnum, dropout_ratio, norm_first) for i in range(depth)])

    def forward(self, x, memory):
        for layer in self.layers:
            x = layer(x, memory)
        return x
class Transformer(nn.Module):
    def __init__(self, device, d_model, in_dim, N_enc, N_dec, h_enc, h_dec, ff_hidnum, hid_pre, hid_post, dropout_pre, dropout_post, dropout_model, norm_first=False):
        super().__init__()
        self.device = device
        self.x_pre = PreLayer(hid_pre, d_model, dropout_pre, in_dim)
        self.y_pre = PreLayer(hid_pre, d_model, dropout_pre, in_dim)
        self.pos = PositionalEncoding(d_model)
        self.enc = Encoder(N_enc,d_model, h_enc, ff_hidnum, dropout_model, norm_first)
        self.dec = Decoder(N_dec,d_model, h_dec, ff_hidnum, dropout_model, norm_first)
        self.post = PostLayer(d_model, 1, hid_post, dropout_post)

    def forward(self, x, y):
        x_emb = self.x_pre(x)
        y_emb = self.y_pre(y)
        x_emb_pos = self.pos(x_emb)
        y_emb_pos = self.pos(y_emb)
        memory = self.enc(x_emb_pos)
        out = self.dec(y_emb_pos, memory)
        out = self.post(out)
        out = out.squeeze(-1)
        return out

    def generate(self, x, forcast_step, y_start,multivariate=False):
        device = x.device
        x = x.to(device)
        B, N, D = x.shape
        x = self.x_pre(x) 
        x = self.pos(x)
        z = self.enc(x) 
        y = y_start
        for i in range(forcast_step):
            y_pred = self.y_pre(y)
            y_pred = self.pos(y_pred)
            y_pred = self.dec(y_pred, z)
            y_pred = self.post(y_pred)
            if multivariate:
                y = torch.cat([y, y_pred[:,[-1],:]], dim=2)
            else:
                y = torch.cat([y, y_pred[:,[-1],:]], dim=1)
        y_pred = y_pred.squeeze(-1)
        return y_pred

    # def generate(self, x, forecast_step, y_start, multivariate=False):
    #     device = x.device
    #     x = x.to(device)
    #     B, N, D = x.shape  # B = batch size, N = sequence length, D = feature dimension
    #     x = self.x_pre(x)
    #     x = self.pos(x)
    #     z = self.enc(x)
    #     y = y_start

    #     for i in range(forecast_step):
    #         y_pred = self.y_pre(y)
    #         y_pred = self.pos(y_pred)
    #         y_pred = self.dec(y_pred, z)
    #         y_pred = self.post(y_pred)

    #         # Ensure proper concatenation for multivariate
    #         if multivariate:
    #             # Ensure y_pred has the same feature dimension as y
    #             y_pred = y_pred.view(B, 1, D)  # Reshape y_pred to match the feature dimension of y
    #             y = torch.cat([y, y_pred[:, -1:, :]], dim=1)  # Concatenate along the sequence length
    #         else:
    #             y_pred = y_pred.view(B, 1, -1)  # Reshape y_pred to match the feature dimension of y
    #             y = torch.cat([y, y_pred[:, -1:, :]], dim=1)  # Concatenate along the sequence length

    #         # Debug: Print shapes at each iteration
    #         print(f"Iteration {i}: y shape = {y.shape}, y_pred shape = {y_pred.shape}")

    #     y_pred = y_pred.squeeze(-1)  # Remove last dimension if it's of size 1
    #     return y_pred

