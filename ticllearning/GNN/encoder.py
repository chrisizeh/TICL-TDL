

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_dropout=0.0, use_qk_norm=True, tau=0.5):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_qk_norm = use_qk_norm  # cosine-style logits if True
        self.tau = tau                  # temperature for cosine attention

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_dropout)

        # Safer init
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # Small output scale (helps early training)
        with torch.no_grad():
            self.W_o.weight.mul_(0.5)

    def split_heads(self, x):
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # [B,H,T,Dk]

    def combine_heads(self, x):
        B, H, T, Dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dk)

    def scaled_dot_product_attention(self, Q, K, V, mask: Optional[torch.Tensor]=None):
        # Compute logits in float32 for stability
        Qf = Q.float()
        Kf = K.float()

        if self.use_qk_norm:
            # Cosine-style: bound logits; very stable with GNNs/AMP
            Qf = F.normalize(Qf, dim=-1)
            Kf = F.normalize(Kf, dim=-1)
            logits = torch.matmul(Qf, Kf.transpose(-2, -1)) / self.tau  # [-1,+1]/tau
        else:
            logits = torch.matmul(Qf, Kf.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Expect True/1 = keep, False/0 = mask (if you ever use it)
            if mask.dtype != torch.bool:
                mask = mask != 0
            # Expand to [B,H,Tq,Tk] if needed
            while mask.dim() < logits.dim():
                mask = mask.unsqueeze(1)
            if mask.size(1) == 1 and logits.size(1) > 1:
                mask = mask.expand(-1, logits.size(1), -1, -1)
            logits = logits.masked_fill(~mask, float('-inf'))

        # Max-subtraction avoids exp overflow; stay in fp32 for softmax
        logits = logits - logits.amax(dim=-1, keepdim=True)
        attn = F.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        # Multiply in fp32, cast back
        out = torch.matmul(attn, V.float()).to(Q.dtype)
        return out

    def forward(self, Q, K, V, mask: Optional[torch.Tensor]=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.combine_heads(attn_output))


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-04)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-04)
        self.final_norm = nn.LayerNorm(d_model, eps=1e-04)
        self.dropout = nn.Dropout(dropout)

    # Pre-Norm variant
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=mask)

        if not torch.isfinite(attn_output).all():
            print("Non-finite values after feed-forward!")

        x = x + self.dropout(attn_output)
        
        ff_output = self.feed_forward(self.norm2(x))

        if not torch.isfinite(ff_output).all():
            print("Non-finite values after feed-forward!")

        x = x + self.dropout(ff_output)
        x = self.final_norm(x)
        return x
