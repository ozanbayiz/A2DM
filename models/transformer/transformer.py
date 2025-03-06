import torch.nn as nn
import torch
from einops import rearrange
from tisa import Tisa

import torch
import xformers.ops as xops

class ConvLayer(nn.Module):
    def __init__(self, d_model, d_ff=2048, dilation=1, dropout = 0.1, bias=False):
        super().__init__() 
        self.conv_1 = nn.Conv1d(d_model, d_ff, 3, padding=dilation, dilation=dilation, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.conv_2 = nn.Conv1d(d_ff, d_model, 3, padding=dilation, dilation=dilation, bias=bias)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.act(self.conv_1(x.permute(0,2,1))))
        x = self.conv_2(x)
        return x.permute(0,2,1)
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1, bias=False):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=bias)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.dropout(self.act(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    

class Attention(nn.Module):
    def __init__(
        self, 
        d_model, 
        num_heads=4, 
        seqlen=100, 
        drop_prob=0.0, 
        tisa_num_kernels=16,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"num_heads ({num_heads}) must divide d_model ({d_model})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(drop_prob) if drop_prob > 0 else None

        # Linear projection that produces 3*d_model channels (for key, value, and query).
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        # A gating linear layer applied after attention.
        self.gate = nn.Linear(d_model, 2 * d_model)

        # Translation invariant bias scorer for relative positions.
        self.position_scorer = Tisa(
            num_attention_heads=num_heads,
            tisa_num_kernels=tisa_num_kernels,
            tisa_dropout_prob=drop_prob,
            num_position_agnostic_heads=0,
            max_field_view=seqlen // 2,
            min_field_view=5
        )
        self.position_scorer._init_weights()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        Returns:
            Tensor of shape (B, L, d_model)
        """
        B, L, d = x.size()
        # Project input into query, key, and value concatenated along the last dimension.
        proj = self.in_proj(x)  # (B, L, 3*d_model)
        # Split into (B, L, 2*d_model) for key/value and (B, L, d_model) for query.
        memory, query = torch.split(proj, (2 * d, d), dim=-1)
        
        # Split and reshape query into shape (B, num_heads, L, head_dim)
        q = query.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # For memory, split equally into key and value.
        k_v = torch.split(memory, d, dim=-1)
        # Replace split_last_dim for k and v
        k = k_v[0].view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = k_v[1].view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention using xformers memory-efficient attention with relative bias.
        attn = self.dot_product_attention(q, k, v)
        
        # Recombine the attention output from (B, num_heads, L, head_dim) to (B, L, d_model)
        out = attn.permute(0, 2, 1, 3).contiguous().view(B, L, d)
        
        if hasattr(self, 'gated_attention') and self.gated_attention:
            # Apply gating.
            gated = self.gate(out)  # (B, L, 2*d_model)
            a, b = gated.chunk(2, dim=-1)
            x_out = a * torch.sigmoid(b)
        else:
            x_out = out
        return x_out

    def dot_product_attention(self, q, k, v):
        """
        Uses xformers memory_efficient_attention with an additive relative bias.
        
        Args:
            q, k, v: Tensors of shape (B, H, L, head_dim)
        Returns:
            Tensor of shape (B, H, L, head_dim)
        """
        B, H, L, _ = q.shape
        # Get the relative bias from the position scorer.
        # This expects a sequence length and returns a tensor of shape (H, L, L).
        rel_bias = self.position_scorer(seq_len=L)  # (H, L, L)
        # Expand bias to batch dimension: (B, H, L, L)
        attn_bias = rel_bias.unsqueeze(0).expand(B, -1, -1, -1)
        # Compute scaled dot product attention using xformers.
        # xops.memory_efficient_attention accepts an attn_bias keyword.
        attn = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        if self.dropout is not None:
            attn = self.dropout(attn)
        return attn

class AttentionBlock(nn.Module):
    def __init__(
        self, 
        d_model,
        num_heads, 
        norm, 
        drop_prob, 
        tisa_num_kernels, 
        seqlen, 
        use_preln=False, 
        d_ff=2048, 
        bias=False, 
        dilation=1
    ):
        super().__init__()
        self.use_preln = use_preln
        self.attn = Attention(
            d_model, 
            num_heads=num_heads, 
            seqlen=seqlen, 
            drop_prob=drop_prob, 
            tisa_num_kernels=tisa_num_kernels
        )
        if (dilation>0):
            self.ff = ConvLayer(d_model, d_ff, dilation=dilation, dropout=drop_prob, bias=bias)
        else:
            self.ff = FeedForward(d_model, d_ff, dropout=drop_prob, bias=bias)

        self.norm = nn.LayerNorm(d_model)
            
        self.dropout_1 = nn.Dropout(drop_prob)
        self.dropout_2 = nn.Dropout(drop_prob)

    def forward(self, x, c):
        x = self.norm(x)
        x = self.attn(x)
        x = self.dropout(x)
        
        if self.use_preln:
            x = self.dropout_1(self.attn(self.norm_2(x))) + x
        else:
            x = self.dropout_1(self.attn(x)) + x
            x = self.norm_2(x)

        if self.use_preln:
            x = self.dropout_2(self.ff(self.norm_1(x))) + x
        else:
            x = self.dropout_2(self.ff(x)) + x
            x = self.norm_1(x)

        return x

class TisaTransformer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        d_model,
        num_blocks,
        num_heads,
        norm,
        drop_prob,
        d_ff=2048,
        tisa_num_kernels=21,
        seqlen=128,
        use_preln=False,
        bias=False,
        dilation=1
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.attention_blocks = nn.ModuleList(
            [
                AttentionBlock(d_model, num_heads, norm, drop_prob, tisa_num_kernels, seqlen, use_preln, d_ff, bias=bias, dilation=dilation)
                for _ in range(num_blocks)
            ]
        )
        self.out_proj = nn.Linear(d_model, out_channels)

    def forward(self, x):
        x = self.in_proj(x)
        for layer in self.attention_blocks:
            x = layer(x)
        x = self.out_proj(x)
        return x