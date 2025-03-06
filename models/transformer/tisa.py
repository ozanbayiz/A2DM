import torch
import torch.nn as nn
import math
import xformers.ops as xops

class Tisa(nn.Module):
    def __init__(
        self, 
        num_attention_heads,
        tisa_num_kernels,
        tisa_dropout_prob,
        num_position_agnostic_heads,
        max_field_view,
        min_field_view,
        p_eps=1e-8
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_kernels = tisa_num_kernels
        self.tisa_dropout_prob = tisa_dropout_prob
        self.num_position_agnostic_heads = num_position_agnostic_heads
        self.max_field_view = max_field_view
        self.min_field_view = min_field_view
        self.p_eps = p_eps
        self.eps = 1e-8

        # Learned parameters for the translation-invariant bias.
        # Shapes: (1, num_kernels, num_attention_heads, 1, 1)
        self.offsets = nn.Parameter(torch.zeros(1, self.num_kernels, self.num_attention_heads, 1, 1))
        self.amplitudes = nn.Parameter(torch.zeros(1, self.num_kernels, self.num_attention_heads, 1, 1))
        self.sharpness = nn.Parameter(torch.zeros(1, self.num_kernels, self.num_attention_heads, 1, 1))
        # A constant bias per head (will be added to the computed bias)
        self.bias = nn.Parameter(torch.zeros(1, self.num_attention_heads, 1, 1))

        self.dropout = nn.Dropout(self.tisa_dropout_prob)

        # Set up which heads are position‐agnostic.
        self.num_position_aware_heads = self.num_attention_heads - self.num_position_agnostic_heads
        self.position_agnostic_heads = torch.arange(
            self.num_position_aware_heads, self.num_attention_heads
        )

        # Compute slopes for the exponential decay.
        # For a one‐sided field view the slope m is computed as m = -log(p_eps)/field_view.
        self.one_side_min_field_view = self.min_field_view / 2
        self.one_side_max_field_view = self.max_field_view / 2

        self.first_slope = -math.log(self.p_eps) / self.one_side_min_field_view
        self.last_slope = -math.log(self.p_eps) / self.one_side_max_field_view
        # Create slopes for each head. (Only the position‐aware heads get a nonzero slope.)
        slopes_all = (self.first_slope * (self.last_slope / self.first_slope) **
                      (torch.arange(self.num_attention_heads, dtype=torch.float32) /
                       (self.num_position_aware_heads - 1))
                     ).reshape(self.num_attention_heads, 1, 1)
        slopes_all[self.position_agnostic_heads] = 0.0
        # Register slopes as a buffer (non‑trainable)
        self.register_buffer("slopes", slopes_all)

    def forward(self, q, k, v, skip_apply_dropout=False):
        """
        q, k, v : Tensors of shape (B, num_attention_heads, L, d)
        Computes self-attention using xformers.memory_efficient_attention and incorporates
        a translation-invariant relative bias based on learned parameters.
        """
        B, H, L, _ = q.shape  # H should equal num_attention_heads

        # --- Relative Bias Computation ---
        # Create a relative position vector from -(L-1) to (L-1)
        rel_positions = torch.arange(-L+1, L, device=q.device, dtype=q.dtype)  # (2L-1,)
        # Expand to shape (1, H, 2L-1)
        rel_positions = rel_positions.view(1, 1, -1).expand(1, H, -1)
        # Compute exponential decay: -|r| * slopes.
        # (slopes is of shape (H, 1, 1); we squeeze and then unsqueeze to broadcast properly.)
        exp_decay = -rel_positions.abs() * self.slopes.squeeze(-1).squeeze(-1).unsqueeze(0)  # (1, H, 2L-1)

        # Now compute the kernel for each learned kernel parameter.
        # First, reshape rel_positions to match parameter dimensions:
        #   from (1, H, 2L-1) to (1, num_kernels, H, 1, 2L-1)
        rel_positions_exp = rel_positions.unsqueeze(1).unsqueeze(3).expand(
            1, self.num_kernels, H, 1, 2*L-1
        )
        # offsets, amplitudes, and sharpness are (1, num_kernels, H, 1, 1)
        # Compute kernel value = |amplitudes| * sigmoid(sign(amplitudes)*((r - offsets)/sharpness))
        kernel = self.amplitudes.abs() * torch.sigmoid(
            self.amplitudes.sign() * ((rel_positions_exp - self.offsets) / self.sharpness)
        )
        if self.training and not skip_apply_dropout:
            kernel = self.dropout(kernel)
        # Sum over the kernel dimension to get a bias contribution per head and relative position:
        # Now shape: (1, H, 1, 2L-1) then squeeze to (1, H, 2L-1)
        kernel_sum = kernel.sum(dim=1).squeeze(2)
        # Add a positive constant bias and epsilon before taking the logarithm.
        bias_rel = torch.log(kernel_sum + self.eps + self.bias.squeeze(-1))  # (1, H, 2L-1)
        # Add the exponential decay arguments.
        rel_bias = bias_rel + exp_decay  # (1, H, 2L-1)

        # Convert the relative bias vector (of shape (1, H, 2L-1)) to an absolute bias matrix (1, H, L, L)
        attn_bias = self._relative_to_absolute(rel_bias, L)
        # Expand attn_bias along the batch dimension.
        attn_bias = attn_bias.expand(B, -1, -1, -1)  # (B, H, L, L)
        
        # --- Compute Memory Efficient Attention ---
        # xformers’ efficient attention will use the provided attn_bias.
        output = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        return output

    def _relative_to_absolute(self, rel_bias, L):
        """
        Converts a relative bias tensor of shape (1, H, 2L-1) to an absolute bias matrix of shape (1, H, L, L)
        using a relative shift operation.
        """
        # One common trick is to prepend a zero column and then reshape.
        B, H, _ = rel_bias.shape  # here B is 1
        zero_pad = torch.zeros(B, H, 1, device=rel_bias.device, dtype=rel_bias.dtype)
        rel_bias_padded = torch.cat([zero_pad, rel_bias], dim=-1)  # shape (1, H, 2L)
        # Reshape to (B, H, L+1, L)
        rel_bias_padded = rel_bias_padded.view(B, H, L+1, L)
        # Remove the first row to get shape (B, H, L, L)
        absolute_bias = rel_bias_padded[:, :, 1:, :]
        return absolute_bias

    def _init_weights(self):
        """Initialize learned parameters."""
        torch.nn.init.normal_(self.offsets, mean=0.0, std=15.0)
        torch.nn.init.normal_(self.amplitudes, mean=0.0, std=0.01)
        self.sharpness.data.fill_(5.0)
        self.bias.data.fill_(1.0)
