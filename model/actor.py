import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings

# Import x-transformers instead of xformers
from x_transformers import TransformerWrapper, Encoder as XEncoder, Decoder as XDecoder
from x_transformers import Attention, AttentionLayers
from x_transformers.x_transformers import AbsolutePositionalEmbedding


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    
    Implementation based on "Attention Is All You Need"
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            x: Tensor, shape [batch_size, seq_len, d_model] with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MemoryEfficientTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with optional memory-efficient attention"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_xformers=True):
        super(MemoryEfficientTransformerEncoderLayer, self).__init__()
        self.use_xformers = use_xformers
        
        # Self-attention
        if self.use_xformers:
            # xFormers implementation will be used in the forward pass
            self.self_attn = None
        else:
            # Standard PyTorch implementation
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Store parameters for xFormers
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, d_model]
            src_mask: Tensor, optional mask for src sequence
            src_key_padding_mask: Tensor, optional mask for src keys per batch
            
        Returns:
            output: Tensor, shape [seq_len, batch_size, d_model]
        """
        src2 = self.norm1(src)
        
        if self.use_xformers:
            # Reshape to [batch_size, seq_len, num_heads, head_dim]
            batch_size = src.size(1)
            seq_len = src.size(0)
            q = k = v = src2.transpose(0, 1).reshape(batch_size, seq_len, self.nhead, self.head_dim)
            
            # Apply xFormers memory-efficient attention
            attention_output = xformers.ops.memory_efficient_attention(
                q, k, v, 
                attn_bias=None if src_mask is None else src_mask.to(q.dtype)
            )
            
            # Reshape back to [seq_len, batch_size, d_model]
            attention_output = attention_output.reshape(batch_size, seq_len, self.d_model).transpose(0, 1)
        else:
            # Use standard PyTorch attention
            attention_output, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                                key_padding_mask=src_key_padding_mask)
        
        # First residual connection
        src = src + self.dropout1(attention_output)
        
        # Feed forward and second residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class MemoryEfficientTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with optional memory-efficient attention"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, use_xformers=True):
        super(MemoryEfficientTransformerDecoderLayer, self).__init__()
        self.use_xformers = use_xformers
        
        # Self-attention and cross-attention
        if self.use_xformers:
            # xFormers implementation will be used in the forward pass
            self.self_attn = None
            self.multihead_attn = None
        else:
            # Standard PyTorch implementation
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Store parameters for xFormers
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: Tensor, shape [tgt_len, batch_size, d_model]
            memory: Tensor, shape [src_len, batch_size, d_model]
            tgt_mask: Tensor, optional mask for tgt sequence
            memory_mask: Tensor, optional mask for memory sequence
            tgt_key_padding_mask: Tensor, optional mask for tgt keys per batch
            memory_key_padding_mask: Tensor, optional mask for memory keys per batch
            
        Returns:
            output: Tensor, shape [tgt_len, batch_size, d_model]
        """
        tgt2 = self.norm1(tgt)
        
        if self.use_xformers:
            # Self-attention with xFormers
            batch_size = tgt.size(1)
            tgt_len = tgt.size(0)
            q = k = v = tgt2.transpose(0, 1).reshape(batch_size, tgt_len, self.nhead, self.head_dim)
            
            self_attention_output = xformers.ops.memory_efficient_attention(
                q, k, v, 
                attn_bias=None if tgt_mask is None else tgt_mask.to(q.dtype)
            )
            
            self_attention_output = self_attention_output.reshape(batch_size, tgt_len, self.d_model).transpose(0, 1)
        else:
            # Self-attention with PyTorch
            self_attention_output, _ = self.self_attn(tgt2, tgt2, tgt2, attn_mask=tgt_mask,
                                                     key_padding_mask=tgt_key_padding_mask)
        
        # First residual connection
        tgt = tgt + self.dropout1(self_attention_output)
        tgt2 = self.norm2(tgt)
        
        if self.use_xformers:
            # Cross-attention with xFormers
            batch_size = tgt.size(1)
            tgt_len = tgt.size(0)
            src_len = memory.size(0)
            
            q = tgt2.transpose(0, 1).reshape(batch_size, tgt_len, self.nhead, self.head_dim)
            k = v = memory.transpose(0, 1).reshape(batch_size, src_len, self.nhead, self.head_dim)
            
            cross_attention_output = xformers.ops.memory_efficient_attention(
                q, k, v,
                attn_bias=None if memory_mask is None else memory_mask.to(q.dtype)
            )
            
            cross_attention_output = cross_attention_output.reshape(batch_size, tgt_len, self.d_model).transpose(0, 1)
        else:
            # Cross-attention with PyTorch
            cross_attention_output, _ = self.multihead_attn(tgt2, memory, memory, attn_mask=memory_mask,
                                                           key_padding_mask=memory_key_padding_mask)
        
        # Second residual connection
        tgt = tgt + self.dropout2(cross_attention_output)
        
        # Feed forward and third residual connection
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        
        return tgt


class MemoryEfficientTransformerEncoder(nn.Module):
    """Memory-efficient Transformer encoder"""
    
    def __init__(self, encoder_layer, num_layers):
        super(MemoryEfficientTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size, d_model]
            mask: Tensor, optional mask for self-attention
            src_key_padding_mask: Tensor, optional mask for input keys
            
        Returns:
            output: Tensor, shape [seq_len, batch_size, d_model]
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            
        return output


class MemoryEfficientTransformerDecoder(nn.Module):
    """Memory-efficient Transformer decoder"""
    
    def __init__(self, decoder_layer, num_layers):
        super(MemoryEfficientTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: Tensor, shape [tgt_len, batch_size, d_model]
            memory: Tensor, shape [src_len, batch_size, d_model]
            tgt_mask: Tensor, optional mask for tgt sequence
            memory_mask: Tensor, optional mask for memory sequence
            tgt_key_padding_mask: Tensor, optional mask for tgt keys
            memory_key_padding_mask: Tensor, optional mask for memory keys
            
        Returns:
            output: Tensor, shape [tgt_len, batch_size, d_model]
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
            
        return output


class Encoder(nn.Module):
    """Transformer Encoder with distribution tokens for VAE using x-transformers"""
    
    def __init__(self, input_dim, latent_dim, d_model=512, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, **kwargs):
        super(Encoder, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Distribution tokens (μ and Σ tokens)
        self.mu_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.sigma_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # X-Transformer encoder
        self.transformer_encoder = TransformerWrapper(
            dim = d_model,
            max_seq_len = 512,  # Assuming max sequence length
            attn_layers = AttentionLayers(
                dim = d_model,
                depth = num_encoder_layers,
                heads = nhead,
                ff_mult = dim_feedforward // d_model,
                attn_dropout = dropout,
                ff_dropout = dropout,
                causal = False
            )
        )
        
        # Output projections for latent space distribution parameters
        self.mu_projection = nn.Linear(d_model, latent_dim)
        self.logvar_projection = nn.Linear(d_model, latent_dim)
    
    def forward(self, src):
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, input_dim]
                 Motion sequence (poses & translations)
        Returns:
            z: Sampled latent vectors, shape [batch_size, latent_dim]
            mu: Mean vectors of the latent distributions, shape [batch_size, latent_dim]
            logvar: Log variance vectors of the latent distributions, shape [batch_size, latent_dim]
        """
        batch_size, seq_len, _ = src.shape
        
        # Project input to d_model dimension
        x = self.input_projection(src)  # [batch_size, seq_len, d_model]
        
        # Prepare distribution tokens for batch
        mu_tokens = self.mu_token.expand(batch_size, 1, -1)  # [batch_size, 1, d_model]
        sigma_tokens = self.sigma_token.expand(batch_size, 1, -1)  # [batch_size, 1, d_model]
        
        # Prepend distribution tokens to sequence
        x = torch.cat([mu_tokens, sigma_tokens, x], dim=1)  # [batch_size, seq_len+2, d_model]
        
        # Apply transformer encoder - x-transformers uses batch-first format by default
        memory = self.transformer_encoder(x)  # [batch_size, seq_len+2, d_model]
        
        # Extract distribution parameters from the first two tokens
        mu_embedding = memory[:, 0]  # [batch_size, d_model]
        logvar_embedding = memory[:, 1]  # [batch_size, d_model]
        
        # Project to latent dimension
        mu = self.mu_projection(mu_embedding)  # [batch_size, latent_dim]
        logvar = self.logvar_projection(logvar_embedding)  # [batch_size, latent_dim]
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)  # [batch_size, latent_dim]
        eps = torch.randn_like(std)  # [batch_size, latent_dim]
        z = mu + eps * std  # [batch_size, latent_dim]
        
        return z, mu, logvar


class Decoder(nn.Module):
    """Transformer Decoder for motion generation using x-transformers"""
    
    def __init__(self, output_dim, latent_dim, seq_len, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, **kwargs):
        super(Decoder, self).__init__()
        
        # Store sequence length
        self.seq_len = seq_len
        
        # Projection from latent dim to d_model
        self.latent_projection = nn.Linear(latent_dim, d_model)
        
        # Query embedding for decoding
        self.query_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # X-Transformer decoder (cross-attention setup)
        # First, transformer for processing the latent vector
        self.context_transformer = TransformerWrapper(
            dim = d_model,
            max_seq_len = 8,  # Small value since we only have one latent token
            attn_layers = AttentionLayers(
                dim = d_model,
                depth = 2,  # Small number of layers to process context
                heads = nhead,
                ff_mult = dim_feedforward // d_model,
                causal = False
            )
        )
        
        # Main decoder with cross-attention
        self.transformer_decoder = TransformerWrapper(
            dim = d_model,
            max_seq_len = seq_len,
            attn_layers = AttentionLayers(
                dim = d_model,
                depth = num_decoder_layers,
                heads = nhead,
                ff_mult = dim_feedforward // d_model,
                cross_attend = True,  # Enable cross-attention
                ff_dropout = dropout,
                attn_dropout = dropout,
                causal = False
            )
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
        # Positional encoding
        self.pos_embedding = AbsolutePositionalEmbedding(d_model, seq_len)
    
    def forward(self, z):
        """
        Args:
            z: Tensor, shape [batch_size, latent_dim]
               Latent vectors
               
        Returns:
            output: Tensor, shape [batch_size, seq_len, output_dim]
                   Generated motion sequence
        """
        batch_size = z.shape[0]
        
        # Project latent vector to d_model dimension
        memory = self.latent_projection(z)  # [batch_size, d_model]
        
        # Reshape to sequence with length 1
        memory = memory.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Process memory with context transformer
        memory = self.context_transformer(memory)  # [batch_size, 1, d_model]
        
        # Create positional queries for each time step
        queries = self.query_embed.expand(batch_size, -1, -1)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        queries = queries + self.pos_embedding(queries)
        
        # Apply transformer decoder with cross-attention to the latent context
        output = self.transformer_decoder(
            queries,  # [batch_size, seq_len, d_model]
            context = memory  # [batch_size, 1, d_model]
        )  # [batch_size, seq_len, d_model]
        
        # Project to output dimension
        output = self.output_projection(output)  # [batch_size, seq_len, output_dim]
        
        return output


class ACTOR(nn.Module):
    """ACTOR: Transformer VAE for motion sequence generation using x-transformers"""
    
    def __init__(
        self, 
        input_dim, 
        output_dim,
        seq_len,
        latent_dim=256, 
        d_model=512, 
        nhead=8, 
        num_encoder_layers=6, 
        num_decoder_layers=6,
        dim_feedforward=2048, 
        dropout=0.1,
        **kwargs
    ):
        """
        Args:
            input_dim: dimension of input motion representation (pose params)
            output_dim: dimension of output motion representation 
            seq_len: length of motion sequences
            latent_dim: dimension of the latent vector z
            d_model: hidden dimension of transformer
            nhead: number of heads in transformer attention
            num_encoder_layers: number of encoder layers
            num_decoder_layers: number of decoder layers
            dim_feedforward: dimension of transformer feedforward network
            dropout: dropout rate
        """
        super(ACTOR, self).__init__()
        
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Encoder
        self.encoder = Encoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        
        # Decoder
        self.decoder = Decoder(
            output_dim=output_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
    
    def encode(self, x):
        """
        Encode motion sequence to latent space
        
        Args:
            x: Tensor, shape [batch_size, seq_len, input_dim]
        
        Returns:
            z: Tensor, shape [batch_size, latent_dim]
            mu: Tensor, shape [batch_size, latent_dim]
            logvar: Tensor, shape [batch_size, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z):
        """
        Decode from latent space to motion sequence
        
        Args:
            z: Tensor, shape [batch_size, latent_dim]
            
        Returns:
            output: Tensor, shape [batch_size, seq_len, output_dim]
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass through the full VAE
        
        Args:
            x: Tensor, shape [batch_size, seq_len, input_dim]
               Input motion sequence
               
        Returns:
            output: Tensor, shape [batch_size, seq_len, output_dim]
                   Reconstructed motion sequence
            z: Tensor, shape [batch_size, latent_dim]
               Sampled latent vectors
            mu: Tensor, shape [batch_size, latent_dim]
               Mean of latent distribution
            logvar: Tensor, shape [batch_size, latent_dim]
                    Log variance of latent distribution
        """
        # Check that input sequence length matches the expected sequence length
        if x.size(1) != self.seq_len:
            raise ValueError(f"Input sequence length {x.size(1)} does not match model's expected sequence length {self.seq_len}")
        
        # Encode
        z, mu, logvar = self.encode(x)
        
        # Decode
        output = self.decode(z)
        
        return output, z, mu, logvar
    
    def sample(self, batch_size=1, device='cuda'):
        """
        Sample new motion sequences without input
        
        Args:
            batch_size: Number of sequences to generate
            device: device to create tensors on
            
        Returns:
            output: Tensor, shape [batch_size, seq_len, output_dim]
                   Generated motion sequence
        """
        # Sample from prior
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # Decode
        output = self.decode(z)
        
        return output
    
    def compute_loss(self, prediction, target, mu, logvar, kl_weight_override=None):
        """
        Compute VAE loss
        
        Args:
            prediction: predicted output
            target: ground truth
            mu: mean of latent space
            logvar: log variance of latent space
            kl_weight_override: optional override for KL weight
            
        Returns:
            total_loss: combined loss
            recon_loss: reconstruction loss
            kl_loss: KL divergence loss
        """
        # Use default or override KL weight
        kl_weight = 0.01 if kl_weight_override is None else kl_weight_override
            
        # Reconstruction loss
        recon_loss = F.mse_loss(prediction, target)
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def update_kl_weight(self, epoch, total_epochs):
        """
        Update KL weight using a cyclical schedule
        
        Args:
            epoch: current epoch
            total_epochs: total number of epochs
            
        Returns:
            kl_weight: updated KL weight
        """
        # Simple cyclical schedule
        cycle = total_epochs // 4
        progress = (epoch % cycle) / cycle
        
        if epoch < cycle:
            # Warm-up
            kl_weight = min(0.01, progress * 0.01)
        else:
            # Cyclical
            kl_weight = 0.01 * (1 + math.sin(2 * math.pi * progress))
        
        return kl_weight


def kl_divergence_loss(mu, logvar):
    """
    KL divergence loss for the VAE
    
    Args:
        mu: Mean of latent distribution, shape [batch_size, latent_dim]
        logvar: Log variance of latent distribution, shape [batch_size, latent_dim]
        
    Returns:
        KL divergence loss (scalar)
    """
    # KL divergence between N(mu, sigma) and N(0, 1)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kld_loss.mean()


def compute_loss(prediction, target, mu, logvar, vertices_pred=None, vertices_target=None, 
                 kl_weight=0.1, vertex_weight=1.0):
    """
    Compute the complete ACTOR loss
    
    Args:
        prediction: predicted motion sequence, shape [batch_size, seq_len, output_dim]
        target: target motion sequence, shape [batch_size, seq_len, input_dim]
        mu: mean of latent distribution, shape [batch_size, latent_dim]
        logvar: log variance of latent distribution, shape [batch_size, latent_dim]
        vertices_pred: predicted SMPL vertices (optional)
        vertices_target: target SMPL vertices (optional)
        kl_weight: weight for KL divergence loss
        vertex_weight: weight for vertex loss
        
    Returns:
        total_loss: weighted sum of reconstruction, KL, and vertex losses
        losses_dict: dictionary with individual loss terms
    """
    # Reconstruction loss
    rec_loss = F.mse_loss(prediction, target)
    
    # KL divergence loss
    kl_loss = kl_divergence_loss(mu, logvar)
    
    # Vertex loss (if provided)
    vertex_loss = 0.0
    if vertices_pred is not None and vertices_target is not None:
        vertex_loss = F.mse_loss(vertices_pred, vertices_target)
    
    # Total loss
    total_loss = rec_loss + kl_weight * kl_loss + vertex_weight * vertex_loss
    
    # Return individual losses for logging
    losses_dict = {
        'total_loss': total_loss.item(),
        'rec_loss': rec_loss.item(),
        'kl_loss': kl_loss.item(),
        'vertex_loss': vertex_loss if isinstance(vertex_loss, float) else vertex_loss.item()
    }
    
    return total_loss, losses_dict


def main():
    """
    Test the forward pass of the ACTOR model
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define model parameters
    input_dim = 69  # Example dimension for SMPL pose parameters
    output_dim = 69  # Output is same dimension as input
    latent_dim = 256
    seq_len = 100  # Example sequence length
    
    # Create a test batch
    batch_size = 4
    test_input = torch.randn(batch_size, seq_len, input_dim)
    
    # Initialize the model
    model = ACTOR(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        latent_dim=latent_dim,
    ).to('mps')
    
    # Set model to evaluation mode
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output, z, mu, logvar = model(test_input)
    
    # Print shapes
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Latent z shape: {z.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    
    # Compute loss
    loss, loss_dict = compute_loss(output, test_input, mu, logvar)
    
    # Print loss values
    print("\nLoss values:")
    for key, value in loss_dict.items():
        print(f"{key}: {value:.6f}")
    
    # Test generation
    with torch.no_grad():
        generated = model.sample(batch_size=batch_size)
    
    print(f"\nGenerated shape: {generated.shape}")
    
    print("\nModel test completed successfully!")


if __name__ == "__main__":
    main()
