"""
Unit tests for the ConvNextVAE model and its components.
Run with: python -m pytest test_convnext_vae.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from models.convnext.vae import (
    drop_path, DropPath, ConvNextBlock1D, ConvNext1D,
    Encoder, Decoder, ConvNextVAE
)


#------------------------------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------------------------------

def assert_tensor_shape(tensor, expected_shape):
    """Assert that a tensor has the expected shape."""
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"

def check_grad_flow(model):
    """Check that gradients can flow through the model."""
    # Get all parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) > 0, "Model has no parameters that require gradients"
    
    # Check if gradients are None
    has_none_grad = [p.grad is None for p in params]
    assert not all(has_none_grad), "No gradients have been calculated"


#------------------------------------------------------------------------------------------------
# DropPath Tests
#------------------------------------------------------------------------------------------------

def test_drop_path_function():
    """Test the drop_path function."""
    # Create a test tensor
    x = torch.ones(2, 3, 4)
    
    # Test with drop_prob=0 (should return x unchanged)
    output = drop_path(x, drop_prob=0.0, training=True)
    assert torch.allclose(output, x), "drop_path should return input unchanged when drop_prob=0"
    
    # Test with drop_prob>0 but training=False (should return x unchanged)
    output = drop_path(x, drop_prob=0.5, training=False)
    assert torch.allclose(output, x), "drop_path should return input unchanged when training=False"
    
    # Test with drop_prob>0 and training=True (should modify x)
    torch.manual_seed(42)  # For reproducibility
    output = drop_path(x, drop_prob=1.0, training=True, scale_by_keep=False)
    assert torch.sum(output) == 0, "All values should be dropped when drop_prob=1.0"
    
    # Test with drop_prob=0.5 and verify statistical properties
    torch.manual_seed(42)
    outputs = []
    for _ in range(100):
        output = drop_path(x, drop_prob=0.5, training=True)
        outputs.append(output.mean().item())
    
    # The mean should be approximately 1.0 because of scaling
    assert 0.9 < np.mean(outputs) < 1.1, "Statistical properties of drop_path are incorrect"


def test_drop_path_module():
    """Test the DropPath module."""
    # Create a test tensor
    x = torch.ones(2, 3, 4)
    
    # Test with drop_prob=0
    layer = DropPath(drop_prob=0.0)
    output = layer(x)
    assert torch.allclose(output, x), "DropPath should return input unchanged when drop_prob=0"
    
    # Test with drop_prob>0 but eval mode
    layer = DropPath(drop_prob=0.5)
    layer.eval()
    output = layer(x)
    assert torch.allclose(output, x), "DropPath should return input unchanged in eval mode"
    
    # Test with drop_prob>0 and train mode
    layer = DropPath(drop_prob=1.0, scale_by_keep=False)
    layer.train()
    torch.manual_seed(42)
    output = layer(x)
    assert torch.sum(output) == 0, "All values should be dropped when drop_prob=1.0"


#------------------------------------------------------------------------------------------------
# ConvNextBlock1D Tests
#------------------------------------------------------------------------------------------------

def test_convnext_block_1d_shape():
    """Test that ConvNextBlock1D preserves shape."""
    batch_size, channels, seq_len = 2, 32, 100
    x = torch.randn(batch_size, channels, seq_len)
    
    # Test shape preservation
    block = ConvNextBlock1D(dim=channels)
    output = block(x)
    assert_tensor_shape(output, (batch_size, channels, seq_len))


def test_convnext_block_1d_forward():
    """Test that ConvNextBlock1D performs a non-trivial transformation."""
    batch_size, channels, seq_len = 2, 32, 100
    x = torch.randn(batch_size, channels, seq_len)
    
    # The output should be different from the input but preserve the residual connection
    block = ConvNextBlock1D(dim=channels)
    output = block(x)
    
    # Check that output is different from input
    assert not torch.allclose(output, x, atol=1e-3), "ConvNextBlock1D output is too similar to input"
    
    # Check that the residual connection is working (correlation should be high)
    input_flat = x.view(-1)
    output_flat = output.view(-1)
    correlation = torch.corrcoef(torch.stack([input_flat, output_flat]))[0, 1]
    assert correlation > 0.3, f"Residual connection may not be working, correlation: {correlation}"


def test_convnext_block_1d_gradients():
    """Test that gradients flow through ConvNextBlock1D."""
    batch_size, channels, seq_len = 2, 32, 100
    x = torch.randn(batch_size, channels, seq_len, requires_grad=True)
    
    block = ConvNextBlock1D(dim=channels)
    output = block(x)
    loss = output.sum()
    loss.backward()
    
    assert x.grad is not None, "Input gradients are None"
    assert torch.sum(torch.abs(x.grad)) > 0, "Input gradients are all zeros"


def test_convnext_1d_stack():
    """Test the ConvNext1D stack of blocks."""
    batch_size, channels, seq_len = 2, 32, 100
    x = torch.randn(batch_size, channels, seq_len)
    
    stack = ConvNext1D(n_channels=channels, n_depth=3)
    output = stack(x)
    
    # Check shape preservation
    assert_tensor_shape(output, (batch_size, channels, seq_len))
    
    # Check that it produces a deeper transformation
    assert not torch.allclose(output, x, atol=1e-3), "ConvNext1D stack output is too similar to input"


#------------------------------------------------------------------------------------------------
# Encoder Tests
#------------------------------------------------------------------------------------------------

def test_encoder_output_shape():
    """Test the shape of the encoder output."""
    batch_size, seq_len, features = 2, 100, 138
    x = torch.randn(batch_size, seq_len, features)
    
    encoder = Encoder(
        T_in=seq_len,
        input_emb_width=features,
        enc_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3
    )
    
    output = encoder(x)
    
    # Expected shape: [batch_size, enc_emb_width, seq_len/(stride_t^down_t)]
    expected_shape = (batch_size, 512, seq_len // (2**2))
    assert_tensor_shape(output, expected_shape)


def test_encoder_downsampling():
    """Test that the encoder properly downsamples the sequence."""
    batch_size, seq_len, features = 2, 100, 138
    x = torch.randn(batch_size, seq_len, features)
    
    # Test different combinations of downsampling parameters
    for down_t in [1, 2, 3]:
        for stride_t in [2, 3]:
            encoder = Encoder(
                T_in=seq_len,
                input_emb_width=features,
                enc_emb_width=512,
                down_t=down_t,
                stride_t=stride_t,
                width=512,
                depth=3,
                dilation_growth_rate=3
            )
            
            output = encoder(x)
            
            # Calculate expected output size using the same formula as the encoder
            expected_t_out = seq_len
            for _ in range(down_t):
                filter_t, pad_t = stride_t * 2, stride_t // 2
                # Formula for output size of Conv1d: ((input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride) + 1
                expected_t_out = ((expected_t_out + 2*pad_t - 1*(filter_t-1) - 1) // stride_t) + 1
            
            assert output.shape[2] == expected_t_out, \
                f"Expected temporal dim {expected_t_out}, got {output.shape[2]} for down_t={down_t}, stride_t={stride_t}"


#------------------------------------------------------------------------------------------------
# Decoder Tests
#------------------------------------------------------------------------------------------------

def test_decoder_output_shape():
    """Test the shape of the decoder output."""
    batch_size, latent_seq_len, channels = 2, 25, 512
    x = torch.randn(batch_size, channels, latent_seq_len)
    
    decoder = Decoder(
        T_out=100,  # Target output sequence length
        enc_emb_width=channels,
        output_emb_width=138,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3
    )
    
    output = decoder(x)
    
    # Expected shape: [batch_size, T_out, output_emb_width]
    expected_shape = (batch_size, 100, 138)
    assert_tensor_shape(output, expected_shape)


def test_decoder_upsampling():
    """Test that the decoder properly upsamples the sequence to the target length."""
    batch_size, channels = 2, 512
    
    # Try different latent sequence lengths and target output lengths
    test_cases = [
        (25, 100),  # (latent_seq_len, T_out)
        (16, 64),
        (10, 90),
    ]
    
    for latent_seq_len, T_out in test_cases:
        x = torch.randn(batch_size, channels, latent_seq_len)
        
        decoder = Decoder(
            T_out=T_out,
            enc_emb_width=channels,
            output_emb_width=138,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3
        )
        
        output = decoder(x)
        
        # Check that output temporal dimension matches T_out
        assert output.shape[1] == T_out, \
            f"Expected output temporal dim {T_out}, got {output.shape[1]} for latent_seq_len={latent_seq_len}"


#------------------------------------------------------------------------------------------------
# Full VAE Tests
#------------------------------------------------------------------------------------------------

def test_vae_forward():
    """Test the forward pass of the full VAE."""
    batch_size, seq_len, features = 2, 100, 138
    x = torch.randn(batch_size, seq_len, features)
    
    vae = ConvNextVAE(
        latent_dim=128,
        T_in=seq_len,
        input_dim=features,
        encoder_output_channels=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3
    )
    
    recon, mu, logvar = vae(x)
    
    # Check shapes
    assert_tensor_shape(recon, (batch_size, seq_len, features))
    assert_tensor_shape(mu, (batch_size, 128))
    assert_tensor_shape(logvar, (batch_size, 128))


def test_vae_encode_decode():
    """Test the encode and decode methods of the VAE."""
    batch_size, seq_len, features = 2, 100, 138
    x = torch.randn(batch_size, seq_len, features)
    
    vae = ConvNextVAE(
        latent_dim=128,
        T_in=seq_len,
        input_dim=features
    )
    
    # Test encode
    mu, logvar = vae.encode(x)
    assert_tensor_shape(mu, (batch_size, 128))
    assert_tensor_shape(logvar, (batch_size, 128))
    
    # Test reparameterize
    z = vae.reparameterize(mu, logvar)
    assert_tensor_shape(z, (batch_size, 128))
    
    # Test decode
    recon = vae.decode(z)
    assert_tensor_shape(recon, (batch_size, seq_len, features))
    
    # Encode-decode pipeline should work end-to-end
    pipeline_recon = vae.decode(vae.reparameterize(*vae.encode(x)))
    assert_tensor_shape(pipeline_recon, (batch_size, seq_len, features))


def test_vae_loss_computation():
    """Test the loss computation method of the VAE."""
    batch_size, seq_len, features = 2, 100, 138
    x = torch.randn(batch_size, seq_len, features)
    recon_x = torch.randn(batch_size, seq_len, features)
    mu = torch.randn(batch_size, 128)
    logvar = torch.randn(batch_size, 128)
    
    vae = ConvNextVAE(
        latent_dim=128,
        T_in=seq_len,
        input_dim=features
    )
    
    # Test with default KL weight
    total_loss, recon_loss, kl_loss = vae.compute_loss(x, recon_x, mu, logvar)
    
    # Check loss shapes
    assert total_loss.ndim == 0, "Total loss should be a scalar"
    assert recon_loss.ndim == 0, "Reconstruction loss should be a scalar"
    assert kl_loss.ndim == 0, "KL loss should be a scalar"
    
    # Check loss relationships
    assert torch.allclose(total_loss, recon_loss + 0.01 * kl_loss), \
        "Total loss should equal recon_loss + 0.01 * kl_loss"
    
    # Test with custom KL weight
    custom_kl_weight = 0.5
    total_loss_custom, recon_loss_custom, kl_loss_custom = vae.compute_loss(
        x, recon_x, mu, logvar, kl_weight_override=custom_kl_weight
    )
    
    assert torch.allclose(total_loss_custom, recon_loss_custom + custom_kl_weight * kl_loss_custom), \
        f"Total loss should equal recon_loss + {custom_kl_weight} * kl_loss"


def test_vae_training_loop():
    """Test that the VAE can be trained in a basic training loop."""
    batch_size, seq_len, features = 2, 100, 138
    x = torch.randn(batch_size, seq_len, features)
    
    vae = ConvNextVAE(
        latent_dim=128,
        T_in=seq_len,
        input_dim=features,
        # Smaller model for faster testing
        encoder_output_channels=128,
        width=128,
        depth=1
    )
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    # Simple training loop
    initial_loss = None
    final_loss = None
    
    for epoch in range(5):
        optimizer.zero_grad()
        recon, mu, logvar = vae(x)
        loss, recon_loss, kl_loss = vae.compute_loss(x, recon, mu, logvar)
        
        if epoch == 0:
            initial_loss = loss.item()
        
        loss.backward()
        optimizer.step()
        
        final_loss = loss.item()
    
    # Check that loss decreased during training
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"
    
    # Verify gradients
    check_grad_flow(vae)


def test_vae_inference():
    """Test VAE inference (sampling from latent space)."""
    batch_size, seq_len, features = 2, 100, 138
    
    vae = ConvNextVAE(
        latent_dim=128,
        T_in=seq_len,
        input_dim=features
    )
    vae.eval()
    
    # Sample from standard normal as latent vectors
    z = torch.randn(batch_size, 128)
    
    # Decode samples
    samples = vae.decode(z)
    
    # Check shape
    assert_tensor_shape(samples, (batch_size, seq_len, features))


if __name__ == "__main__":
    # Run tests manually if needed
    test_drop_path_function()
    test_drop_path_module()
    test_convnext_block_1d_shape()
    test_convnext_block_1d_forward()
    test_convnext_block_1d_gradients()
    test_convnext_1d_stack()
    test_encoder_output_shape()
    test_encoder_downsampling()
    test_decoder_output_shape()
    test_decoder_upsampling()
    test_vae_forward()
    test_vae_encode_decode()
    test_vae_loss_computation()
    test_vae_training_loop()
    test_vae_inference()
    
    print("All tests passed!")
