import math
import os
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import torch
from einops import einsum, rearrange
from lightning.pytorch.callbacks import Callback
from torch import nn
from torchvision.ops import stochastic_depth
import numpy as np
import matplotlib.cm as cm


class DropPath(nn.Module):
    """Stochastic depth for regularization https://arxiv.org/abs/1603.09382"""

    def __init__(self, p: float = 0.0, mode: str = "row"):
        super(DropPath, self).__init__()
        assert 0 <= p <= 1
        assert mode in ("batch", "row")
        self.p, self.mode = p, mode

    def forward(self, inputs: torch.Tensor):
        return stochastic_depth(inputs, p=self.p, mode=self.mode, training=self.training)


class RotaryPosEmb(nn.Module):
    """
    Rotary position embedding (RoPE)
    Reference
    - Su et al. 2021 https://arxiv.org/abs/2104.09864
    - Sun et al. 2022 https://arxiv.org/abs/2212.10554
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        reg_tokens: int,
        scale_base: int = 512,
        use_xpos: bool = True,
    ):
        super(RotaryPosEmb, self).__init__()
        self.num_tokens = num_tokens
        self.reg_tokens = reg_tokens

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        self.register_buffer("scale", (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim))
        self.create_embedding(n=num_tokens)

    def create_embedding(self, n: int):
        device = self.scale.device
        t = torch.arange(n, dtype=self.inv_freq.dtype, device=device)
        freq = torch.einsum("i , j -> i j", t, self.inv_freq)
        freq = torch.cat((freq, freq), dim=-1)
        if self.use_xpos:
            power = (t - (n // 2)) / self.scale_base
            scale = self.scale ** rearrange(power, "n -> n 1")
            scale = torch.cat((scale, scale), dim=-1)
        else:
            scale = torch.ones(1, device=device)
        self.register_buffer("emb_sin", torch.sin(freq), persistent=False)
        self.register_buffer("emb_cos", torch.cos(freq), persistent=False)
        self.register_buffer("emb_scale", scale, persistent=False)

    def get_embedding(self, n: int):
        if self.emb_sin is None or self.emb_sin.shape[-2] < n:
            self.create_embedding(n)
        return self.emb_sin[:n], self.emb_cos[:n], self.emb_scale[:n]

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def rotate(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
        scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            (q * cos * scale) + (cls.rotate_half(q) * sin * scale),
            (k * cos * scale) + (cls.rotate_half(k) * sin * scale),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        n = q.size(2)
        q_reg, k_reg = None, None
        if self.reg_tokens:
            q_reg = q[:, :, -self.reg_tokens :, :]
            k_reg = k[:, :, -self.reg_tokens :, :]
            n -= self.reg_tokens
            q = q[:, :, : -self.reg_tokens, :]
            k = k[:, :, : -self.reg_tokens, :]
        sin, cos, scale = self.get_embedding(n)
        q, k = self.rotate(q, k, sin, cos, scale)
        if q_reg is not None and k_reg is not None:
            q = torch.cat((q, q_reg), dim=2)
            k = torch.cat((k, k_reg), dim=2)
        return q, k


class SinCosPosEmb(nn.Module):
    def __init__(self, emb_dim: int, input_shape: Tuple[int, int]):
        super(SinCosPosEmb, self).__init__()
        assert emb_dim % 2 == 0, f"emb_dim must be divisible by 2, got {emb_dim}."
        self.emb_dim = emb_dim

        h, w = input_shape
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
        grid = torch.stack(grid, dim=0)
        grid = grid.reshape([2, 1, h, w])
        emb_h = self._1d_sin_cos_pos_emb(self.emb_dim // 2, pos=grid[0])
        emb_w = self._1d_sin_cos_pos_emb(self.emb_dim // 2, pos=grid[1])
        pos_emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)

        self.register_buffer("pos_emb", pos_emb, persistent=False)

    @staticmethod
    def _1d_sin_cos_pos_emb(emb_dim: int, pos: torch.Tensor):
        omega = torch.arange(emb_dim // 2, dtype=torch.float32)
        omega /= emb_dim / 2.0
        omega = 1.0 / 10000**omega
        pos = torch.flatten(pos)
        out = einsum(pos, omega, "m, d -> m d")
        emb = torch.cat([torch.sin(out), torch.cos(out)], dim=1)
        return emb

    def forward(self, inputs: torch.Tensor):
        b, t, p, d = inputs.shape
        return inputs + self.pos_emb[None, None, :p]


class SinusoidalPosEmb(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_length: int,
        dimension: Literal["spatial", "temporal"],
        dropout: float = 0.0,
    ):
        super(SinusoidalPosEmb, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.dimension = dimension

        # Use a larger max_length to accommodate test set
        # Adjust this value if needed (150 for your test set)
        actual_max_length = max(max_length, 750)

        position = torch.arange(actual_max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        match self.dimension:
            case "temporal":
                pos_encoding = torch.zeros(1, actual_max_length, 1, d_model)
                pos_encoding[0, :, 0, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, :, 0, 1::2] = torch.cos(position * div_term)
            case "spatial":
                pos_encoding = torch.zeros(1, 1, actual_max_length, d_model)
                pos_encoding[0, 0, :, 0::2] = torch.sin(position * div_term)
                pos_encoding[0, 0, :, 1::2] = torch.cos(position * div_term)
            case _:
                raise NotImplementedError(f"invalid dimension {self.dimension} in SinusoidalPositionalEncoding")

        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        match self.dimension:
            case "temporal":
                outputs += self.pos_encoding[:, : inputs.size(1), :, :]
            case "spatial":
                outputs += self.pos_encoding[:, :, : inputs.size(2), :]
        return self.dropout(outputs)


def get_norm_layer(norm_type: str, normalized_shape: int | Tuple[int, ...]) -> nn.Module:
    norm_key = norm_type.lower()
    if norm_key in ("layernorm", "layer_norm", "ln"):
        return nn.LayerNorm(normalized_shape)
    if norm_key in ("rmsnorm", "rsmnorm", "rms_norm", "rsm_norm"):
        return nn.RMSNorm(normalized_shape)
    raise ValueError(f"Unsupported normalization type '{norm_type}'.")

import os
import torch
import numpy as np
from matplotlib import cm
from scipy.ndimage import zoom
def extract_attention_maps(
    pl_module,
    val_dataloader,
    target_session,
    outdir="./attention_viz",
    device='cuda',
    neuron_idx=0,
    head_idx=0,
    batch_idx=0,
    highlight_patch=True,
    highlight_color=(0.0, 0.0, 0.0),  # RGB, red by default
    point_size=1
):
    """
    Extract attention maps from a trained model, matching the temporal sampling
    strategy used during prediction (patch centers), then interpolating to original
    temporal dimension.
    
    Args:
        pl_module: Lightning module with core and readout
        val_dataloader: Validation dataloader
        target_session: Session name to extract from
        outdir: Output directory for saved arrays
        device: Device to run inference on
        neuron_idx: Index of neuron to visualize
        head_idx: Attention head index
        batch_idx: Batch index to use
        highlight_patch: Whether to draw a point at the neuron's patch center
        highlight_color: RGB tuple for the point color (values 0-1)
        point_size: Radius of the point in pixels
        
    Returns:
        dict with 'original_frames', 'overlaid_frames', 'attention_maps', 'metadata'
    """
    
    pl_module.eval().to(device)
    
    # Find target session batch
    batch = None
    for session_name, data_point in val_dataloader:
        if session_name == target_session:
            batch = data_point
            break
    
    if batch is None:
        raise ValueError(f"Session {target_session} not found in dataloader")
    
    # Get input frames
    frames = batch.inputs.to(device)  # (B, C, T, H, W)
    B, C, T, H, W = frames.shape
    b = min(batch_idx, B - 1)
    
    # Get tokenizer parameters
    core = pl_module.core
    kernel_size = core.tokenizer.kernel_size[0]
    stride = core.tokenizer.stride[0]
    
    # Compute patch centers (same as in training_step)
    num_patches = (T - kernel_size) // stride + 1
    patch_center_indices = torch.arange(num_patches, device=device) * stride + kernel_size // 2
    
    print(f"Video shape: {frames.shape}")
    print(f"Temporal params: kernel={kernel_size}, stride={stride}")
    print(f"Number of patches: {num_patches}, centers at: {patch_center_indices.tolist()}")
    
    # Forward pass to get attention maps
    with torch.no_grad():
        frames_batch = frames[b:b+1]
        tokens = core.tokenizer(frames_batch)
        attn_maps = core.get_spatial_attention_maps(tokens, layer_idx=-1)  # (num_patches, num_heads, P, P)
        
        if attn_maps is None:
            raise ValueError("Model returned None for attention maps")
    
    # Get spatial dimensions
    H_tok, W_tok = core.tokenizer.new_shape
    P = H_tok * W_tok
    
    # Get neuron's spatial location
    session_readout = pl_module.readout[target_session]
    grid = session_readout.grid  # (1, N_neurons, 1, 2)
    gx, gy = grid[0, neuron_idx, 0, 0], grid[0, neuron_idx, 0, 1]  # [-1, 1] normalized
    print(attention_entropy(attn_maps))
    
    # Convert to patch coordinates
    x_patch = int(torch.clamp(((gx + 1) * 0.5 * W_tok), 0, W_tok - 1).item())
    y_patch = int(torch.clamp(((gy + 1) * 0.5 * H_tok), 0, H_tok - 1).item())
    neuron_patch_idx = y_patch * W_tok + x_patch
    
    print(f"Neuron {neuron_idx} at grid ({gx:.2f}, {gy:.2f}) -> patch ({y_patch}, {x_patch}) = {neuron_patch_idx}")
    
    # Calculate patch center in original resolution
    patch_h = H / H_tok
    patch_w = W / W_tok
    center_y = int((y_patch + 0.5) * patch_h)
    center_x = int((x_patch + 0.5) * patch_w)
    
    print(f"Patch center in original resolution: ({center_y}, {center_x})")
    
    # Extract attention at patch centers
    attention_at_centers = []
    for patch_idx in range(num_patches):
        attn_head = attn_maps[patch_idx, head_idx]  # (P, P)
        attn_from_neuron = attn_head[:,neuron_patch_idx]  # (P,)
        
        # Reshape to spatial grid and normalize
        attn_spatial = attn_from_neuron.view(H_tok, W_tok)
        attn_spatial = (attn_spatial - attn_spatial.min())
        if attn_spatial.max() > 0:
            attn_spatial = attn_spatial / attn_spatial.max()
        
        # Upsample to original resolution
        attn_upsampled = torch.nn.functional.interpolate(
            attn_spatial[None, None, :, :],
            size=(H, W),
            mode='bilinear',
            align_corners=True
        ).squeeze().cpu().numpy()
        
        attention_at_centers.append(attn_upsampled)
    
    # Interpolate attention maps to original temporal dimension
    attention_at_centers = np.array(attention_at_centers)  # (num_patches, H, W)
    
    # Use scipy zoom for temporal interpolation
    temporal_zoom_factor = T / num_patches
    attention_maps_full = zoom(attention_at_centers, (temporal_zoom_factor, 1, 1), order=1)
    
    # Ensure exact length match
    if attention_maps_full.shape[0] != T:
        attention_maps_full = attention_maps_full[:T]
    
    # Create overlaid visualizations
    original_frames = []
    overlaid_frames = []
    
    frames_cpu = frames[b].cpu().numpy()  # (C, T, H, W)
    for t in range(T):
        frame = frames_cpu[:, t, :, :]  # (C, H, W)
        attn = attention_maps_full[t]  # (H, W)
        
        # Normalize frame
        base = frame[0]
        base_norm = (base - base.min()) / (base.max() - base.min() + 1e-8)
        
        # Apply colormap to attention
        cmap = cm.get_cmap('viridis')
        attn_colored = cmap(attn)[:, :, :3]
        
        # Overlay
        base_rgb = np.stack([base_norm] * 3, axis=-1)
        overlaid = np.clip(0.55 * base_rgb + 0.45 * attn_colored, 0, 1)
        
        # Draw point at neuron's patch center
        if highlight_patch:
            # Create a circular mask
            y_coords, x_coords = np.ogrid[:H, :W]
            mask = (y_coords - center_y)**2 + (x_coords - center_x)**2 <= point_size**2
            overlaid[mask] = highlight_color
        
        original_frames.append(frame)
        overlaid_frames.append(overlaid)
    
    # Save results
    os.makedirs(outdir, exist_ok=True)
    save_name = f"{target_session}_b{b}_neuron{neuron_idx}_head{head_idx}"
    
    np.save(os.path.join(outdir, f"{save_name}_original.npy"), np.stack(original_frames))
    np.save(os.path.join(outdir, f"{save_name}_overlaid.npy"), np.stack(overlaid_frames))
    np.save(os.path.join(outdir, f"{save_name}_attention.npy"), attention_maps_full)
    
    print(f"Saved to {outdir}/{save_name}_*.npy")
    
    metadata = {
        'session_name': target_session,
        'batch_idx': b,
        'neuron_idx': neuron_idx,
        'head_idx': head_idx,
        'shape': (T, H, W),
        'num_patches': num_patches,
        'patch_centers': patch_center_indices.cpu().tolist(),
        'temporal_stride': stride,
        'temporal_kernel_size': kernel_size,
        'neuron_patch_idx': neuron_patch_idx,
        'patch_center_coords': (center_y, center_x)
    }
    
    return {
        'original_frames': np.stack(original_frames),
        'overlaid_frames': np.stack(overlaid_frames),
        'attention_maps': attention_maps_full,
        'metadata': metadata
    }
import torch
import torch.nn.functional as F
import torch

import torch

def attention_entropy(attn):
    """
    attn: Tensor of shape (num_patches, num_heads, P, P)
    returns a vector of length num_heads with average entropy per head
    """
    p = attn.clamp(min=1e-12)
    entropy = -(p * p.log()).sum(dim=-1)               # sum over key dimension (P)
    return entropy.mean(dim=(0, 2))                   # average over patches and queries


def temporal_gaussian_smooth(x, kernel_size=15, log_sigma=None):
    """
    Gaussian smoothing with learnable or fixed sigma
    x: (B, T, N)
    log_sigma: learnable parameter (scalar tensor) or None for fixed sigma=4.0
    """
    pad = kernel_size // 2
    B, T, N = x.shape
    
    # Store original mean per sequence
    original_mean = x.mean(dim=1, keepdim=True)  # (B, 1, N)
    
    x_reshape = x.permute(0, 2, 1).reshape(B*N, 1, T)
    x_padded = F.pad(x_reshape, (pad, pad), mode='replicate')
    
    # Use learnable sigma if provided, otherwise fixed
    if log_sigma is not None:
        sigma = torch.exp(log_sigma) 
    else:
        sigma = 4.0
    
    # Create Gaussian kernel
    coords = torch.arange(kernel_size, device=x.device, dtype=torch.float32)
    coords = coords - kernel_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    
    # Convolution
    x_smooth = F.conv1d(x_padded, kernel, padding=0)
    x_smooth = x_smooth.view(B, N, T).permute(0, 2, 1)
    
    # Restore original mean
    smoothed_mean = x_smooth.mean(dim=1, keepdim=True)
    x_smooth = x_smooth - smoothed_mean + original_mean
    
    return x_smooth


import os
import torch
import numpy as np
from matplotlib import cm




def extract_temporal_attention_maps(
    pl_module,
    val_dataloader,
    target_session,
    outdir="./attention_viz",
    device='cuda',
    neuron_idx=0,
    head_idx=0,
    batch_idx=0
):
    """
    Extract attention maps from a trained model, matching the temporal sampling
    strategy used during prediction (patch centers), then interpolating to original
    temporal dimension.
    
    Args:
        pl_module: Lightning module with core and readout
        val_dataloader: Validation dataloader
        target_session: Session name to extract from
        outdir: Output directory for saved arrays
        device: Device to run inference on
        neuron_idx: Index of neuron to visualize
        head_idx: Attention head index
        batch_idx: Batch index to use
        
    Returns:
        dict with 'original_frames', 'overlaid_frames', 'attention_maps', 'metadata'
    """
    
    pl_module.eval().to(device)
    
    # Find target session batch
    batch = None
    for session_name, data_point in val_dataloader:
        if session_name == target_session:
            batch = data_point
            break
    
    if batch is None:
        raise ValueError(f"Session {target_session} not found in dataloader")
    
    # Get input frames
    frames = batch.inputs.to(device)  # (B, C, T, H, W)
    B, C, T, H, W = frames.shape
    b = min(batch_idx, B - 1)
    
    # Get tokenizer parameters
    core = pl_module.core
    kernel_size = core.tokenizer.kernel_size[0]
    stride = core.tokenizer.stride[0]
    
    # Compute patch centers (same as in training_step)
    num_patches = (T - kernel_size) // stride + 1
    patch_center_indices = torch.arange(num_patches, device=device) * stride + kernel_size // 2
    
    print(f"Video shape: {frames.shape}")
    print(f"Temporal params: kernel={kernel_size}, stride={stride}")
    print(f"Number of patches: {num_patches}, centers at: {patch_center_indices.tolist()}")
    
    # Forward pass to get attention maps
    with torch.no_grad():
        frames_batch = frames[b:b+1]
        tokens = core.tokenizer(frames_batch)
        attn_maps = get_temporal_attention_maps_util(
            tokens=tokens,
            temporal_blocks=core.vivit.temporal_transformer.blocks,
            layer_idx=-1
        )

        
        if attn_maps is None:
            raise ValueError("Model returned None for attention maps")
    print(attention_entropy(attn_maps))

    # Get spatial dimensions
    H_tok, W_tok = core.tokenizer.new_shape
    P = H_tok * W_tok
    
    # Get neuron's spatial location
    session_readout = pl_module.readout[target_session]
    grid = session_readout.grid  # (1, N_neurons, 1, 2)
    gx, gy = grid[0, neuron_idx, 0, 0], grid[0, neuron_idx, 0, 1]  # [-1, 1] normalized
    
    # Convert to patch coordinates
    x_patch = int(torch.clamp(((gx + 1) * 0.5 * W_tok), 0, W_tok - 1).item())
    y_patch = int(torch.clamp(((gy + 1) * 0.5 * H_tok), 0, H_tok - 1).item())
    neuron_patch_idx = y_patch * W_tok + x_patch
    
    print(f"Neuron {neuron_idx} at grid ({gx:.2f}, {gy:.2f}) -> patch ({y_patch}, {x_patch}) = {neuron_patch_idx}")      
      
    # Save results
    os.makedirs(outdir, exist_ok=True)
    save_name = f"{target_session}_b{b}_neuron{neuron_idx}_head{head_idx}"
    mappa = attn_maps[:,head_idx,:,:].cpu()
    
    np.save(os.path.join(outdir, f"{save_name}_temporal.npy"), mappa)
    
    print(f"Saved to {outdir}/{save_name}_*.npy")
    
    return mappa

def get_temporal_attention_maps_util(
    tokens: torch.Tensor,
    temporal_blocks,
    layer_idx: int = -1
):
    """
    Compute temporal attention maps for a given token tensor and a list of transformer blocks.
    
    Args:
        tokens: (B, T, P, C) tensor (already tokenized)
        temporal_blocks: list/ModuleList of temporal transformer blocks
        layer_idx: which block index (-1 = final)
        
    Returns:
        attn_weights: (B*T, num_heads, P, P)
    """
    with torch.no_grad():
        x = tokens
        b, t, p, c = x.shape
        
        # Rearrange to match original encoding order
        x = rearrange(x, "b t p c -> (b p) t c")
        
        target_idx = layer_idx if layer_idx >= 0 else len(temporal_blocks) - 1
        
        for idx, block in enumerate(temporal_blocks):
            if idx == target_idx:
                # Same logic as your method
                x_norm = block.norm(x)
                q, k, v, ff = block.fused_linear(x_norm).split(block.fused_dims, dim=-1)

                if block.normalize_qk:
                    q = block.norm_q(q)
                    k = block.norm_k(k)

                q = rearrange(q, "b t (h d) -> b h t d", h=block.num_heads)
                k = rearrange(k, "b t (h d) -> b h t d", h=block.num_heads)

                if block.use_rope:
                    q, k = block.rotary_position_embedding(q=q, k=k)

                q_len = q.size(-2)
                k_len = k.size(-2)

                attn_bias = torch.zeros(q_len, k_len, device=q.device, dtype=q.dtype)

                if block.is_causal:
                    mask = torch.ones(q_len, k_len, device=q.device, dtype=torch.bool).tril(0)
                    attn_bias = attn_bias.masked_fill(~mask, float("-inf"))

                attn = torch.matmul(q * block.scale, k.transpose(-2, -1))
                attn = torch.softmax(attn + attn_bias, dim=-1)

                return attn

            else:
                x = block(x)

    return None
