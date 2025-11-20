import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from openretina.utils.transformer_utils import (
    SinCosPosEmb,
    SinusoidalPosEmb,
    get_norm_layer,
)


class Tokenizer(nn.Module):
    """
    Tokenizes video clips into 3D spatio-temporal patches.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        patch_size: int,
        temporal_patch_size: int,
        temporal_stride: int,
        spatial_stride: int,
        Demb: int,
        ptoken: float,
        pad_frame: bool,
        norm: str = "layernorm",
        patch_mode: int = 1,
        pos_encoding: int = 2,
    ):
        super(Tokenizer, self).__init__()
        self.input_shape = input_shape
        self.ptoken = ptoken
        self.patch_mode = patch_mode
        self.norm_type = norm  # Renamed to avoid confusion with nn.LayerNorm instance

        """
        Patch mode:
        0: extract 3D patches via tensor.unfold followed by linear projection
        1: extract 3D patches via a 3D convolution layer
        """

        c, t, h, w = input_shape

        h_pad = self.pad_size(h, patch_size, spatial_stride)
        w_pad = self.pad_size(w, patch_size, spatial_stride)
        t_pad = self.pad_size(t, temporal_patch_size, temporal_stride)

        self.pad = None
        if pad_frame:
            self.padding = (
                w_pad // 2,  # padding left
                w_pad - w_pad // 2,  # padding right
                h_pad // 2,  # padding top
                h_pad - h_pad // 2,  # padding bottom
                t_pad,  # padding front
                0,  # padding back
            )
            self.pad = nn.ZeroPad3d(self.padding)

            w = w + self.padding[0] + self.padding[1]
            h = h + self.padding[2] + self.padding[3]
            t = t + self.padding[4] + self.padding[5]

        new_t = self.unfold_size(t, temporal_patch_size, stride=temporal_stride)
        new_h = self.unfold_size(h, patch_size, stride=spatial_stride)
        new_w = self.unfold_size(w, patch_size, stride=spatial_stride)

        self.kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.stride = (temporal_stride, spatial_stride, spatial_stride)

        if self.patch_mode == 0:
            # Mode 0: Unfold + Linear projection
            patch_dim = int(c * np.prod(self.kernel_size))
            self.norm = get_norm_layer(norm, patch_dim)
            self.linear = nn.Linear(in_features=patch_dim, out_features=Demb)
            self.proj = None

        else:
            # Mode 1: 3D Convolution
            self.proj = nn.Conv3d(
                in_channels=c, out_channels=Demb, kernel_size=self.kernel_size, stride=self.stride, bias=False
            )

            self.norm = get_norm_layer(norm, Demb)
            self.linear = None

        # ===== ADD THESE ATTRIBUTES =====
        # Store the output shape information
        self.new_shape = (new_h, new_w)  # Spatial dimensions after patching
        self.output_shape = (new_t, new_h * new_w, Demb)  # (num_temporal_patches, num_spatial_patches, embed_dim)

        # Alternative format for ViViT that expects (T, num_patches, Demb)
        # But after the rearrange in forward, it becomes (B, T*P, Demb)
        # So we need to store what ViViT actually expects
        num_total_patches = new_t * new_h * new_w
        self.flat_output_shape = (num_total_patches, Demb)  # After flattening

        # For ViViT compatibility (before flattening temporal and spatial)
        self.vivit_input_shape = (new_t, new_h * new_w, Demb)
        # ================================

        # ===== ADD INSIDE __init__ AFTER self.vivit_input_shape =====

        self.pos_encoding = pos_encoding  # expect user to pass an int

        match self.pos_encoding:
            case 1:
                self.pos_embedding = nn.Parameter(torch.randn(1, new_t, new_h * new_w, Demb))
            case 2:
                self.spatial_pos_embedding = nn.Parameter(torch.randn(1, 1, new_h * new_w, Demb))
                self.temporal_pos_embedding = nn.Parameter(torch.randn(1, new_t, 1, Demb))
            case 3:
                self.spatial_pos_embedding = nn.Parameter(torch.randn(1, 1, new_h * new_w, Demb))
                self.temporal_pos_encoding = SinusoidalPosEmb(
                    d_model=Demb,
                    max_length=new_t,
                    dimension="temporal",
                    dropout=0.0,
                )
            case 4:
                self.spatial_pos_embedding = SinusoidalPosEmb(
                    d_model=Demb,
                    max_length=new_h * new_w,
                    dimension="spatial",
                    dropout=0.0,
                )
                self.temporal_pos_encoding = SinusoidalPosEmb(
                    d_model=Demb,
                    max_length=new_t,
                    dimension="temporal",
                    dropout=0.0,
                )
            case 5:
                pass  # Uses only RoPE, no additional positional encoding.
            case 6:
                self.spatial_pos_embedding = nn.Parameter(torch.randn(1, 1, new_h * new_w, Demb))
            case 7:
                self.spatial_pos_embedding = SinCosPosEmb(emb_dim=Demb, input_shape=(new_h, new_w))

    @staticmethod
    def pad_size(dim: int, patch_size: int, stride: int = 1):
        """Compute the zero padding needed to cover the entire dimension"""
        return (math.ceil(dim / stride) - 1) * stride + patch_size - dim

    @staticmethod
    def unfold_size(dim: int, patch_size: int, stride: int = 1):
        return math.floor(((dim - patch_size) / stride) + 1)

    def forward(self, inputs: torch.Tensor):
        """
        Input:
            inputs: (B, C, T, H, W)
        Output:
            outputs: (B, T, num_spatial_patches, Demb)
        """

        outputs = inputs

        # Apply padding if needed
        if self.pad is not None:
            outputs = self.pad(outputs)

        if self.patch_mode == 0:
            # Mode 0: Unfold + Linear projection
            # Unfold in 3D: extract patches
            outputs = outputs.unfold(2, size=self.kernel_size[0], step=self.stride[0])
            outputs = outputs.unfold(3, size=self.kernel_size[1], step=self.stride[1])
            outputs = outputs.unfold(4, size=self.kernel_size[2], step=self.stride[2])

            # outputs shape: (B, C, nt, nh, nw, pt, ph, pw)

            # Rearrange to (B, nt, nh*nw, C*pt*ph*pw)
            outputs = rearrange(outputs, "b c nt nh nw pt ph pw -> b nt (nh nw) (c pt ph pw)")

            # Apply normalization
            outputs = self.norm(outputs)

            # Linear projection to embedding dimension
            outputs = self.linear(outputs)

        else:
            # Mode 1: 3D Convolution
            # Extract patches and project via 3D convolution
            # Input: (B, C, T, H, W) -> Output: (B, Demb, nt, nh, nw)
            outputs = self.proj(outputs)

            B, Demb, nt, nh, nw = outputs.shape

            # Reshape to (B, nt, nh*nw, Demb)
            outputs = outputs.permute(0, 2, 3, 4, 1)  # (B, nt, nh, nw, Demb)
            outputs = outputs.reshape(B, nt, nh * nw, Demb)

            # Apply normalization
            outputs = self.norm(outputs)

        # Output at this point: (B, nt, nh*nw, Demb)
        # This is what ViViT expects!

        # Optional dropout
        if self.training and self.ptoken > 0:
            outputs = self.apply_patch_dropout(outputs)

        # outputs: (B, nt, nh*nw, Demb)
        B, t, p, _ = outputs.shape
        max_t = max(t, 750)

        match self.pos_encoding:
            case 1:
                outputs = outputs + self.pos_embedding[:, :max_t, :p, :]
            case 2:
                outputs = (
                    outputs + self.spatial_pos_embedding[:, :, :p, :] + self.temporal_pos_embedding[:, :max_t, :, :]
                )
            case 3:
                outputs = outputs + self.spatial_pos_embedding[:, :, :p, :]
                outputs = self.temporal_pos_encoding(outputs)
            case 4:
                outputs = self.spatial_pos_embedding(outputs)
                outputs = self.temporal_pos_encoding(outputs)
            case 6:
                outputs = outputs + self.spatial_pos_embedding[:, :, :p, :]
            case 7:
                outputs = self.spatial_pos_embedding(outputs)

        return outputs

    def apply_patch_dropout(self, x: torch.Tensor):
        """Drop random patches during training."""
        B, T, P, D = x.shape
        mask = torch.rand(B, T, P, 1, device=x.device) > self.ptoken
        return x * mask
