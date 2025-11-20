import torch
import torch.nn as nn
from typing import Tuple, Any
import warnings
from einops import rearrange
#sono cambiati!!
from openretina.models.new_tokenizer import Tokenizer
from openretina.models.spatial_temporal_trans import ViViT

from einops.layers.torch import Rearrange
from openretina.modules.core.base_core import Core
import matplotlib.pyplot as plt
import os
import numpy as np
import math
from openretina.utils.transformer_utils import temporal_moving_average
from typing import Any


class ViViTCoreWrapper(Core):
 
    def __init__(
            self,
            in_shape: tuple[int, int, int, int, int],  # (B, C, T, H, W)
            patch_size: int,
            temporal_patch_size: int,
            spatial_stride: int,
            temporal_stride: int,
            Demb: int,
            ptoken: float,
            pad_frame: bool,
            norm: str,
            patch_mode: bool,
            pos_encoding: int, 
            num_heads: int,
            reg_tokens: int,
            num_spatial_blocks: int,
            num_temporal_blocks: int,
            dropout: float,
            mlp_ratio: float,
            channels: int,
            spatial_depth:int,
            temporal_depth: int,
            head_dim: int,
            ff_dim: int,
            ff_activation: str,
            mha_dropout: float,
            drop_path: float,
            use_rope: bool,
            ff_dropout: float,
            use_causal_attention: bool,
            reg_lambda: float,
            activation_lambda: float,
            smooth_lambda: float,
            **kwargs  # Catches _target_, _convert_, and any other Hydra internals
        ):
        super(ViViTCoreWrapper, self).__init__()
        
        self.input_shape = in_shape
        self.reg_tokens = reg_tokens
        self.reg_lambda = reg_lambda
        self.activation_lambda = activation_lambda  # activation penalty strength
        self.smooth_lambda = smooth_lambda    # temporal smoothness penalty

        
        print("1. Creating Tokenizer...")
        
        self.tokenizer = Tokenizer(
            input_shape=in_shape,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            Demb=Demb,
            ptoken=ptoken,
            pad_frame=bool(pad_frame),
            norm=norm,
            patch_mode=patch_mode,
            pos_encoding=pos_encoding
        )

        print(f"2. Tokenizer created. Output shape: {self.tokenizer.output_shape}")
        
# Create args object for ViViT
        from types import SimpleNamespace
        args = SimpleNamespace(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            Demb=Demb,
            ptoken=ptoken,
            pad_frame=pad_frame,
            norm=norm,
            patch_mode=patch_mode,
            pos_encoding = pos_encoding,
            num_heads=num_heads,
            reg_tokens = reg_tokens,
            num_spatial_blocks=num_spatial_blocks,
            num_temporal_blocks=num_temporal_blocks,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            channels=channels,
            spatial_depth = spatial_depth,
            temporal_depth=temporal_depth,
            head_dim=head_dim,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            drop_path = drop_path,
            use_rope = use_rope,
            mha_dropout=mha_dropout,
            ff_dropout=ff_dropout,
            use_causal_attention=use_causal_attention,
            reg_lambda = reg_lambda,
            activation_lambda = activation_lambda,
            smooth_lambda = smooth_lambda
        )
        
        # Use the vivit_input_shape which is (T, num_patches, Demb)
        # This matches what ViViT expects
        self.vivit = ViViT(
            args,
            input_shape=self.tokenizer.vivit_input_shape,  # Use vivit_input_shape
        )
        
        print(f"3. ViViT created with input shape: {self.tokenizer.vivit_input_shape}")

        # Get the spatial dimensions for rearranging
        new_h, new_w = self.tokenizer.new_shape
        self.new_h = new_h
        self.new_w = new_w
        
        print(f"4. Spatial shape after patching: h={new_h}, w={new_w}")
        
        # Rearrange from (b, t, p, c) back to (b, c, t, h, w) format
        self.rearrange = Rearrange("b t (h w) c -> b c t h w", h=new_h, w=new_w)
        self.activation = nn.ELU()  # Note: should be nn.ELU() not nn.ELU

        # Output shape for readout: (Demb, T, H, W)
        self.output_shape = (
            Demb,
            self.tokenizer.output_shape[0],  # T
            new_h,
            new_w,
        )
        
        print(f"5. Core output shape: {self.output_shape}")

    def get_spatial_attention_maps(self, inputs: torch.Tensor, layer_idx: int = -1):
        """
        Extract spatial attention maps from a specific layer.
        
        Args:
            inputs: (B, T, P, C) tensor (already tokenized)
            layer_idx: which spatial transformer layer (-1 for last)
        
        Returns:
            attention_map: (B*T, num_heads, P, P)
        """
        self.eval()
        with torch.no_grad():
            outputs = inputs
            b, t, p, _ = outputs.shape
            
            if self.reg_tokens:
                outputs = self.vivit.add_spatial_reg_tokens(outputs)
            
            outputs = rearrange(outputs, "b t p c -> (b t) p c")
            
            target_idx = layer_idx if layer_idx >= 0 else len(self.vivit.spatial_transformer.blocks) - 1
            
            for idx, block in enumerate(self.vivit.spatial_transformer.blocks):
                if idx == target_idx:
                    x = block.norm(outputs)
                    q, k, v, ff = block.fused_linear(x).split(block.fused_dims, dim=-1)
                    
                    if block.normalize_qk:
                        q, k = block.norm_q(q), block.norm_k(k)
                    
                    q = rearrange(q, "bt p (h d) -> bt h p d", h=block.num_heads)
                    k = rearrange(k, "bt p (h d) -> bt h p d", h=block.num_heads)
                    
                    if block.use_rope:
                        q, k = block.rotary_position_embedding(q=q, k=k)
                    
                    attn_weights = torch.matmul(q * block.scale, k.transpose(-2, -1))
                    attn_weights = torch.softmax(attn_weights, dim=-1)
                    
                    if self.reg_tokens:
                        attn_weights = attn_weights[:, :, :-self.reg_tokens, :-self.reg_tokens]
                    
                    return attn_weights
                else:
                    outputs = block(outputs)
        
        return None
    
    def regularizer(self, activations=None):
        reg_loss = 0.0

        # ----- 1. L2 weight penalty -----
        for param in self.parameters():
            reg_loss += torch.sum(param ** 2)
        reg_loss *= self.reg_lambda

        if activations is not None:
            # ----- 2. L2 activation penalty -----
            reg_loss += self.activation_lambda * torch.sum(activations ** 2)

            # ----- 3. Temporal smoothness -----
            # assumes activations: (batch, channels, time, H, W)
            if activations.shape[2] > 1:  # check time dimension
                diff = activations[..., 1:, :, :] - activations[..., :-1, :, :]
                smooth_loss = torch.mean(diff ** 2)
                reg_loss += self.smooth_lambda * smooth_loss

        return reg_loss

    def forward(
        self,
        inputs: torch.Tensor
    ):
        """
        Input: (B, C, T, H, W)
        Output: (B, Demb, T, H, W) where T, H, W are reduced based on patching
        """
        outputs = inputs
        
        # Tokenize: (B, C, T, H, W) -> (B, nt, nh*nw, Demb)
        outputs = self.tokenizer(outputs)
        
        # ViViT expects (B, T, P, Demb) which is what tokenizer outputs
        # ViViT processes and returns same shape
        outputs = self.vivit(outputs)
 
        if self.smooth_lambda > 0:
            outputs = temporal_moving_average(outputs, kernel_size=21) * self.smooth_lambda + outputs * (1 - self.smooth_lambda)

        # Rearrange back to spatial format: (B, T, H*W, Demb) -> (B, Demb, T, H, W)
        outputs = self.rearrange(outputs)
        
        # Apply activation
        outputs = self.activation(outputs)
        
        return outputs