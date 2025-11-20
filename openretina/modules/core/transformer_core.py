import warnings

import torch
from einops import rearrange
from einops.layers.torch import Rearrange

from openretina.modules.core.base_core import Core
from openretina.modules.layers.attention import ViViT
from openretina.modules.layers.tokenizer import Tokenizer


class ViViTCore(Core):
    def __init__(
        self,
        in_shape: tuple[int, int, int, int],  # (C, T, H, W)
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
        mlp_ratio: float,
        ff_activation: str,
        mha_dropout: float,
        drop_path: float,
        ff_dropout: float,
        use_causal_attention: bool,
        use_sdpa_attention: bool = False,
        use_torch_compile: bool = False,
        reg_scale: float = 0.0,
        head_dim: int | None = None,
        ff_dim: int | None = None,
        normalize_qk: bool = False,
        **kwargs,
    ):
        super(ViViTCore, self).__init__()

        self.input_shape = in_shape
        self.reg_tokens = reg_tokens
        derived_head_dim = head_dim if head_dim is not None else Demb // num_heads
        derived_ff_dim = ff_dim if ff_dim is not None else int(Demb * mlp_ratio)

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
            pos_encoding=pos_encoding,
        )

        self.vivit = ViViT(
            input_shape=self.tokenizer.vivit_input_shape,
            emb_dim=Demb,
            num_heads=num_heads,
            reg_tokens=reg_tokens,
            spatial_depth=num_spatial_blocks,
            temporal_depth=num_temporal_blocks,
            head_dim=derived_head_dim,
            ff_dim=derived_ff_dim,
            ff_activation=ff_activation,
            mha_dropout=mha_dropout,
            ff_dropout=ff_dropout,
            drop_path=drop_path,
            pos_encoding=pos_encoding,
            norm=norm,
            use_causal_attention=use_causal_attention,
            normalize_qk=normalize_qk,
            use_sdpa_attention=use_sdpa_attention,
            reg_scale=reg_scale,
        )

        if use_torch_compile:
            try:
                self.vivit.compile()
            except Exception as exc:
                warnings.warn(
                    f"torch.compile failed for ViViT core ({exc}); continuing without compilation.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Get the spatial dimensions for rearranging
        new_h, new_w = self.tokenizer.new_shape
        self.new_h = new_h
        self.new_w = new_w

        # Rearrange from (b, t, p, c) back to (b, c, t, h, w) format
        self.rearrange = Rearrange("b t (h w) c -> b c t h w", h=new_h, w=new_w)

        # Output shape for readout: (Demb, T, H, W)
        self.output_shape = (
            Demb,
            self.tokenizer.output_shape[0],  # T
            new_h,
            new_w,
        )

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
                        attn_weights = attn_weights[:, :, : -self.reg_tokens, : -self.reg_tokens]

                    return attn_weights
                else:
                    outputs = block(outputs)

        return None

    def regularizer(self):
        return self.vivit.regularizer()

    def forward(self, inputs: torch.Tensor):
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

        # Rearrange back to spatial format: (B, T, H*W, Demb) -> (B, Demb, T, H, W)
        outputs = self.rearrange(outputs)

        return outputs
