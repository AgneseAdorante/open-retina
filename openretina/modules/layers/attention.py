from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from openretina.utils.transformer_utils import DropPath, RotaryPosEmb, get_norm_layer


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        reg_tokens: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        ff_activation: str,
        mha_dropout: float,
        drop_path: float,
        use_rope: bool,
        ff_dropout: float,
        is_causal: bool,
        norm: str,
        normalize_qk: bool,
    ):
        super(TransformerBlock, self).__init__()
        self.input_shape = input_shape
        self.reg_tokens = reg_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ff_dim = ff_dim
        self.ff_activation = ff_activation
        self.mha_dropout = mha_dropout
        self.ff_dropout = ff_dropout
        self.drop_path = drop_path
        self.use_rope = use_rope
        self.is_causal = is_causal
        self.normalize_qk = normalize_qk
        self.norm_type = norm
        self.Demb = input_shape[-1]
        self.emb_dim = input_shape[-1]  # Added: alias for Demb
        self.inner_dim = head_dim * num_heads
        self.register_buffer("scale", torch.tensor(head_dim**-0.5))

    @staticmethod
    def init_weight(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q_len, k_len = q.size(-2), k.size(-2)
        attn_bias = torch.zeros(q_len, k_len, dtype=q.dtype, device=q.device)
        if self.is_causal:
            mask = torch.ones(q_len, k_len, dtype=torch.bool, device=q.device).tril(0)
            attn_bias.masked_fill_(mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn_weights += attn_bias
        attn = torch.softmax(attn_weights, dim=-1)
        outputs = torch.matmul(attn, v)
        outputs = F.dropout(outputs, p=self.mha_dropout, training=self.training)
        return outputs


class ParallelAttentionBlock(TransformerBlock):
    """
    Standard transformer attention block with multi-head self-attention,
    feedforward network, and residual connections.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int],
        reg_tokens: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        ff_activation: str,
        mha_dropout: float,
        drop_path: float,
        use_rope: bool,
        ff_dropout: float,
        is_causal: bool,
        norm: str,
        normalize_qk: bool,
        use_sdpa_attention: bool = False,
    ):
        super(ParallelAttentionBlock, self).__init__(
            input_shape=input_shape,
            reg_tokens=reg_tokens,
            num_heads=num_heads,
            head_dim=head_dim,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            mha_dropout=mha_dropout,
            drop_path=drop_path,
            use_rope=use_rope,
            ff_dropout=ff_dropout,
            is_causal=is_causal,
            norm=norm,
            normalize_qk=normalize_qk,
        )

        # Layer normalization
        self.inner_dim = num_heads * head_dim
        self.emb_dim = self.inner_dim
        self.norm = get_norm_layer(norm, self.emb_dim)

        # Get activation function from string
        if ff_activation.lower() == "gelu":
            ff_activation_fn = nn.GELU
        elif ff_activation.lower() == "relu":
            ff_activation_fn = nn.ReLU
        elif ff_activation.lower() == "elu":
            ff_activation_fn = nn.ELU
        else:
            ff_activation_fn = nn.GELU  # default

        # Fused linear layer for Q, K, V, and FF
        ff_out = ff_dim
        self.fused_dims = (self.inner_dim, self.inner_dim, self.inner_dim, ff_out)
        self.fused_linear = nn.Linear(in_features=self.emb_dim, out_features=sum(self.fused_dims), bias=False)

        # Attention output projection
        self.attn_out = nn.Linear(in_features=self.inner_dim, out_features=self.emb_dim, bias=False)

        # Feedforward output
        self.ff_out = nn.Sequential(
            ff_activation_fn(),
            nn.Dropout(p=ff_dropout),
            nn.Linear(in_features=ff_dim, out_features=self.emb_dim, bias=False),
        )

        # Stochastic depth / drop path (optional - set to identity if not needed)
        self.drop_path1 = DropPath(p=drop_path, mode="row")
        self.drop_path2 = DropPath(p=drop_path, mode="row")
        # Q, K normalization (optional)
        self.normalize_qk = normalize_qk
        if self.normalize_qk:
            self.norm_q = get_norm_layer(norm, self.inner_dim)
            self.norm_k = get_norm_layer(norm, self.inner_dim)
        self.use_sdpa_attention = use_sdpa_attention

        if self.use_rope:
            self.rotary_position_embedding = RotaryPosEmb(
                dim=head_dim, num_tokens=input_shape[0], reg_tokens=reg_tokens
            )

        # Initialize weights
        self.apply(self.init_weight)

    def parallel_attention(self, inputs: torch.Tensor):
        # Normalize input
        outputs = self.norm(inputs)

        # Fused linear projection for Q, K, V, and FF
        q, k, v, ff = self.fused_linear(outputs).split(self.fused_dims, dim=-1)

        # Optionally normalize Q and K
        if self.normalize_qk:
            q, k = self.norm_q(q), self.norm_k(k)

        # Reshape for multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Apply RoPE if enabled
        if self.use_rope:
            q, k = self.rotary_position_embedding(q=q, k=k)

        # Scaled dot-product attention
        if self.use_sdpa_attention:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.mha_dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            attn = self.scaled_dot_product_attention(q, k, v)
        attn = rearrange(attn, "b h n d -> b n (h d)")

        # Parallel residual connections for attention and feedforward
        outputs = inputs + self.drop_path1(self.attn_out(attn)) + self.drop_path2(self.ff_out(ff))
        return outputs

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        outputs = self.parallel_attention(outputs)
        return outputs


class Transformer(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        reg_tokens: int,
        depth: int,
        num_heads: int,
        head_dim: int,
        ff_dim: int,
        drop_path: float,
        use_rope: bool,
        ff_activation: str,
        mha_dropout: float,
        ff_dropout: float,
        is_causal: bool,
        norm: str,
        normalize_qk: bool,
        use_sdpa_attention: bool = False,
    ):
        super(Transformer, self).__init__()
        self.blocks = nn.ModuleList(
            [
                ParallelAttentionBlock(
                    input_shape=input_shape,
                    reg_tokens=reg_tokens,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ff_dim=ff_dim,
                    ff_activation=ff_activation,
                    mha_dropout=mha_dropout,
                    drop_path=drop_path,
                    use_rope=use_rope,
                    ff_dropout=ff_dropout,
                    is_causal=is_causal,
                    norm=norm,
                    normalize_qk=normalize_qk,
                    use_sdpa_attention=use_sdpa_attention,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, inputs: torch.Tensor):
        outputs = inputs
        for block in self.blocks:
            outputs = block(outputs)
        return outputs


class ViViT(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        *,
        emb_dim: int,
        num_heads: int,
        reg_tokens: int,
        spatial_depth: int,
        temporal_depth: int,
        head_dim: int,
        ff_dim: int,
        ff_activation: str,
        mha_dropout: float,
        ff_dropout: float,
        drop_path: float,
        pos_encoding: int,
        norm: str,
        use_causal_attention: bool,
        normalize_qk: bool = False,
        use_sdpa_attention: bool = False,
        reg_scale: float = 0.0,
        verbose: int = 0,
    ):
        super(ViViT, self).__init__()
        self.reg_tokens = reg_tokens
        self.reg_s_tokens = nn.Parameter(torch.randn(self.reg_tokens, emb_dim))
        self.reg_t_tokens = nn.Parameter(torch.randn(self.reg_tokens, emb_dim))

        # Spatial transformer processes each time step independently
        # Input: (num_spatial_patches, emb_dim)
        self.spatial_transformer = Transformer(
            input_shape=(input_shape[1], emb_dim),  # (num_patches, emb_dim)
            reg_tokens=self.reg_tokens,
            depth=spatial_depth,
            num_heads=num_heads,
            drop_path=drop_path,
            head_dim=head_dim,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            mha_dropout=mha_dropout,
            ff_dropout=ff_dropout,
            use_rope=pos_encoding == 5,
            is_causal=False,
            norm=norm,
            normalize_qk=normalize_qk,
            use_sdpa_attention=use_sdpa_attention,
        )

        # Temporal transformer processes each spatial location across time
        # Input: (num_temporal_patches, emb_dim)
        self.temporal_transformer = Transformer(
            input_shape=(input_shape[0], emb_dim),  # (num_time_patches, emb_dim)
            reg_tokens=self.reg_tokens,
            depth=temporal_depth,
            num_heads=num_heads,
            drop_path=drop_path,
            use_rope=pos_encoding in (5, 6, 7),
            head_dim=head_dim,
            ff_dim=ff_dim,
            ff_activation=ff_activation,
            mha_dropout=mha_dropout,
            ff_dropout=ff_dropout,
            is_causal=use_causal_attention,
            norm=norm,
            normalize_qk=normalize_qk,
            use_sdpa_attention=use_sdpa_attention,
        )

        self.output_shape = input_shape

        # Optional L2 regularization
        self.reg_scale = reg_scale

    def compile(self):
        """Compile spatial and temporal transformer modules"""
        print("torch.compile spatial and temporal transformers in ViViT")
        self.spatial_transformer = torch.compile(
            self.spatial_transformer,
            fullgraph=True,
        )
        self.temporal_transformer = torch.compile(
            self.temporal_transformer,
            fullgraph=True,
        )

    def regularizer(self):
        """Optional L2 regularization scoped to ViViT parameters."""
        if self.reg_scale <= 0.0:
            return 0.0
        return self.reg_scale * sum(p.pow(2).sum() for p in self.parameters())

    def add_reg_tokens(self, tokens: torch.Tensor):
        b, t, p, _ = tokens.shape
        # append spatial register tokens
        tokens = torch.cat((tokens, repeat(self.reg_s_tokens, "r c -> b t r c", b=b, t=t)), dim=2)
        p += self.reg_tokens
        # append temporal register tokens
        tokens = torch.cat((tokens, repeat(self.reg_t_tokens, "r c -> b r p c", b=b, p=p)), dim=1)
        return tokens

    def remove_reg_tokens(self, tokens: torch.Tensor):
        return tokens[:, : -self.reg_tokens, : -self.reg_tokens, :]

    def add_spatial_reg_tokens(self, tokens: torch.Tensor):
        b, t, p, _ = tokens.shape
        tokens = torch.cat((tokens, repeat(self.reg_s_tokens, "r c -> b t r c", b=b, t=t)), dim=2)
        return tokens

    def remove_spatial_reg_tokens(self, tokens: torch.Tensor):
        return tokens[:, :, : -self.reg_tokens, :]

    def add_temporal_reg_tokens(self, tokens: torch.Tensor):
        b, t, p, _ = tokens.shape
        tokens = torch.cat((tokens, repeat(self.reg_t_tokens, "r c -> b r p c", b=b, p=p)), dim=1)
        return tokens

    def remove_temporal_reg_tokens(self, tokens: torch.Tensor):
        return tokens[:, : -self.reg_tokens, :, :]

    def forward(self, inputs: torch.Tensor):
        """
        Input: (B, T, P, C) where T=temporal patches, P=spatial patches, C=emb_dim
        Output: (B, T, P, C)
        """
        outputs = inputs
        b, t, p, _ = outputs.shape

        if self.reg_tokens:
            outputs = self.add_spatial_reg_tokens(outputs)
        # Spatial attention: process each time step independently
        # Reshape to process all time steps in batch
        outputs = rearrange(outputs, "b t p c -> (b t) p c")
        outputs = self.spatial_transformer(outputs)
        outputs = rearrange(outputs, "(b t) p c -> b t p c", b=b)

        if self.reg_tokens:
            outputs = self.remove_spatial_reg_tokens(outputs)
            outputs = self.add_temporal_reg_tokens(outputs)
        # Temporal attention: process each spatial location across time
        # Reshape to process all spatial locations in batch
        outputs = rearrange(outputs, "b t p c -> (b p) t c")
        outputs = self.temporal_transformer(outputs)
        outputs = rearrange(outputs, "(b p) t c -> b t p c", b=b)
        if self.reg_tokens:
            outputs = self.remove_temporal_reg_tokens(outputs)

        return outputs
