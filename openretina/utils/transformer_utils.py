import math
import os
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import torch
from einops import einsum, rearrange
from lightning.pytorch.callbacks import Callback
from torch import nn
from torchvision.ops import stochastic_depth


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


class SparseAttentionViz(Callback):
    def __init__(self, outdir, n_layers=1, device="cuda", head_limit=None):
        super().__init__()
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.n_layers = n_layers  # Number of last layers to extract
        self.device = device
        self.head_limit = head_limit
        print(
            f"[SparseAttentionViz] Initialized with outdir={outdir}, n_layers={n_layers}, "
            f"device={device}, head_limit={head_limit}"
        )

    def _find_core(self, pl_module):
        for name in ["core", "core_wrapper", "core_readout"]:
            if hasattr(pl_module, name):
                obj = getattr(pl_module, name)
                if hasattr(obj, "tokenizer") and hasattr(obj, "get_spatial_attention_maps"):
                    print(f"[SparseAttentionViz] Found core: {name}")
                    return obj
        if hasattr(pl_module, "module"):
            return self._find_core(pl_module.module)
        raise RuntimeError("No core with tokenizer+get_spatial_attention_maps found")

    def on_train_end(self, trainer, pl_module):
        # Run at the very end of training (works with early stopping)
        print(f"[SparseAttentionViz] Running visualization at end of training (epoch {trainer.current_epoch})")

        # Safely get a batch from the first val dataloader
        try:
            session_name, batch = next(iter(trainer.val_dataloaders))
            print(f"[SparseAttentionViz] Got batch from session: {session_name}")
        except Exception as e:
            print(f"[SparseAttentionViz] Failed to get validation batch: {e}")
            return

        # Extract frames (videos) from batch.inputs
        frames = getattr(batch, "inputs", None)
        if frames is None or not torch.is_tensor(frames) or frames.ndim != 5:
            print(
                f"[SparseAttentionViz] batch.inputs not found or not 5D, got {type(frames)}"
                f" with shape {getattr(frames, 'shape', None)}"
            )
            return

        frames = frames.to(self.device)
        B, C, T, H0, W0 = frames.shape
        print(f"[SparseAttentionViz] Frames shape: {frames.shape}")

        # pick random b,t for display
        b = torch.randint(0, B, ()).item()
        t = torch.randint(0, T, ()).item()
        print(f"[SparseAttentionViz] Selected random indices b={b}, t={t}")

        # find core
        core = self._find_core(pl_module)
        core.eval()

        # Create output folder for this visualization
        viz_folder = os.path.join(self.outdir, f"epoch{trainer.current_epoch:03d}_{session_name}_b{b}_t{t}")
        os.makedirs(viz_folder, exist_ok=True)
        print(f"[SparseAttentionViz] Created folder: {viz_folder}")

        # Get original frame for overlay
        frame_np = frames[b, :, t].cpu().numpy()
        frame_disp = frame_np[0]

        # Save the original frame
        fig_orig, ax_orig = plt.subplots(1, 1, figsize=(6, 6))
        ax_orig.imshow(frame_disp if C > 1 else frame_disp, cmap="gray" if C == 1 else None, vmin=0, vmax=1)
        ax_orig.set_title("Original Frame")
        ax_orig.axis("off")
        orig_path = os.path.join(viz_folder, "original_frame.png")
        plt.savefig(orig_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig_orig)
        print(f"[SparseAttentionViz] Saved original frame to {orig_path}")

        # Extract attention from last n_layers
        all_layer_attns = []
        all_layer_imps = []

        with torch.no_grad():
            # Tokenize once
            tokens = core.tokenizer(frames)  # (B, T, P, C)

            # Get total number of layers (assuming get_spatial_attention_maps can handle this)
            # First, get attention from layer -1 to determine total layers
            attn_test = core.get_spatial_attention_maps(tokens, layer_idx=-1)
            if attn_test is None:
                print("[SparseAttentionViz] Attention maps returned None")
                return

            # Extract attention from last n_layers
            for layer_offset in range(self.n_layers):
                layer_idx = -(layer_offset + 1)  # -1, -2, -3, ...
                print(f"[SparseAttentionViz] Extracting attention from layer {layer_idx}")

                attn = core.get_spatial_attention_maps(tokens, layer_idx=layer_idx)
                if attn is None:
                    print(f"[SparseAttentionViz] Attention maps returned None for layer {layer_idx}")
                    continue

                print(f"[SparseAttentionViz] Layer {layer_idx} attention shape: {attn.shape}")

                # Pick the same random frame from attention maps
                bt_index = torch.randint(0, attn.shape[0], ()).item()
                attn = attn[bt_index]  # (n_heads, P, P)
                n_heads = attn.shape[0]
                if self.head_limit:
                    n_heads = min(n_heads, self.head_limit)
                    attn = attn[:n_heads]

                # convert attention to spatial importance
                P = attn.shape[-1]
                h_patch = core.new_h
                w_patch = core.new_w
                if P != h_patch * w_patch:
                    print(
                        f"[SparseAttentionViz] Warning: P={P} does not match h_patch*w_patch={h_patch * w_patch},"
                        " using sqrt(P) for visualization"
                    )
                    h_patch = w_patch = int(P**0.5)

                query_token = torch.randint(0, P, ()).item()
                imp = attn[:, query_token, :].view(n_heads, h_patch, w_patch)
                imp = imp - imp.amin(dim=(1, 2), keepdim=True)
                denom = imp.amax(dim=(1, 2), keepdim=True)
                denom[denom == 0] = 1
                imp = imp / denom

                # Upsample to original resolution
                imp_up = (
                    torch.nn.functional.interpolate(
                        imp.unsqueeze(1), size=(H0, W0), mode="bilinear", align_corners=False
                    )
                    .squeeze(1)
                    .cpu()
                    .numpy()
                )

                all_layer_attns.append(attn)
                all_layer_imps.append(imp_up)

        if not all_layer_imps:
            print("[SparseAttentionViz] No attention maps extracted")
            return

        n_layers_extracted = len(all_layer_imps)
        n_heads = all_layer_imps[0].shape[0]
        print(f"[SparseAttentionViz] Extracted {n_layers_extracted} layers with {n_heads} heads each")

        # Save individual images per layer and head
        for layer_idx, imp_up in enumerate(all_layer_imps):
            for head_idx in range(n_heads):
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(frame_disp if C > 1 else frame_disp, cmap="gray" if C == 1 else None, vmin=0, vmax=1)
                ax.imshow(imp_up[head_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1)
                ax.set_title(f"Layer {-(layer_idx + 1)} - Head {head_idx}")
                ax.axis("off")

                out_path = os.path.join(viz_folder, f"layer{layer_idx:02d}_head{head_idx:02d}.png")
                plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

        print(f"[SparseAttentionViz] Saved {n_layers_extracted * n_heads} individual attention maps")

        # Create comprehensive subplot: original + all layers x all heads
        fig, axes = plt.subplots(n_layers_extracted, n_heads + 1, figsize=(3 * (n_heads + 1), 3 * n_layers_extracted))

        # Handle single layer case
        if n_layers_extracted == 1:
            axes = axes.reshape(1, -1)

        for layer_idx in range(n_layers_extracted):
            # First column: original frame
            ax = axes[layer_idx, 0]
            ax.imshow(frame_disp if C > 1 else frame_disp, cmap="gray" if C == 1 else None, vmin=0, vmax=1)
            ax.set_title(f"Layer {-(layer_idx + 1)}\nOriginal")
            ax.axis("off")

            # Remaining columns: attention heads
            imp_up = all_layer_imps[layer_idx]
            for head_idx in range(n_heads):
                ax = axes[layer_idx, head_idx + 1]
                ax.imshow(frame_disp if C > 1 else frame_disp, cmap="gray" if C == 1 else None, vmin=0, vmax=1)
                ax.imshow(imp_up[head_idx], cmap="jet", alpha=0.5, vmin=0, vmax=1)
                ax.set_title(f"Head {head_idx}")
                ax.axis("off")

        subplot_path = os.path.join(viz_folder, "all_layers_heads_grid.png")
        plt.savefig(subplot_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"[SparseAttentionViz] Saved comprehensive grid to {subplot_path}")

        print(f"[SparseAttentionViz] Visualization complete in folder: {viz_folder}")


def get_norm_layer(norm_type: str, normalized_shape: int | Tuple[int, ...]) -> nn.Module:
    norm_key = norm_type.lower()
    if norm_key in ("layernorm", "layer_norm", "ln"):
        return nn.LayerNorm(normalized_shape)
    if norm_key in ("rmsnorm", "rsmnorm", "rms_norm", "rsm_norm"):
        return nn.RMSNorm(normalized_shape)
    raise ValueError(f"Unsupported normalization type '{norm_type}'.")
