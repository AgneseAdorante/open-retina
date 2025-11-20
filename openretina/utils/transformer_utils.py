from torch import nn
from torchvision.ops import stochastic_depth
import torch
from typing import Tuple
from einops import einsum
from typing import Literal
import math
from einops import rearrange
import os
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback


class DropPath(nn.Module):
    """Stochastic depth for regularization https://arxiv.org/abs/1603.09382"""

    def __init__(self, p: float = 0.0, mode: str = "row"):
        super(DropPath, self).__init__()
        assert 0 <= p <= 1
        assert mode in ("batch", "row")
        self.p, self.mode = p, mode

    def forward(self, inputs: torch.Tensor):
        return stochastic_depth(
            inputs, p=self.p, mode=self.mode, training=self.training
        )
    
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
        self.register_buffer(
            "scale", (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        )
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
        n, device = q.size(2), q.device.type
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
        actual_max_length = max(max_length, 10000)
        
        position = torch.arange(actual_max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
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
                raise NotImplementedError(
                    f"invalid dimension {self.dimension} in "
                    f"SinusoidalPositionalEncoding"
                )

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
    def __init__(self, outdir, n_layers=1, device='cuda', head_limit=None, target_session=None):
        super().__init__()
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.n_layers = n_layers  # Number of last layers to extract
        self.device = device
        self.head_limit = head_limit
        self.target_session = target_session  # Session name string to visualize (or None for first)
        print(f"[SparseAttentionViz] Initialized with outdir={outdir}, n_layers={n_layers}, device={device}, head_limit={head_limit}, target_session={target_session}")

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
        
        # Iterate through validation dataloader to find target session
        val_dataloaders = trainer.val_dataloaders
        if not isinstance(val_dataloaders, list):
            val_dataloaders = [val_dataloaders]
        
        session_name = None
        batch = None
        found = False
        
        # Try to find the target session
        for val_loader in val_dataloaders:
            try:
                for item in val_loader:
                    current_session_name = item[0]  # str with session name
                    current_batch = item[1]  # batch object with .inputs and .targets
                    
                    # If target_session is None, take the first one
                    # If target_session is specified, match it
                    if self.target_session is None or current_session_name == self.target_session:
                        session_name = current_session_name
                        batch = current_batch
                        found = True
                        print(f"[SparseAttentionViz] Found target session: {session_name}")
                        break
                
                if found:
                    break
                    
            except Exception as e:
                print(f"[SparseAttentionViz] Error iterating dataloader: {e}")
                continue
        
        if not found or session_name is None or batch is None:
            if self.target_session is not None:
                print(f"[SparseAttentionViz] Could not find target session '{self.target_session}'")
            else:
                print(f"[SparseAttentionViz] Could not get any validation batch")
            return

        # Extract frames (videos) from batch.inputs
        frames = getattr(batch, "inputs", None)
        if frames is None or not torch.is_tensor(frames) or frames.ndim != 5:
            print(f"[SparseAttentionViz] batch.inputs not found or not 5D, got {type(frames)} with shape {getattr(frames,'shape',None)}")
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
        ax_orig.imshow(frame_disp if C > 1 else frame_disp, cmap='gray' if C == 1 else None, vmin=0, vmax=1)
        ax_orig.set_title("Original Frame")
        ax_orig.axis("off")
        orig_path = os.path.join(viz_folder, "original_frame.png")
        plt.savefig(orig_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig_orig)
        print(f"[SparseAttentionViz] Saved original frame to {orig_path}")

        ###############################################################
        # EXTRACT 30-FRAME SEQUENCE FROM LAST LAYER, ONE HEAD
        ###############################################################
        with torch.no_grad():
            # Pick random center frame
            t0 = torch.randint(0, T, ()).item()

            t_start = max(0, t0 - 14)
            t_end   = min(T, t0 + 16)     # exclusive → gives 15 frames after t0

            window_len = t_end - t_start
            if window_len < 30:
                if t_start == 0:
                    t_end = min(T, 30)
                else:
                    t_start = max(0, T - 30)
                window_len = t_end - t_start

            if window_len != 30:
                print(f"[SparseAttentionViz] Expected 30 frames, got {window_len}, aborting.")
                return

            print(f"[SparseAttentionViz] Using frames [{t_start}:{t_end}) around center frame t0={t0}")

            # EFFICIENCY IMPROVEMENT: Tokenize only the frames we need
            frames_window = frames[b:b+1, :, t_start:t_end]  # (1, C, 30, H0, W0)
            tokens = core.tokenizer(frames_window)  # (1, 30, P, C)

            # Extract attention once for all frames in the window
            attn_full = core.get_spatial_attention_maps(tokens, layer_idx=-1)
            if attn_full is None:
                print("[SparseAttentionViz] Attention maps returned None")
                return

            ###############################################
            # Extract Gaussian readout grid (mu locations)
            ###############################################
            readout = pl_module.readout
            session_readout = readout[session_name]

            # sample=False → deterministic grid (means μ)
            grid = session_readout.sample_grid(batch_size=1, sample=False)
            # has shape: (1, outdims, 1, 2)

            gx = grid[0, :, 0, 0].cpu()      # x coords in [-1,1]
            gy = grid[0, :, 0, 1].cpu()      # y coords in [-1,1]

            # convert normalized coord → patch index
            px = ((gx + 1) * 0.5 * (core.new_w - 1)).round().long()
            py = ((gy + 1) * 0.5 * (core.new_h - 1)).round().long()

            # linear attention token index for each neuron
            query_idx_per_neuron = py * core.new_w + px   # shape (outdims,)

            # Force one head only
            head_idx = 0

            # Prepare arrays for original frames and overlaid frames
            original_frames = []
            overlaid_frames = []

            # Loop through each frame in window
            for frame_idx, t in enumerate(range(t_start, t_end)):
                # Now attn_full has shape (1*30, n_heads, P, P) = (30, n_heads, P, P)
                attn_bt = attn_full[frame_idx]  # (n_heads, P, P)
                if attn_bt.ndim != 3:
                    print("[SparseAttentionViz] Unexpected attn shape:", attn_bt.shape)
                    continue

                if head_idx >= attn_bt.shape[0]:
                    print("[SparseAttentionViz] Requested head 0 but fewer heads exist.")
                    return

                attn_head = attn_bt[head_idx]  # (P, P)

                P = attn_head.shape[-1]
                h_patch = core.new_h
                w_patch = core.new_w

                # Query token for neuron 0
                neuron = 100
                query_token = query_idx_per_neuron[neuron].item()

                imp = attn_head[query_token].view(h_patch, w_patch)

                # Normalize
                imp = imp - imp.min()
                mx = imp.max()
                if mx > 0:
                    imp = imp / mx

                # Upsample to original frame size
                imp_up = torch.nn.functional.interpolate(
                    imp[None, None, :, :],
                    size=(H0, W0),
                    mode="bilinear",
                    align_corners=False
                ).squeeze().cpu().numpy()

                # Get original frame
                frame_np = frames[b, :, t].cpu().numpy()
                base = frame_np[0]  # Shape: (H0, W0)

                # Store original frame
                original_frames.append(base)

                # Create overlaid frame by blending attention map with original
                # Convert attention map to RGB using 'hot' colormap
                import matplotlib.cm as cm
                hot_cmap = cm.get_cmap('hot')
                imp_colored = hot_cmap(imp_up)[:, :, :3]  # RGB, drop alpha

                # Blend: overlaid = (1-alpha)*base + alpha*attention
                alpha = 0.45
                if base.ndim == 2:  # Grayscale
                    base_rgb = np.stack([base, base, base], axis=-1)
                else:
                    base_rgb = base
                
                overlaid = (1 - alpha) * base_rgb + alpha * imp_colored
                overlaid = np.clip(overlaid, 0, 1)
                
                overlaid_frames.append(overlaid)

            # Convert lists to numpy arrays
            original_frames_arr = np.stack(original_frames, axis=0)  # Shape: (30, H0, W0)
            overlaid_frames_arr = np.stack(overlaid_frames, axis=0)  # Shape: (30, H0, W0, 3)

            # Save as numpy arrays
            original_path = os.path.join(viz_folder, "original_frames.npy")
            overlaid_path = os.path.join(viz_folder, "overlaid_frames.npy")
            
            np.save(original_path, original_frames_arr)
            np.save(overlaid_path, overlaid_frames_arr)

            print(f"[SparseAttentionViz] Saved original frames to {original_path} with shape {original_frames_arr.shape}")
            print(f"[SparseAttentionViz] Saved overlaid frames to {overlaid_path} with shape {overlaid_frames_arr.shape}")

        return
    
import torch
import torch.nn.functional as F

def temporal_moving_average(x, kernel_size=3):
    """
    x: (B, T, P, Demb)
    kernel_size: number of timesteps to average over (must be odd for symmetric)
    """
    assert kernel_size % 2 == 1, "Kernel size should be odd for symmetric smoothing"
    pad = kernel_size // 2

    # reshape to (B*P*Demb, 1, T) for conv1d
    B, T, P, D = x.shape
    x_reshape = x.permute(0, 2, 3, 1).reshape(B*P*D, 1, T)  # (B*P*D, 1, T)

    # create averaging kernel
    kernel = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size

    # apply conv1d with padding='same'
    x_smooth = F.conv1d(x_reshape, kernel, padding=pad)

    # reshape back
    x_smooth = x_smooth.view(B, P, D, T).permute(0, 3, 1, 2)  # (B, T, P, Demb)
    return x_smooth
