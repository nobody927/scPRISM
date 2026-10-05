import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import math


# ==========================================
# 1. Basic components
# ==========================================
class CellLayerNorm(nn.Module):
    """LayerNorm over channel dim for [B, C, L] tensors (matching scDiffusion-X behavior)."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B, C, L] → transpose → LayerNorm(C) → transpose back
        return self.norm(x.transpose(-1, -2)).transpose(-1, -2)


def normalization_cell(channels):
    return CellLayerNorm(channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def checkpoint(func, inputs, params, flag):
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)
    return func(*inputs)


# ==========================================
# 2. Basic modules
# ==========================================
class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb): pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class CellConv(nn.Module):
    """1D Conv wrapper. For initial/output blocks, use kernel_size=1 (= Linear per position)."""
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.out_channels = out_channels

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.stride = 2
        if use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 2, padding='same')

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.stride, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv1d(self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


# ==========================================
# 3. Attention
# ==========================================
class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(ch)
        q = q.reshape(bs * self.n_heads, ch, length) * scale
        k = k.reshape(bs * self.n_heads, ch, length)
        v = v.reshape(bs * self.n_heads, ch, length)
        weight = torch.einsum("bct,bcs->bts", q, k)
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization_cell(channels)
        self.qkv = CellConv(channels, channels * 3, kernel_size=1, padding=0)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(CellConv(channels, channels, kernel_size=1, padding=0))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, l = x.shape
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return x + h


# ==========================================
# 4. ResBlock (timestep conditioning)
# ==========================================
class ResBlock_cell(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_checkpoint=False, use_scale_shift_norm=True,
                 up=False, down=False, use_conv=False, num_heads=4):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization_cell(channels),
            nn.SiLU(),
            CellConv(channels, self.out_channels, kernel_size=3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, True)
            self.x_upd = Upsample(channels, True)
        elif down:
            self.h_upd = Downsample(channels, True)
            self.x_upd = Downsample(channels, True)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )

        self.out_layers = nn.Sequential(
            normalization_cell(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            CellConv(self.out_channels, self.out_channels, kernel_size=1, padding=0),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = CellConv(channels, self.out_channels, kernel_size=3, padding=1)
        else:
            self.skip_connection = CellConv(channels, self.out_channels, kernel_size=1, padding=0)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            h = self.in_layers(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


# ==========================================
# 5. Composite Block: ResBlock → Attention → ResBlock
# ==========================================
class ResAttnBlock(TimestepBlock):
    """
    Each block follows: ResBlock → Attention → ResBlock
    """
    def __init__(self, channels, emb_channels, dropout, out_channels=None,
                 use_checkpoint=False, use_scale_shift_norm=False,
                 up=False, down=False, num_heads=4):
        super().__init__()
        self.res1 = ResBlock_cell(channels, emb_channels, dropout,
                                  out_channels=out_channels,
                                  use_checkpoint=use_checkpoint,
                                  use_scale_shift_norm=use_scale_shift_norm,
                                  up=up, down=down)
        ch = out_channels or channels
        self.attn = AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint)
        self.res2 = ResBlock_cell(ch, emb_channels, dropout,
                                  use_checkpoint=use_checkpoint,
                                  use_scale_shift_norm=use_scale_shift_norm)

    def forward(self, x, emb):
        x = self.res1(x, emb)
        x = self.attn(x)
        x = self.res2(x, emb)
        return x


# ==========================================
# 6. Main model: MicDiffUNet (U-Net architecture)
# ==========================================
class MicDiffUNet(nn.Module):
    """
    1D U-Net for diffusion in 60D latent space.
    Architecture: Input blocks (encoder) → Middle block → Output blocks (decoder)
    Each block: ResBlock → Attention → ResBlock
    Conditioning: timestep + perturbation + cellline (optional)
    """

    def __init__(
            self,
            rna_dim,                    # latent dimension, e.g. [1, 60] or [2, 60]
            model_channels,             # base channel count, e.g. 64
            out_channels,               # output channels (LEARNED_RANGE: in_ch * 2)
            num_res_blocks=2,           # blocks per level (default 2, matching scDiffusion-X)
            dropout=0.1,
            channel_mult=(1, 2, 4),     # channel multiplier per level
            num_classes=0,              # cell line classes (0 = disabled)
            num_perturbations=0,        # drug perturbation classes (0 = disabled, gene task uses pert_embed_dim)
            use_checkpoint=False,
            use_fp16=False,
            num_heads=4,
            use_scale_shift_norm=True,
            resblock_updown=True,
            pert_embed_dim=0,           # gene2vec external embedding dim (0 = disabled)
            concat_ctrl=False,          # whether to concatenate control data
    ):
        super().__init__()

        self.rna_dim = rna_dim
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.concat_ctrl = concat_ctrl

        # ---- Conditioning embedding dimensions ----
        time_embed_dim = model_channels * 4

        # 1. Timestep embedding
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 2. Cell line embedding (optional)
        self.num_classes = int(num_classes) if num_classes and num_classes > 0 else 0
        if self.num_classes > 0:
            self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)

        # 3. Drug perturbation embedding (optional, nn.Embedding)
        self.num_perturbations = int(num_perturbations) if num_perturbations and num_perturbations > 0 else 0
        if self.num_perturbations > 0:
            self.pert_emb = nn.Embedding(self.num_perturbations, time_embed_dim)

        # 4.  Gene perturbation projection (optional, gene2vec continuous vector)
        self.pert_embed_dim = pert_embed_dim
        if self.pert_embed_dim > 0:
            self.pert_proj = nn.Sequential(
                linear(self.pert_embed_dim, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )

        # ---- Input channels ----
        input_channels = 2 if concat_ctrl else 1

        # ---- Attention layers: only at level == len(channel_mult)-2 and middle ----
        attn_level = len(channel_mult) - 2  # e.g., channel_mult=[1,2,4] → level 1

        # ---- Build U-Net (scDiffusion-X style) ----
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                CellConv(input_channels, ch)  # Linear projection
            )
        ])
        input_block_chans = [ch]

        # ---- Encoder ----
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock_cell(
                    ch, time_embed_dim, dropout,
                    out_channels=int(mult * model_channels),
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )]
                ch = int(mult * model_channels)
                # Attention only at specified layers
                if level == attn_level:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            # Downsample (last level not downsampled)
            if level != len(channel_mult) - 1:
                out_ch = ch
                if resblock_updown:
                    down_layer = ResBlock_cell(
                        ch, time_embed_dim, dropout, out_channels=out_ch,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                    )
                else:
                    down_layer = Downsample(ch, True, out_channels=out_ch)
                self.input_blocks.append(TimestepEmbedSequential(down_layer))
                ch = out_ch
                input_block_chans.append(ch)

        # ---- Middle: Attention + ResBlock ----
        self.middle_blocks = TimestepEmbedSequential(
            AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
            ResBlock_cell(ch, time_embed_dim, dropout,
                          use_checkpoint=use_checkpoint,
                          use_scale_shift_norm=use_scale_shift_norm),
        )

        # ---- Decoder ----
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock_cell(
                    ch + ich, time_embed_dim, dropout,
                    out_channels=int(model_channels * mult),
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )]
                ch = int(model_channels * mult)
                # Attention only at specified layers
                if level == attn_level:
                    layers.append(AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint))

                # Upsample (no upsample after last block of last level)
                if level > 0 and i == num_res_blocks:
                    out_ch = ch
                    if resblock_updown:
                        layers.append(ResBlock_cell(
                            ch, time_embed_dim, dropout, out_channels=out_ch,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        ))
                    else:
                        layers.append(Upsample(ch, True, out_channels=out_ch))

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # ---- Output layer ----
        self.out = nn.Sequential(
            normalization_cell(ch),
            nn.SiLU(),
            zero_module(CellConv(ch, out_channels)),
        )

    def forward(self, x, timesteps, cond_vec=None, pert_id=None, label=None, x_ctrl=None, **kwargs):
        """
        x:         [B, 1, L] noise input
        timesteps: [B] diffusion step
        cond_vec:  [B, D]  gene perturbation continuous vector (gene2vec), optional
        pert_id:   [B] drug perturbation integer ID, optional
        label:     [B] cell line integer ID, optional
        x_ctrl:    [B, 1, L] control input (required when concat_ctrl=True)
        """
        # Unify dimensions
        input_is_2d = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            input_is_2d = True

        # Concatenate control
        if self.concat_ctrl:
            if x_ctrl is None:
                raise ValueError("concat_ctrl=True but x_ctrl not provided")
            if x_ctrl.dim() == 2:
                x_ctrl = x_ctrl.unsqueeze(1)
            x = torch.cat([x, x_ctrl.type(x.dtype)], dim=1)

        # ---- Compute condition embeddings ----
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # Cell line
        if self.num_classes > 0 and label is not None:
            emb = emb + self.label_emb(label)

        # Drug perturbation (integer ID → nn.Embedding)
        if self.num_perturbations > 0 and pert_id is not None:
            emb = emb + self.pert_emb(pert_id)

        #  Gene perturbation (continuous vector → linear projection)
        if self.pert_embed_dim > 0 and cond_vec is not None:
            emb = emb + self.pert_proj(cond_vec.type(emb.dtype))

        # Type conversion
        x = x.type(self.dtype)
        emb = emb.type(self.dtype)

        # ---- Encoder ----
        hs = []
        for module in self.input_blocks:
            x = module(x, emb)
            hs.append(x)

        # ---- Middle ----
        x = self.middle_blocks(x, emb)

        # ---- Decoder (skip connections) ----
        for module in self.output_blocks:
            h = hs.pop()
            x = torch.cat([x, h], dim=1)
            x = module(x, emb)

        # ---- Output ----
        x = self.out(x)

        if input_is_2d:
            x = x.squeeze(1)

        return x
