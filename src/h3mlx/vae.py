"""The MiniMax H3 video VAE wrapper in MLX (diffusers 0.42.1 `AutoencoderKLMiniMaxH3`): spatial tiling, 17-frame
temporal chunking with `token_drop`, chunked decode with cross-fade, and the conditioning recipe of
`encoders.encode_vae_condition`.

Layout is channels-last throughout: pixels and latents are (B, F, H, W, C).
"""

import math
from dataclasses import dataclass

import mlx.core as mx

from . import vae_ops as V


@dataclass
class VaeConfig:
    latent_channels: int = 24
    block_out_channels: tuple = (128, 256, 256, 512, 512, 1024)
    layers_per_block: int = 2
    spatial_downsample_factors: tuple = (2, 2, 2, 2, 1, 1)
    temporal_downsample_factors: tuple = (1, 2, 2, 1, 1, 1)
    norm_num_groups: int = 32
    norm_eps: float = 1e-6
    decoder_num_layers: int = 36
    decoder_num_attention_heads: int = 32
    decoder_attention_head_dim: int = 64
    decoder_rope_theta: float = 100.0
    decoder_rope_dim_ratio: float = 0.75
    decoder_norm_eps: float = 1e-5
    clip_length: int = 17
    token_drop: int = 3
    tile_h: int = 256
    tile_w: int = 256
    tile_overlap_h: int = 64
    tile_overlap_w: int = 64
    use_tiling: bool = True

    @classmethod
    def from_diffusers(cls, cfg: dict, **kw):
        names = cls.__dataclass_fields__
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in names}, **kw)

    @property
    def spatial_ratio(self):
        return math.prod(self.spatial_downsample_factors)

    @property
    def temporal_ratio(self):
        return math.prod(self.temporal_downsample_factors)

    @property
    def frame_pre_padding(self):
        return (-self.clip_length) % self.temporal_ratio

    @property
    def tokens_chunk_size(self):
        return math.ceil(self.clip_length / self.temporal_ratio)

    @property
    def token_overlap(self):
        return (-self.token_drop) % self.tokens_chunk_size

    @property
    def frame_overlap(self):
        return max(self.token_overlap * self.temporal_ratio - self.frame_pre_padding, 0)


def split_tiles(length: int, tile: int, min_overlap: int, ratio: int):
    if tile >= length:
        return [0], [length], []
    n = math.ceil(length / tile)
    while tile * n - min_overlap * (n - 1) - length < 0:
        n += 1
    overlaps = [min_overlap] * (n - 1)
    remaining = tile * n - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (n - 1)] += ratio
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts, [tile] * n, overlaps


def _slice(x, axis, start, stop):
    idx = [slice(None)] * x.ndim
    idx[axis] = slice(start, stop)
    return x[tuple(idx)]


def blend(a: mx.array, b: mx.array, extent: int, axis: int) -> mx.array:
    """Linear cross-fade of a's tail into b's head over `extent` along `axis`; ramp built in b's dtype."""
    extent = min(a.shape[axis], b.shape[axis], extent)
    pos = mx.arange(extent).astype(b.dtype)
    shape = [1] * a.ndim
    shape[axis] = extent
    wa = (1 - pos / extent).reshape(shape)
    wb = (pos / extent).reshape(shape)
    out = _slice(a, axis, -extent, None) * wa + _slice(b, axis, 0, extent) * wb
    if extent == b.shape[axis]:
        return out
    return mx.concatenate([out, _slice(b, axis, extent, None)], axis=axis)


def stitch_tiles(tiles, h_overlaps, w_overlaps, h_axis=2, w_axis=3):
    rows = []
    for i, row in enumerate(tiles):
        out_row = []
        for j, t in enumerate(row):
            if i > 0:
                t = blend(tiles[i - 1][j], t, h_overlaps[i - 1], h_axis)
            if j > 0:
                t = blend(row[j - 1], t, w_overlaps[j - 1], w_axis)
            if i < len(tiles) - 1:
                t = _slice(t, h_axis, 0, -h_overlaps[i])
            if j < len(row) - 1:
                t = _slice(t, w_axis, 0, -w_overlaps[j])
            out_row.append(t)
        rows.append(mx.concatenate(out_row, axis=w_axis))
    return mx.concatenate(rows, axis=h_axis)


def _sub(p, prefix):
    return {k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}


class VideoVAE:
    """Holds MLX-layout weights (conv weights already transposed with `vae_ops.conv3d_weight`)."""

    def __init__(self, params: dict, cfg: VaeConfig):
        self.p, self.cfg = params, cfg
        self.enc = _sub(params, "encoder.")
        self.dec = _sub(params, "decoder.")

    # -- encode ------------------------------------------------------------------------------------------------------

    def _encoder(self, x):
        c = self.cfg
        h = V.encoder3d(x, self.enc, c.block_out_channels, c.layers_per_block, c.spatial_downsample_factors,
                        c.temporal_downsample_factors, c.norm_num_groups, c.norm_eps)
        return V.causal_conv3d(h, self.p["quant_conv.weight"], self.p["quant_conv.bias"])

    def encode_clip(self, x):
        c = self.cfg
        if not c.use_tiling:
            return self._encoder(x)
        ys, yl, yo = split_tiles(x.shape[2], c.tile_h, c.tile_overlap_h, c.spatial_ratio)
        xs, xl, xo = split_tiles(x.shape[3], c.tile_w, c.tile_overlap_w, c.spatial_ratio)
        rows = [[self._encoder(x[:, :, i : i + il, j : j + jl]) for j, jl in zip(xs, xl)] for i, il in zip(ys, yl)]
        r = c.spatial_ratio
        return stitch_tiles(rows, [o // r for o in yo], [o // r for o in xo])

    def encode_moments(self, x):
        """Pixels (B, F, H, W, 3) -> moments (B, F', H', W', 2 * latent). One frame: spatial encoder alone."""
        c = self.cfg
        n = x.shape[1]
        if n == 1:
            return self.encode_clip(x)
        if n % c.clip_length:
            x = mx.concatenate([x, mx.repeat(x[:, -1:], (-n) % c.clip_length, axis=1)], axis=1)
        m = mx.concatenate(
            [self.encode_clip(x[:, i * c.clip_length : (i + 1) * c.clip_length]) for i in range(x.shape[1] // c.clip_length)],
            axis=1,
        )
        return m[:, : -c.token_drop] if c.token_drop > 0 else m

    # -- decode ------------------------------------------------------------------------------------------------------

    def _decoder(self, z):
        c = self.cfg
        z = V.causal_conv3d(z, self.p["post_quant_conv.weight"], self.p["post_quant_conv.bias"])
        return V.vit_decoder(z, self.dec, c.decoder_num_layers, c.decoder_num_attention_heads, c.spatial_ratio,
                             c.temporal_ratio, 3, int(c.decoder_attention_head_dim * c.decoder_rope_dim_ratio),
                             c.decoder_rope_theta, c.decoder_norm_eps)

    def decode_clip(self, z):
        c = self.cfg
        if not c.use_tiling:
            return self._decoder(z)
        r = c.spatial_ratio
        ys, yl, yo = split_tiles(z.shape[2] * r, c.tile_h, c.tile_overlap_h, r)
        xs, xl, xo = split_tiles(z.shape[3] * r, c.tile_w, c.tile_overlap_w, r)
        rows = [[self._decoder(z[:, :, i // r : i // r + il // r, j // r : j // r + jl // r]) for j, jl in zip(xs, xl)]
                for i, il in zip(ys, yl)]
        return stitch_tiles(rows, yo, xo)

    def decode(self, z):
        """Latents (B, F, H, W, latent) -> pixels (B, frames, H*r, W*r, 3), chunked and cross-faded like `_decode`."""
        c = self.cfg
        tcs, drop, tr = c.tokens_chunk_size, c.token_drop, c.temporal_ratio
        chunk_frames = tcs * tr
        num_tokens = z.shape[1] + drop
        pad = (-num_tokens) % tcs
        n_chunks = (num_tokens + pad) // tcs - int(drop > 0)
        if pad:
            z = mx.concatenate([z, mx.repeat(z[:, -1:], pad, axis=1)], axis=1)
        out, overlap = [], None
        for i in range(n_chunks):
            clip = self.decode_clip(z[:, i * tcs : i * tcs + tcs + c.token_overlap])
            for j in range(int(drop > 0) + 1):
                ch = clip[:, j * chunk_frames : j * chunk_frames + chunk_frames][:, c.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        ch = blend(overlap, ch, c.frame_overlap, axis=1)
                    out.append(ch)
                else:
                    overlap = ch
        if overlap is not None:
            out.append(overlap)
        dec = mx.concatenate(out, axis=1)
        if pad:
            tail = c.clip_length % tr
            before = z.shape[1] - pad
            pad_frames = sum(tail if tail and (before + k) % tcs == 0 else tr for k in range(pad))
            dec = dec[:, :-pad_frames]
        return dec


def posterior_sample(moments: mx.array, noise: mx.array) -> mx.array:
    """DiagonalGaussianDistribution.sample: mean + exp(0.5 * clip(logvar, -30, 20)) * noise; moments split on C."""
    mean, logvar = mx.split(moments, 2, axis=-1)
    return mean + mx.exp(0.5 * mx.clip(logvar, -30.0, 20.0)) * noise


def encode_condition(vae: VideoVAE, pixels_u8: mx.array, pixel_mean, pixel_std, latents_mean, latents_std, noise_fn):
    """`encode_vae_condition`: uint8 pixels (1, F, H, W, 3) -> ImageNet-normalised -> moments -> posterior sample
    (noise from `noise_fn(shape)`, which must reproduce torch.Generator().manual_seed(42) to match the release) ->
    float16 round trip -> (latent - mean) / std."""
    x = (pixels_u8.astype(mx.float32) / 255.0 - mx.array(pixel_mean)) / mx.array(pixel_std)
    m = vae.encode_moments(x)
    lat = posterior_sample(m, noise_fn(m.shape[:-1] + (m.shape[-1] // 2,)))
    lat = lat.astype(mx.float16).astype(mx.float32)
    return (lat - mx.array(latents_mean)) / mx.array(latents_std)
