"""MiniMax H3 video VAE encoder ops in MLX (diffusers 0.42.1 `autoencoder_kl_minimax_h3.py`).

MLX convolutions are channels-last, so these ops take and return `(B, F, H, W, C)`; torch's layout is
`(B, C, F, H, W)`. Conv weights are converted once with `conv3d_weight` (torch `(O, I, kt, kh, kw)` ->
MLX `(O, kt, kh, kw, I)`).
"""

import mlx.core as mx


def conv3d_weight(w_torch_layout: mx.array) -> mx.array:
    return w_torch_layout.transpose(0, 2, 3, 4, 1)


def reflect_pad(x: mx.array, axis: int, before: int, after: int) -> mx.array:
    """torch F.pad(mode="reflect") along one axis (edge sample not repeated)."""
    n = x.shape[axis]
    if before >= n or after >= n:
        raise ValueError(f"reflect pad ({before}, {after}) needs size > pad, got {n}")
    widths = [(0, 0)] * x.ndim
    widths[axis] = (before, after)
    return mx.pad(x, widths, mode="reflect")


def causal_conv3d(
    x: mx.array,
    w: mx.array,
    b: mx.array | None,
    stride: tuple[int, int, int] = (1, 1, 1),
    spatial_padding: int = 0,
    temporal_padding: int = 0,
) -> mx.array:
    """MiniMaxH3VideoCausalConv3d: symmetric reflect pad on H, W; `temporal_padding` zero frames prepended only.
    x: (B, F, H, W, C); w: MLX layout (O, kt, kh, kw, I)."""
    if spatial_padding > 0:
        x = reflect_pad(x, 2, spatial_padding, spatial_padding)
        x = reflect_pad(x, 3, spatial_padding, spatial_padding)
    if temporal_padding > 0:
        x = mx.pad(x, [(0, 0), (temporal_padding, 0), (0, 0), (0, 0), (0, 0)])
    y = mx.conv3d(x, w, stride=stride, padding=0)
    return y + b if b is not None else y


def group_norm_per_frame(x: mx.array, groups: int, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
    """MiniMaxH3VideoGroupNorm: GroupNorm with statistics per (batch, frame), never across frames. Biased variance,
    computed in float32."""
    b, f, h, w, c = x.shape
    xf = x.astype(mx.float32).reshape(b, f, h * w, groups, c // groups)
    mean = xf.mean(axis=(2, 4), keepdims=True)
    var = ((xf - mean) ** 2).mean(axis=(2, 4), keepdims=True)
    y = ((xf - mean) * mx.rsqrt(var + eps)).reshape(b, f, h, w, c).astype(x.dtype)
    return y * weight + bias


def silu(x: mx.array) -> mx.array:
    """Single rounding like torch: float32 opmath, cast back."""
    xf = x.astype(mx.float32)
    return (xf * mx.sigmoid(xf)).astype(x.dtype)


def resnet_block3d(x: mx.array, p: dict, groups: int, eps: float = 1e-6) -> mx.array:
    """MiniMaxH3VideoResnetBlock3d. `p` holds MLX-layout conv weights (see `conv3d_weight`)."""
    h = silu(group_norm_per_frame(x, groups, p["norm1.weight"], p["norm1.bias"], eps))
    h = causal_conv3d(h, p["conv1.weight"], p["conv1.bias"], spatial_padding=1, temporal_padding=2)
    h = silu(group_norm_per_frame(h, groups, p["norm2.weight"], p["norm2.bias"], eps))
    h = causal_conv3d(h, p["conv2.weight"], p["conv2.bias"], spatial_padding=1, temporal_padding=2)
    if "conv_shortcut.weight" in p:
        x = causal_conv3d(x, p["conv_shortcut.weight"], p["conv_shortcut.bias"])
    return x + h


def downsample3d(x: mx.array, w: mx.array, b: mx.array, temporal_stride: int = 1, spatial_stride: int = 2) -> mx.array:
    """MiniMaxH3VideoDownsample3d: reflect pad bottom/right by 1 when the spatial stride is 2, then a 3x3x3 causal conv
    (2 zero frames prepended, no spatial padding) with stride (t, s, s)."""
    if spatial_stride == 2:
        x = reflect_pad(x, 2, 0, 1)
        x = reflect_pad(x, 3, 0, 1)
    return causal_conv3d(x, w, b, stride=(temporal_stride, spatial_stride, spatial_stride), temporal_padding=2)


def _sub(p: dict, prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}


def encoder3d(
    x: mx.array,
    p: dict,
    block_out_channels: tuple[int, ...],
    layers_per_block: int,
    spatial_downsample_factors: tuple[int, ...],
    temporal_downsample_factors: tuple[int, ...],
    groups: int,
    eps: float = 1e-6,
) -> mx.array:
    """MiniMaxH3VideoEncoder3d on channels-last (B, F, H, W, 3) pixels -> (B, F', H', W', out_channels)."""
    h = causal_conv3d(x, p["conv_in.weight"], p["conv_in.bias"], spatial_padding=1, temporal_padding=2)
    for i in range(len(block_out_channels)):
        bp = _sub(p, f"down_blocks.{i}.")
        for j in range(layers_per_block):
            h = resnet_block3d(h, _sub(bp, f"resnets.{j}."), groups, eps)
        if temporal_downsample_factors[i] * spatial_downsample_factors[i] > 1:
            h = downsample3d(
                h, bp["downsamplers.0.conv.weight"], bp["downsamplers.0.conv.bias"],
                temporal_downsample_factors[i], spatial_downsample_factors[i],
            )
    h = silu(group_norm_per_frame(h, groups, p["norm_out.weight"], p["norm_out.bias"], eps))
    return causal_conv3d(h, p["conv_out.weight"], p["conv_out.bias"], spatial_padding=1, temporal_padding=2)


# --- ViT decoder (MiniMaxH3VideoViTDecoder3d) -----------------------------------------------------------------------


def vit_rope_inv_freq(dim: int, theta: float = 100.0, num_axes: int = 3) -> mx.array:
    """1 / theta ** arange(0, 1, 2 * num_axes / dim), float32 (numpy, like the DiT's inv_freq)."""
    import numpy as np

    exps = np.arange(0, 1, 2 * num_axes / dim, dtype=np.float32)
    return mx.array((np.float32(1) / np.power(np.float32(theta), exps)).astype(np.float32))


def vit_positions(frames: int, height: int, width: int, extra: int, frame_position: float | None = None) -> mx.array:
    """Voxel centres normalised to [-1, 1) per axis, (t, h, w) in raster order, then `extra` zero rows. With
    `frame_position` (one frame only) the time axis is that fixed coordinate instead (P-H3-Image's decoder)."""
    import numpy as np

    axes = [2.0 * (np.arange(0.5, n, dtype=np.float32) / np.float32(n)) - 1.0 for n in (frames, height, width)]
    if frame_position is not None:
        if frames != 1:
            raise ValueError(f"a fixed frame position decodes one latent frame, got {frames}")
        axes[0] = np.array([frame_position], dtype=np.float32)
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    return mx.array(np.concatenate([grid, np.zeros((extra, 3), dtype=np.float32)]))


def vit_rope(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
    """(n, 3) -> cos, sin of shape (n, 2 * 3 * len(inv_freq)): 2*pi*pos*inv_freq, axes concatenated, then tiled x2."""
    import math

    ang = 2.0 * math.pi * position_ids[:, :, None] * inv_freq[None, None, :]
    ang = ang.reshape(position_ids.shape[0], -1)
    ang = mx.concatenate([ang, ang], axis=-1)
    return mx.cos(ang), mx.sin(ang)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    c, s = cos.astype(x.dtype)[None, :, None, :], sin.astype(x.dtype)[None, :, None, :]
    x1, x2 = xr[..., : rot // 2], xr[..., rot // 2 :]
    return mx.concatenate([xr * c + mx.concatenate([-x2, x1], axis=-1) * s, xp], axis=-1)


def _lin(x, p, name):
    y = x @ p[f"{name}.weight"].T
    return y + p[f"{name}.bias"] if f"{name}.bias" in p else y


def vit_attention(x: mx.array, p: dict, heads: int, eps: float, rotary) -> mx.array:
    """MiniMaxH3VideoAttnProcessor: biased q/k/v, affine-free RMSNorm on q/k in float32, rope, sdpa, to_out."""
    import math

    b, n, _ = x.shape
    q = _lin(x, p, "to_q").reshape(b, n, heads, -1)
    k = _lin(x, p, "to_k").reshape(b, n, heads, -1)
    v = _lin(x, p, "to_v").reshape(b, n, heads, -1)
    ones = mx.ones((q.shape[-1],), dtype=mx.float32)
    q = mx.fast.rms_norm(q.astype(mx.float32), ones, eps).astype(q.dtype)
    k = mx.fast.rms_norm(k.astype(mx.float32), ones, eps).astype(k.dtype)
    if rotary is not None:
        q, k = _rotate(q, *rotary), _rotate(k, *rotary)
    o = mx.fast.scaled_dot_product_attention(
        q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3), scale=1.0 / math.sqrt(q.shape[-1])
    ).transpose(0, 2, 1, 3)
    return _lin(o.reshape(b, n, -1), p, "to_out.0")


def vit_block(x: mx.array, p: dict, heads: int, eps: float, rotary) -> mx.array:
    """Norms in float32, residual stream float32, attention/FF at the projections' dtype, layer scales."""
    cdt = p["attn.to_q.weight"].dtype
    h = mx.fast.rms_norm(x.astype(mx.float32), p["norm1.weight"].astype(mx.float32), eps).astype(cdt)
    x = x + vit_attention(h, _sub(p, "attn."), heads, eps, rotary) * p["scale1"]
    h = mx.fast.rms_norm(x.astype(mx.float32), p["norm2.weight"].astype(mx.float32), eps).astype(cdt)
    f = _lin(h, p, "ff.net.0.proj")
    f, gate = mx.split(f, 2, axis=-1)
    return x + _lin(f * silu(gate), p, "ff.net.2") * p["scale2"]


def vit_decoder(
    z: mx.array, p: dict, num_layers: int, heads: int, patch_size: int, patch_size_t: int, out_channels: int,
    rope_dim: int, rope_theta: float = 100.0, eps: float = 1e-5, frame_position: float | None = None,
) -> mx.array:
    """MiniMaxH3VideoViTDecoder3d. z: channels-last latents (B, F, H, W, C) -> pixels (B, F*pt, H*p, W*p, out).
    P-H3-Image's `PH3ImageViTDecoder` is the same with `patch_size_t=1` and one frame at `frame_position`."""
    b, f, h, w, c = z.shape
    x = _lin(z.reshape(b, f * h * w, c), p, "proj_in")
    n = x.shape[1]
    regs = mx.broadcast_to(p["register_tokens"].astype(x.dtype), (b, *p["register_tokens"].shape[1:]))
    x = mx.concatenate([x, regs, mx.zeros((b, 1, x.shape[-1]), dtype=x.dtype)], axis=1)
    rotary = vit_rope(vit_positions(f, h, w, regs.shape[1] + 1, frame_position), vit_rope_inv_freq(rope_dim, rope_theta))
    for i in range(num_layers):
        x = vit_block(x, _sub(p, f"transformer_blocks.{i}."), heads, eps, rotary)
        mx.eval(x)  # 36 f32 blocks per tile: keep each one its own GPU graph
    x = mx.fast.layer_norm(x.astype(mx.float32), p["norm_out.weight"].astype(mx.float32),
                           p["norm_out.bias"].astype(mx.float32), eps)
    x = _lin(x.astype(p["proj_out.weight"].dtype), p, "proj_out")[:, :n]
    x = x.reshape(b, f, h, w, out_channels, patch_size_t, patch_size, patch_size)
    x = x.transpose(0, 1, 5, 2, 6, 3, 7, 4)  # (b, f, pt, h, p, w, p, c)
    return x.reshape(b, f * patch_size_t, h * patch_size, w * patch_size, out_channels)
