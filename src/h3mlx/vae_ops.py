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
    idx = list(range(before, 0, -1)) + list(range(n)) + list(range(n - 2, n - 2 - after, -1))
    return mx.take(x, mx.array(idx), axis=axis)


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
    return x * mx.sigmoid(x)


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
