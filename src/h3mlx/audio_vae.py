"""MiniMax H3 audio VAE in MLX (diffusers 0.42.1 `autoencoder_kl_minimax_h3_audio.py`): DAC encoder, causal-attention
`pre_block`, mean head (the pipeline takes the posterior mode), BigVGAN decoder with alias-free SnakeBeta. Float32
throughout (bf16 decodes ~20 dB quieter per diffusers). Channels-last: (B, L, C). Mono 32 kHz; stereo = batch of 2.

Weights: pass the diffusers state dict through `fold_weight_norm` (weight_g * weight_v / ||weight_v||, dim 0) and
`to_mlx_layout` (Conv1d (O, I, K) -> (O, K, I); ConvTranspose1d (I, O, K) -> (O, K, I)).
"""

import math

import mlx.core as mx
import numpy as np

# --- weights ------------------------------------------------------------------------------------------------------


def fold_weight_norm(sd: dict) -> dict:
    """torch.nn.utils.weight_norm (dim=0): weight = g * v / ||v|| with the norm over every axis but 0."""
    out = {}
    for k, v in sd.items():
        if k.endswith(".weight_g"):
            continue
        if k.endswith(".weight_v"):
            base = k[: -len("_v")]
            g = np.asarray(sd[base + "_g"], dtype=np.float32)
            vv = np.asarray(v, dtype=np.float32)
            norm = np.sqrt((vv.astype(np.float64) ** 2).sum(axis=tuple(range(1, vv.ndim)), keepdims=True))
            out[base] = (g * vv / norm).astype(np.float32)
        else:
            out[k] = np.asarray(v)
    return out


def to_mlx_layout(sd: dict, transpose_conv_prefixes=("decoder.ups.",)) -> dict:
    """numpy torch-layout weights -> MLX arrays in MLX conv layout. 3-D `weight`s are convs."""
    out = {}
    for k, v in sd.items():
        a = np.asarray(v, dtype=np.float32) if v.dtype != np.float32 else np.asarray(v)
        if k.endswith("weight") and a.ndim == 3 and not k.endswith("filter"):
            a = a.transpose(1, 2, 0) if k.startswith(transpose_conv_prefixes) else a.transpose(0, 2, 1)
        out[k] = mx.array(a)
    return out


# --- activations ----------------------------------------------------------------------------------------------------


def snake(x, alpha):
    """Snake1d: alpha (1, C, 1) torch layout -> broadcast over (B, L, C)."""
    a = alpha.reshape(1, 1, -1)
    return x + (1.0 / (a + 1e-9)) * mx.sin(a * x) ** 2


def snake_beta(x, alpha, beta):
    a = mx.exp(alpha).reshape(1, 1, -1)
    b = mx.exp(beta).reshape(1, 1, -1)
    return x + (1.0 / (b + 1e-9)) * mx.sin(a * x) ** 2


def _depthwise(filt: mx.array, channels: int) -> mx.array:
    k = filt.size
    return mx.broadcast_to(filt.reshape(1, k, 1), (channels, k, 1))


def lowpass(x, filt: mx.array, stride: int):
    k = filt.size
    even = k % 2 == 0
    x = mx.pad(x, [(0, 0), (k // 2 - int(even), k // 2), (0, 0)], mode="edge")
    c = x.shape[-1]
    return mx.conv1d(x, _depthwise(filt, c), stride=stride, groups=c)


def upsample(x, filt: mx.array, ratio: int):
    """ratio * conv_transpose1d(replicate-pad(x, pad), filter, stride=ratio, groups=C), then crop."""
    k = filt.size
    pad = k // ratio - 1
    pad_left = pad * ratio + (k - ratio) // 2
    pad_right = pad * ratio + (k - ratio + 1) // 2
    x = mx.pad(x, [(0, 0), (pad, pad), (0, 0)], mode="edge")
    b, n, c = x.shape
    # transposed depthwise conv as zero-insertion + conv with the flipped kernel
    up = mx.zeros((b, (n - 1) * ratio + 1, c), dtype=x.dtype)
    up[:, ::ratio] = x
    up = mx.pad(up, [(0, 0), (k - 1, k - 1), (0, 0)])
    y = ratio * mx.conv1d(up, _depthwise(filt.reshape(-1)[::-1], c), groups=c)
    return y[:, pad_left : y.shape[1] - pad_right]


def activation1d(x, p, ratio: int = 2):
    """Alias-free SnakeBeta: upsample x2, activate, low-pass downsample x2. Filters are the checkpoint's buffers
    (`upsample.filter`, `downsample.lowpass.filter`), Kaiser-sinc, kernel 12."""
    x = upsample(x, p["upsample.filter"], ratio)
    x = snake_beta(x, p["act.alpha"], p["act.beta"])
    return lowpass(x, p["downsample.lowpass.filter"], ratio)


def conv1d(x, p, name, padding=0, dilation=1, stride=1):
    y = mx.conv1d(x, p[f"{name}.weight"], stride=stride, padding=padding, dilation=dilation)
    return y + p[f"{name}.bias"] if f"{name}.bias" in p else y


def _sub(p, prefix):
    return {k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}


# --- encoder --------------------------------------------------------------------------------------------------------


def residual_unit(x, p, dilation):
    h = snake(x, p["block.0.alpha"])
    h = conv1d(h, p, "block.1", padding=((7 - 1) * dilation) // 2, dilation=dilation)
    h = snake(h, p["block.2.alpha"])
    h = conv1d(h, p, "block.3")
    pad = (x.shape[1] - h.shape[1]) // 2
    if pad > 0:
        x = x[:, pad:-pad]
    return x + h


def encoder(x, p, strides):
    h = conv1d(x, p, "block.0", padding=3)
    for i, s in enumerate(strides):
        bp = _sub(p, f"block.{i + 1}.block.")
        for j, d in enumerate((1, 3, 9)):
            h = residual_unit(h, _sub(bp, f"{j}."), d)
        h = snake(h, bp["3.alpha"])
        h = conv1d(h, bp, "4", padding=math.ceil(s / 2), stride=s)
    n = len(strides) + 1
    h = snake(h, p[f"block.{n}.alpha"])
    return conv1d(h, p, f"block.{n + 1}", padding=1)


def layer_norm(x, p, name, eps=1e-5):
    return mx.fast.layer_norm(x, p[f"{name}.weight"], p[f"{name}.bias"], eps)


def lin(x, p, name):
    y = x @ p[f"{name}.weight"].T
    return y + p[f"{name}.bias"] if f"{name}.bias" in p else y


def adaptive_avg_pool_last(x, out_dim):
    d = x.shape[-1]
    cols = []
    for i in range(out_dim):
        s, e = (i * d) // out_dim, -((-(i + 1) * d) // out_dim)
        cols.append(x[..., s:e].mean(axis=-1, keepdims=True))
    return mx.concatenate(cols, axis=-1)


def gelu_tanh(x):
    return 0.5 * x * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


def causal_attention(x, p, heads, out_dim):
    b, n, c = x.shape
    bias = mx.concatenate([p["q_bias"], mx.zeros_like(p["q_bias"]), p["v_bias"]])
    qkv = (x @ p["qkv.weight"].T + bias).reshape(b, n, 3, heads, c // heads)
    q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(c // heads), mask="causal")
    o = o.transpose(0, 2, 1, 3).mean(axis=2)  # heads mean-pooled away
    return lin(adaptive_avg_pool_last(o, out_dim), p, "proj")


def pre_block(x, p, heads, out_dim):
    h = lin(layer_norm(x, p, "norm3"), p, "proj") + causal_attention(layer_norm(x, p, "norm1"), _sub(p, "attn."), heads, out_dim)
    m = _sub(p, "mlp.")
    y = layer_norm(h, p, "norm2")
    y = layer_norm(y, m, "norm")
    y = gelu_tanh(lin(y, m, "w0")) * lin(y, m, "w1")
    return h + lin(y, m, "w2")


# --- decoder --------------------------------------------------------------------------------------------------------


def amp_block(x, p, kernel, dilations):
    for i, d in enumerate(dilations):
        r = activation1d(x, _sub(p, f"activations.{2 * i}."))
        r = conv1d(r, p, f"convs1.{i}", padding=(kernel * d - d) // 2, dilation=d)
        r = activation1d(r, _sub(p, f"activations.{2 * i + 1}."))
        r = conv1d(r, p, f"convs2.{i}", padding=(kernel - 1) // 2)
        x = r + x
    return x


def bigvgan(x, p, rates, kernels, rb_kernels, rb_dilations):
    h = conv1d(x, p, "conv_pre", padding=3)
    nk = len(rb_kernels)
    for i, (rate, k) in enumerate(zip(rates, kernels)):
        h = mx.conv_transpose1d(h, p[f"ups.{i}.0.weight"], stride=rate, padding=(k - rate) // 2) + p[f"ups.{i}.0.bias"]
        acc = None
        for j in range(nk):
            blk = amp_block(h, _sub(p, f"resblocks.{i * nk + j}."), rb_kernels[j], rb_dilations[j])
            acc = blk if acc is None else acc + blk
        h = acc / nk
    h = activation1d(h, _sub(p, "activation_post."))
    return mx.clip(conv1d(h, p, "conv_post", padding=3), -1.0, 1.0)


class AudioVAE:
    def __init__(self, params: dict, cfg: dict):
        self.p, self.cfg = params, cfg
        self.hop = math.prod(cfg["encoder_rates"])

    def encode_mean(self, wave: mx.array) -> mx.array:
        """(B, samples, 1) mono -> posterior mean (B, samples / hop, latent_channels); right-pads to the hop."""
        pad = math.ceil(wave.shape[1] / self.hop) * self.hop - wave.shape[1]
        if pad:
            wave = mx.pad(wave, [(0, 0), (0, pad), (0, 0)])
        h = encoder(wave, _sub(self.p, "encoder."), self.cfg["encoder_rates"])
        h = pre_block(h, _sub(self.p, "pre_block."), self.cfg["num_attention_heads"], self.cfg["latent_channels"])
        return conv1d(h, self.p, "mean_proj")

    def decode(self, latents: mx.array) -> mx.array:
        """(B, frames, latent_channels) -> (B, frames * hop, 1) in [-1, 1]."""
        c = self.cfg
        h = conv1d(latents, self.p, "dec_in_proj")
        return bigvgan(h, _sub(self.p, "decoder."), c["decoder_rates"], c["decoder_kernel_sizes"],
                       c["resblock_kernel_sizes"], c["resblock_dilation_sizes"])
