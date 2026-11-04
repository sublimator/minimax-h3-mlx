"""MiniMax H3 transformer ops in MLX, one function per op.

Each op names the diffusers 0.42.1 code it mirrors (`transformer_minimax_h3.py` unless said otherwise). Weights are
passed in as arrays so the tests can copy them from a small randomly initialised torch module.
"""

import math

import mlx.core as mx
import numpy as np

MODALITY_NUM = 3  # 0 video, 1 text, 2 audio (MINIMAX_H3_MODALITY_NUM)


# --- rotary embedding (MiniMaxH3RotaryPosEmbed, _apply_rotary_emb) --------------------------------------------------


def rope_inv_freq(rope_freq_dim: int = 16, rope_theta: float = 10000.0) -> mx.array:
    """Built in numpy float32: bit-identical to torch's `theta ** (arange / 2f)`; MLX's pow is 1 ulp off at f=16."""
    exps = np.arange(0, 2 * rope_freq_dim, 2, dtype=np.float32) / np.float32(2 * rope_freq_dim)
    return mx.array((np.float32(1) / np.power(np.float32(rope_theta), exps)).astype(np.float32))


def rope_cos_sin(position_ids: mx.array, inv_freq: mx.array) -> tuple[mx.array, mx.array]:
    """(seq, 3) positions -> cos, sin of shape (seq, 2 * 3 * rope_freq_dim), float32."""
    pos = position_ids.astype(mx.float32)
    freqs = pos[:, :, None] * inv_freq[None, None, :]  # (seq, 3, f)
    freqs = freqs.reshape(pos.shape[0], -1)  # t block, h block, w block
    freqs = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rotary(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x: (batch, seq, heads, head_dim); rotates the leading cos.shape[-1] channels (rotate_half), passes the rest."""
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    c = cos.astype(x.dtype)[None, :, None, :]
    s = sin.astype(x.dtype)[None, :, None, :]
    x1, x2 = xr[..., : rot // 2], xr[..., rot // 2 :]
    rotated = mx.concatenate([-x2, x1], axis=-1)
    return mx.concatenate([xr * c + rotated * s, xp], axis=-1)


# --- norms and AdaLN ---------------------------------------------------------------------------------------------


def rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    """torch.nn.RMSNorm over the last axis. Output keeps x's dtype, as torch does with a wider affine weight."""
    if weight is None:
        weight = mx.ones((x.shape[-1],), dtype=x.dtype)
    return mx.fast.rms_norm(x, weight, eps).astype(x.dtype)


def silu(x: mx.array) -> mx.array:
    """torch's silu rounds once: compute in float32, cast back (a bf16 x * sigmoid(x) is off by one ulp at ~30% of
    inputs)."""
    xf = x.astype(mx.float32)
    return (xf * mx.sigmoid(xf)).astype(x.dtype)


def linear(x: mx.array, w: mx.array, b: mx.array | None = None) -> mx.array:
    """torch.nn.Linear: w is (out, in)."""
    y = x @ w.T
    return y + b if b is not None else y


def adaln_modulation(temb: mx.array, w: mx.array, b: mx.array, hidden_size: int) -> list[mx.array]:
    """MiniMaxH3AdaLayerNormModulation: (num_t, temb_dim) -> 6 x (num_t * 3, hidden), rows [t0m0, t0m1, t0m2, t1m0..].

    Order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp. silu at temb's precision, then cast to
    the projection's dtype.
    """
    out = linear(silu(temb).astype(w.dtype), w, b)
    out = out.reshape(-1, 6 * hidden_size)
    return mx.split(out, 6, axis=-1)


def modulate(x: mx.array, shift: mx.array, scale: mx.array, rows: mx.array) -> mx.array:
    """x * (1 + scale[rows]) + shift[rows]; x is (batch, seq, hidden), rows (seq,) index the AdaLN table."""
    return x * (1.0 + mx.take(scale, rows, axis=0)) + mx.take(shift, rows, axis=0)


def norm_out(
    x: mx.array, temb: mx.array, norm_w: mx.array, w: mx.array, b: mx.array, timestep_indices: mx.array, eps: float
) -> mx.array:
    """MiniMaxH3AdaLayerNormOut: RMSNorm, then shift/scale (in that order) picked per row by timestep index."""
    shift, scale = mx.split(linear(silu(temb).astype(w.dtype), w, b), 2, axis=-1)
    h = rms_norm(x, norm_w, eps)
    return h * (1.0 + mx.take(scale, timestep_indices, axis=0)) + mx.take(shift, timestep_indices, axis=0)


# --- attention ---------------------------------------------------------------------------------------------------


def attention(q: mx.array, k: mx.array, v: mx.array, mask: mx.array | None = None) -> mx.array:
    """Full self-attention. q, k, v: (batch, seq, heads, head_dim) as diffusers lays them out; returns the same layout.

    `mask` is an additive or boolean mask broadcastable to (batch, heads, seq_q, seq_k); the model itself passes none.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    qt, kt, vt = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
    o = mx.fast.scaled_dot_product_attention(qt, kt, vt, scale=scale, mask=mask)
    return o.transpose(0, 2, 1, 3)


def key_padding_mask(used: int, total: int) -> mx.array:
    """Boolean (1, 1, total, total) mask letting every query see only the first `used` keys (the reference pads the
    packed sequence to a multiple of 64 and splits the tail off with cu_seqlens = [0, used, S])."""
    keep = mx.arange(total) < used
    return mx.broadcast_to(keep[None, None, None, :], (1, 1, total, total))


def attn_block(
    x: mx.array,
    p: dict,
    heads: int,
    qk_eps: float,
    rotary: tuple[mx.array, mx.array] | None,
    mask: mx.array | None = None,
) -> mx.array:
    """MiniMaxH3AttnProcessor with separate to_q/to_k/to_v (no bias), per-head RMSNorm on q/k, rope, to_out."""
    b, s, _ = x.shape
    q = linear(x, p["to_q.weight"]).reshape(b, s, heads, -1)
    k = linear(x, p["to_k.weight"]).reshape(b, s, heads, -1)
    v = linear(x, p["to_v.weight"]).reshape(b, s, heads, -1)
    q = rms_norm(q, p["norm_q.weight"], qk_eps)
    k = rms_norm(k, p["norm_k.weight"], qk_eps)
    if rotary is not None:
        q = apply_rotary(q, *rotary)
        k = apply_rotary(k, *rotary)
    o = attention(q, k, v, mask).reshape(b, s, -1)
    return linear(o, p["to_out.0.weight"])


def swiglu_ff(x: mx.array, p: dict) -> mx.array:
    """diffusers FeedForward(activation_fn="swiglu", bias=False): proj -> (h, gate) -> h * silu(gate) -> out."""
    h = linear(x, p["net.0.proj.weight"])
    h, gate = mx.split(h, 2, axis=-1)
    return linear(h * silu(gate), p["net.2.weight"])


# --- timestep embedding (embeddings.Timesteps + TimestepEmbedding) -------------------------------------------------


def timestep_sinusoid(t: mx.array, dim: int = 256, max_period: float = 10000.0) -> mx.array:
    """get_timestep_embedding with flip_sin_to_cos=True, downscale_freq_shift=0, scale=1: [cos, sin]."""
    half = dim // 2
    exponent = -math.log(max_period) * mx.arange(half, dtype=mx.float32) / half
    args = t.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


def timestep_embedding(t: mx.array, p: dict, freq_dim: int = 256) -> mx.array:
    """time_proj then time_embedder (linear_1, silu, linear_2)."""
    e = timestep_sinusoid(t, freq_dim)
    e = linear(e, p["linear_1.weight"], p["linear_1.bias"])
    return linear(silu(e), p["linear_2.weight"], p["linear_2.bias"])


# --- patchify (modular_pipelines/minimax_h3/before_denoise.py, decoders.py) ---------------------------------------


def patchify(latents: mx.array, patch: tuple[int, int, int] = (1, 2, 2)) -> mx.array:
    """(B, C, F, H, W) -> (B * patches, C * pt * ph * pw); rows frame-major then row-major."""
    pt, ph, pw = patch
    b, c, f, h, w = latents.shape
    if f % pt or h % ph or w % pw:
        raise ValueError(f"latents {latents.shape} not divisible by patch {patch}")
    x = latents.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw)
    x = x.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(-1, c * pt * ph * pw)


def unpatchify(rows: mx.array, channels: int, frames: int, height: int, width: int, patch=(1, 2, 2)) -> mx.array:
    pt, ph, pw = patch
    x = rows.reshape(-1, frames // pt, height // ph, width // pw, channels, pt, ph, pw)
    x = x.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(-1, channels, frames, height, width)


# --- rotary position grids (before_denoise.py), float64 numpy like the reference ----------------------------------

ROPE_FRAME_RESCALE = 5.0 / 3.0
ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
ROPE_SPATIAL_SCALE = 32


def spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * ROPE_SPATIAL_SCALE


def temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    spans = np.array(
        [ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % len(ROPE_FRAMES_PER_LATENT)] for i in range(num_latent_frames)],
        dtype=np.float64,
    )
    return origin + np.concatenate([np.zeros(1), np.cumsum(spans[:-1])])


def frame_position_grid(latent_h: int, latent_w: int, patch_h: int, patch_w: int) -> tuple[np.ndarray, np.ndarray]:
    sqrt_area = np.sqrt(latent_h * latent_w)
    hg = spatial_position_grid(latent_h, patch_h, sqrt_area)
    wg = spatial_position_grid(latent_w, patch_w, sqrt_area)
    gh, gw = np.meshgrid(hg, wg, indexing="ij")
    return np.stack([gh.reshape(-1), gw.reshape(-1)], axis=-1), wg


# --- scheduler (schedulers/scheduling_minimax_h3.py) ----------------------------------------------------------------


def torch_linspace_f32(start: float, end: float, steps: int) -> np.ndarray:
    """torch.linspace(start, end, steps, dtype=float32) bit for bit (CPU, arm64): float32 step, the first half filled
    from `start` and the second from `end`, each as one fused multiply-add (exact product, single rounding).
    np.linspace differs by 1 ulp at e.g. steps=50."""
    if steps < 2:
        raise ValueError(f"steps must be >= 2, got {steps}")
    start32, end32 = np.float32(start), np.float32(end)
    step = np.float32((end32 - start32) / np.float32(steps - 1))
    i = np.arange(steps, dtype=np.float64)
    out = np.where(
        i < steps // 2,
        np.float64(start32) + np.float64(step) * i,
        np.float64(end32) - np.float64(step) * (steps - 1 - i),
    )
    return out.astype(np.float32)


def flow_sigmas(num_inference_steps: int, shift: float) -> np.ndarray:
    """linspace(1, 0, n) in float32, exponential shift, consecutive duplicates collapsed. Terminal 0 included."""
    if num_inference_steps is None or num_inference_steps < 2:
        raise ValueError(f"num_inference_steps must be >= 2, got {num_inference_steps}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    base = torch_linspace_f32(1.0, 0.0, num_inference_steps)
    shift32 = np.float32(shift)
    sig = (shift32 * base / (np.float32(1) + (shift32 - np.float32(1)) * base)).astype(np.float32)
    keep = np.concatenate([[True], sig[1:] != sig[:-1]])
    return sig[keep]


def flow_step(v: mx.array, t: float, x: mx.array, sigma: float, sigma_next: float) -> mx.array:
    """x0 = x + (1 - t) * v (sigma from the timestep, in x's dtype); x_next = r*x + (1 - r)*x0, r = s_next / s, f32."""
    st = (mx.array(1.0, dtype=x.dtype) - mx.array(t, dtype=x.dtype)).astype(x.dtype)
    x0 = x + st * v
    cdt = mx.float32 if x.dtype in (mx.float16, mx.bfloat16) else x.dtype
    s = mx.array(sigma, dtype=cdt)
    sn = mx.array(sigma_next, dtype=cdt)
    r = sn / s
    return (r * x.astype(cdt) + (1.0 - r) * x0.astype(cdt)).astype(x.dtype)


# --- one transformer block (MiniMaxH3TransformerBlock) --------------------------------------------------------------


def dit_block(
    x: mx.array,
    temb: mx.array,
    adaln_indices: mx.array,
    rotary: tuple[mx.array, mx.array],
    p: dict,
    heads: int,
    hidden_size: int,
    norm_eps: float = 1e-5,
    qk_eps: float = 1e-5,
    modulation: list[mx.array] | None = None,
) -> mx.array:
    """Pre-norm attention and SwiGLU FF, each shift/scale-modulated and gated per row from the (t, modality) table.
    `modulation`: the six precomputed tables (from `ModulationCache`) instead of projecting `temb` here."""
    if modulation is None:
        modulation = adaln_modulation(temb, p["adaln_proj.linear.weight"], p["adaln_proj.linear.bias"], hidden_size)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation
    sub = lambda prefix: {k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}  # noqa: E731
    h = modulate(rms_norm(x, p["norm1.weight"], norm_eps), shift_msa, scale_msa, adaln_indices)
    x = x + mx.take(gate_msa, adaln_indices, axis=0) * attn_block(h, sub("attn."), heads, qk_eps, rotary)
    h = modulate(rms_norm(x, p["norm2.weight"], norm_eps), shift_mlp, scale_mlp, adaln_indices)
    return x + mx.take(gate_mlp, adaln_indices, axis=0) * swiglu_ff(h, sub("ff."))


# --- token refiner and the whole forward (MiniMaxH3TokenRefiner, MiniMaxH3Transformer3DModel.forward) ---------------


def _sub(p: dict, prefix: str) -> dict:
    return {k[len(prefix) :]: v for k, v in p.items() if k.startswith(prefix)}


def token_refiner(x: mx.array, p: dict, num_layers: int, heads: int, norm_eps: float, qk_eps: float, final_eps: float):
    """Plain pre-norm blocks (no AdaLN, no rope), then a final RMSNorm."""
    for i in range(num_layers):
        b = _sub(p, f"refiner_blocks.{i}.")
        x = x + attn_block(rms_norm(x, b["norm1.weight"], norm_eps), _sub(b, "attn."), heads, qk_eps, None)
        x = x + swiglu_ff(rms_norm(x, b["norm2.weight"], norm_eps), _sub(b, "ff."))
    return rms_norm(x, p["final_norm.weight"], final_eps)


def dit_forward(
    p: dict,
    cfg: dict,
    video: mx.array,
    audio: mx.array,
    text: mx.array,
    timestep: mx.array,
    timestep_indices: mx.array,
    token_tags: mx.array,
    position_ids: mx.array,
    video_indices: mx.array,
    audio_indices: mx.array,
    text_indices: mx.array,
    cache: "ModulationCache | None" = None,
) -> tuple[mx.array, mx.array]:
    """MiniMaxH3Transformer3DModel.forward over one packed sequence. `p` is the state dict (diffusers names), `cfg`
    its config. Returns (video velocity rows, audio velocity rows) in the order of video_indices / audio_indices.
    With `cache`, the AdaLN tables come from it (the `adaln_proj` / `norm_out.linear` weights may then be absent)."""
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    eps, qk_eps, final_eps = cfg["norm_eps"], cfg["qk_norm_eps"], cfg["final_norm_eps"]
    seq = position_ids.shape[0]
    rotary = rope_cos_sin(position_ids, rope_inv_freq(cfg["rope_freq_dim"], cfg["rope_theta"]))

    #@@start project-and-scatter
    v = linear(video.astype(p["proj_in.weight"].dtype), p["proj_in.weight"], p["proj_in.bias"])
    a = linear(audio.astype(p["audio_proj_in.weight"].dtype), p["audio_proj_in.weight"], p["audio_proj_in.bias"])
    t = linear(text.astype(p["context_embedder.weight"].dtype), p["context_embedder.weight"], p["context_embedder.bias"])
    t = token_refiner(t, _sub(p, "token_refiner."), cfg["num_refiner_layers"], heads, eps, qk_eps, final_eps)

    x = mx.zeros((t.shape[0], seq, t.shape[-1]), dtype=t.dtype)
    x[:, text_indices] = t
    x[:, video_indices] = v.astype(t.dtype)
    x[:, audio_indices] = a.astype(t.dtype)
    #@@end project-and-scatter

    #@@start adaln-rows-and-blocks
    mx.eval(x)  # projections + refiner + scatter as their own graph, before the first block
    if cache is None:
        temb = timestep_embedding(timestep, _sub(p, "time_embedder."), cfg["freq_dim"])
        t_rows = timestep_indices
    else:
        temb = None
        t_rows = cache.rows(timestep, timestep_indices)  # step-local timestep index -> row of the cached table
    adaln_indices = t_rows * MODALITY_NUM + token_tags
    for i in range(cfg["num_layers"]):
        x = dit_block(x, temb, adaln_indices, rotary, _sub(p, f"transformer_blocks.{i}."), heads, hidden, eps, qk_eps,
                      modulation=None if cache is None else cache.blocks[i])
        mx.eval(x)  # one block's graph at a time: bounds activation memory at 20-50k rows
    #@@end adaln-rows-and-blocks

    #@@start norm-out-and-heads
    if cache is None:
        x = norm_out(x, temb, p["norm_out.norm.weight"], p["norm_out.linear.weight"], p["norm_out.linear.bias"],
                     timestep_indices, final_eps)
    else:
        shift, scale = cache.out
        h = rms_norm(x, p["norm_out.norm.weight"], final_eps)
        x = h * (1.0 + mx.take(scale, t_rows, axis=0)) + mx.take(shift, t_rows, axis=0)
    x = x.astype(p["proj_out.weight"].dtype)
    vid = linear(x, p["proj_out.weight"], p["proj_out.bias"])[:, video_indices]
    aud = linear(x, p["audio_proj_out.weight"], p["audio_proj_out.bias"])[:, audio_indices]
    #@@end norm-out-and-heads
    return vid, aud


# --- AdaLN modulation cache ---------------------------------------------------------------------------------------


class ModulationCache:
    """Every block's six AdaLN tables (and norm_out's shift/scale) for a fixed set of timestep values, computed once.

    The modulation depends only on t and the weights, never on the latents, so a sampling run can precompute it for
    the union of its timesteps and drop the `adaln_proj` weights (~26 GB of transformer_ref's ~66 GB in bf16). The
    rows of each table are [t0m0, t0m1, t0m2, t1m0, ...] over `self.t`, exactly what `adaln_modulation` returns.
    """

    def __init__(self, p: dict, cfg: dict, timesteps):
        t = np.unique(np.asarray(timesteps, dtype=np.float32))
        self.t = t
        self._index = {float(v): i for i, v in enumerate(t)}
        temb = timestep_embedding(mx.array(t), _sub(p, "time_embedder."), cfg["freq_dim"])
        hidden = cfg["hidden_size"]
        self.blocks = []
        for i in range(cfg["num_layers"]):
            b = f"transformer_blocks.{i}.adaln_proj.linear."
            tables = adaln_modulation(temb, p[b + "weight"], p[b + "bias"], hidden)
            mx.eval(tables)
            self.blocks.append(tables)
        w, bias = p["norm_out.linear.weight"], p["norm_out.linear.bias"]
        self.out = mx.split(linear(silu(temb).astype(w.dtype), w, bias), 2, axis=-1)
        mx.eval(self.out)

    def rows(self, timestep: mx.array, timestep_indices: mx.array) -> mx.array:
        """Map a step's (distinct timesteps, per-row index into them) onto rows of the cached tables."""
        local = np.array(timestep.astype(mx.float32))
        try:
            glob = np.array([self._index[float(v)] for v in local], dtype=np.int64)
        except KeyError as e:
            raise KeyError(f"timestep {e} not in the cache (built for {len(self.t)} values)") from None
        return mx.take(mx.array(glob), timestep_indices)

    @staticmethod
    def drop_weights(p: dict, cfg: dict) -> dict:
        """A new dict without the projections the cache replaces. The memory is freed only when nothing else (e.g. the
        caller's original dict) still references those arrays."""
        gone = {f"transformer_blocks.{i}.adaln_proj.linear.{n}" for i in range(cfg["num_layers"]) for n in ("weight", "bias")}
        gone |= {"norm_out.linear.weight", "norm_out.linear.bias"}
        return {k: v for k, v in p.items() if k not in gone}
