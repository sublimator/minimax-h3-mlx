"""VAE encoder op tests against diffusers 0.42.1 `autoencoder_kl_minimax_h3.py`, small random tensors."""

import mlx.core as mx
import pytest
import torch
from conftest import assert_close, to_mx
from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as ref

from h3mlx import vae_ops as V

TOL = dict(atol=5e-5, rtol=5e-5)


def cl(t: torch.Tensor) -> mx.array:
    """torch (B, C, F, H, W) -> MLX channels-last (B, F, H, W, C)."""
    return to_mx(t.permute(0, 2, 3, 4, 1).contiguous())


def back(a: mx.array) -> mx.array:
    return a.transpose(0, 4, 1, 2, 3)


def mlx_params(module: torch.nn.Module) -> dict:
    out = {}
    for k, v in module.state_dict().items():
        a = to_mx(v)
        out[k] = V.conv3d_weight(a) if a.ndim == 5 else a
    return out


@pytest.mark.parametrize("before,after,n", [(1, 1, 5), (2, 2, 6), (0, 1, 7), (0, 1, 2)])
def test_reflect_pad_matches_torch(before, after, n):
    x = torch.randn(1, 2, 3, n, 4)
    want = torch.nn.functional.pad(x, (0, 0, before, after, 0, 0), mode="reflect")
    got = V.reflect_pad(to_mx(x), 3, before, after)
    assert_close(got, want, atol=0, what="reflect")


@pytest.mark.parametrize(
    "cin,cout,k,stride,sp,tp",
    [(8, 16, 3, 1, 1, 2), (16, 16, 3, 1, 1, 2), (8, 16, 1, 1, 0, 0), (8, 8, 3, (2, 1, 1), 1, 2)],
)
def test_causal_conv3d_matches(cin, cout, k, stride, sp, tp):
    conv = ref.MiniMaxH3VideoCausalConv3d(cin, cout, kernel_size=k, stride=stride, spatial_padding=sp, temporal_padding=tp)
    x = torch.randn(2, cin, 5, 7, 6)
    want = conv(x)
    st = stride if isinstance(stride, tuple) else (stride,) * 3
    got = V.causal_conv3d(cl(x), V.conv3d_weight(to_mx(conv.weight)), to_mx(conv.bias), st, sp, tp)
    assert_close(back(got), want, **TOL, what="causal conv3d")


def test_causal_conv3d_is_causal():
    """Changing frame 3 must not change outputs at frames 0..2."""
    conv = ref.MiniMaxH3VideoCausalConv3d(4, 4, kernel_size=3, spatial_padding=1, temporal_padding=2)
    w, b = V.conv3d_weight(to_mx(conv.weight)), to_mx(conv.bias)
    x = torch.randn(1, 4, 6, 5, 5)
    x2 = x.clone()
    x2[:, :, 3] += 10
    y1 = V.causal_conv3d(cl(x), w, b, spatial_padding=1, temporal_padding=2)
    y2 = V.causal_conv3d(cl(x2), w, b, spatial_padding=1, temporal_padding=2)
    assert float(mx.abs(y1[:, :3] - y2[:, :3]).max()) == 0.0
    assert float(mx.abs(y1[:, 3] - y2[:, 3]).max()) > 0.1


def test_group_norm_per_frame_matches():
    gn = ref.MiniMaxH3VideoGroupNorm(4, 16, eps=1e-6, affine=True)
    torch.nn.init.normal_(gn.weight)
    torch.nn.init.normal_(gn.bias)
    x = torch.randn(2, 16, 5, 6, 7) * 3 + torch.arange(5).view(1, 1, 5, 1, 1) * 10  # frames differ in mean
    got = V.group_norm_per_frame(cl(x), 4, to_mx(gn.weight), to_mx(gn.bias), 1e-6)
    assert_close(back(got), gn(x), **TOL, what="group norm per frame")


@pytest.mark.parametrize("cin,cout", [(16, 16), (8, 16)])
def test_resnet_block3d_matches(cin, cout):
    blk = ref.MiniMaxH3VideoResnetBlock3d(cin, cout, norm_num_groups=4, norm_eps=1e-6)
    for n in (blk.norm1, blk.norm2):
        torch.nn.init.normal_(n.weight)
        torch.nn.init.normal_(n.bias)
    x = torch.randn(1, cin, 5, 8, 6)
    got = V.resnet_block3d(cl(x), mlx_params(blk), groups=4)
    assert_close(back(got), blk(x), **TOL, what="resnet block")


@pytest.mark.parametrize("ts,ss,shape", [(1, 2, (5, 8, 6)), (2, 2, (5, 7, 9)), (2, 1, (6, 4, 4))])
def test_downsample3d_matches(ts, ss, shape):
    d = ref.MiniMaxH3VideoDownsample3d(8, 8, temporal_stride=ts, spatial_stride=ss)
    x = torch.randn(1, 8, *shape)
    want = d(x)
    got = V.downsample3d(cl(x), V.conv3d_weight(to_mx(d.conv.weight)), to_mx(d.conv.bias), ts, ss)
    assert_close(back(got), want, **TOL, what="downsample")
