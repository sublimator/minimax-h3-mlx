"""The training example for one picture, with and without reference pictures.

The example follows a third-party trainer's recipe (see `train.build_example`), which is read and not run. Each step
that diffusers also has for sampling is tested against diffusers' own function: the two layouts, the row patchify, the
scheduler's forward process, the row timesteps. The torch side below assembles those in the order the recipe gives,
with the noise drawn from one torch generator in the pipeline's order. The loss mask and the reference encode's
float16 choice are tested on their own.
"""

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, to_mx, to_np
from diffusers.modular_pipelines.p_h3_image.before_denoise import (
    PH3ImageEditPrepareLayoutStep,
    PH3ImagePrepareLayoutStep,
    PH3ImageSetTimestepsStep,
    patchify_video_latents,
)
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from test_dit_tiny import TINY
from test_p_h3_image import PIXEL_MEAN, PIXEL_STD, VAE, back, cl, vae_pair  # noqa: F401  (vae_pair is a fixture)

from h3mlx import ops, train
from h3mlx.pipeline import Noise
from h3mlx.vae import encode_condition

CHANNELS, AUDIO = TINY["in_channels"], TINY["audio_in_channels"]
PATCH = tuple(TINY["patch_size"])
LEVEL = 0.999  # the conditioning timestep the tests pass; the builder has no default
TAGS = np.array([1] * 5 + [0] * 3 + [1] * 2)  # a prompt with a vision block's rows tagged as video
SIZES = [(6, 4), (4, 8)]  # latent height and width of up to two reference pictures, each its own


def latents_of(n_refs, seed=3, height=8, width=6):
    """A picture's latents and `n_refs` references', channels-first torch tensors (1, C, 1, h, w)."""
    g = torch.Generator().manual_seed(seed)
    picture = torch.randn(1, CHANNELS, 1, height, width, generator=g)
    return picture, [torch.randn(1, CHANNELS, 1, h, w, generator=g) for h, w in SIZES[:n_refs]]


def ours(picture, refs, sigma, seed=11, **kw):
    kw.setdefault("condition_t", LEVEL)
    return train.build_example(TINY, TAGS, cl(picture), sigma, Noise(seed), condition_latents=[cl(r) for r in refs], **kw)


def theirs(picture, refs, sigma, seed=11, level=LEVEL, audio_sigma=None):
    """The same example from diffusers' functions and torch."""
    lh, lw = picture.shape[3], picture.shape[4]
    tags = torch.from_numpy(TAGS)
    if refs:
        layout = PH3ImageEditPrepareLayoutStep.build_edit_packed_sequence(tags, refs, lh, lw, 2, PATCH, 2, 2, 0)
    else:
        layout = PH3ImagePrepareLayoutStep.build_packed_sequence(tags, 1, lh, lw, 2, PATCH, 2, 2, 0)
    position_ids, token_tags, video_indices, audio_indices, text_indices, n_cond, n_cond_audio = layout
    sch = MiniMaxH3Scheduler()
    g = torch.Generator().manual_seed(seed)
    rows = []
    for ref in refs:  # PH3ImagePrepareConditionLatentsStep
        drawn = torch.randn(ref.shape, generator=g)
        rows.append(patchify_video_latents(sch.scale_noise(ref, torch.tensor(level), drawn), PATCH))
    drawn = torch.randn(picture.shape, generator=g)  # PH3ImagePrepareLatentsStep's draws: the picture, then audio
    t = torch.tensor(1.0) - torch.tensor(float(np.float32(sigma)))
    t_audio = t if audio_sigma is None else torch.tensor(1.0) - torch.tensor(float(np.float32(audio_sigma)))
    rows.append(patchify_video_latents(sch.scale_noise(picture, t, drawn), PATCH))
    audio_noise = torch.randn((4, AUDIO), generator=g)
    audio_rows = sch.scale_noise(torch.zeros(4, AUDIO), t_audio, audio_noise)
    timestep, indices = PH3ImageSetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, n_cond, n_cond_audio, text_indices.numel(), float(t), float(t_audio),
        max(float(t), level), 1.0)
    return dict(position_ids=position_ids, token_tags=token_tags, video_indices=video_indices,
                audio_indices=audio_indices, text_indices=text_indices, n_cond=n_cond, n_cond_audio=n_cond_audio,
                video_rows=torch.cat(rows), audio_rows=audio_rows, audio_noise=audio_noise,
                target=patchify_video_latents(picture - drawn, PATCH), clean=patchify_video_latents(picture, PATCH),
                noise=patchify_video_latents(drawn, PATCH), timestep=timestep, indices=indices)


def same(got, want):
    return np.array_equal(to_np(got), to_np(want))


# --- the example against diffusers' pieces -------------------------------------------------------------------------


@pytest.mark.parametrize("n_refs", [0, 1, 2])
@pytest.mark.parametrize("sigma", [1.0, 0.7, 0.25, 0.0005])
def test_example_is_diffusers_pieces(n_refs, sigma):
    """Layout, rows, audio rows, target and row timesteps, bit for bit, for text-to-image and an edit with one and two
    references, at four noise levels: pure noise, two in between, and one whose timestep (0.9995) is above the
    conditioning level."""
    picture, refs = latents_of(n_refs)
    ex = ours(picture, refs, sigma)
    want = theirs(picture, refs, sigma)
    lay = ex.layout
    assert np.array_equal(lay.position_ids, want["position_ids"].numpy())
    for name in ("token_tags", "video_indices", "audio_indices", "text_indices"):
        assert np.array_equal(getattr(lay, name), want[name].numpy()), name
    assert (lay.num_condition_video_rows, lay.num_condition_audio_rows) == (want["n_cond"], want["n_cond_audio"])
    assert ex.num_condition_rows == sum(h * w // 4 for h, w in SIZES[:n_refs])
    assert ex.video_rows.dtype == ex.audio_rows.dtype == ex.target.dtype == mx.float32
    for name in ("video_rows", "audio_rows", "target"):
        assert same(getattr(ex, name), want[name]), name
    assert same(ex.clean_rows, want["clean"])
    assert np.array_equal(ex.timestep, want["timestep"].numpy()) and ex.timestep.dtype == np.float32
    assert np.array_equal(ex.timestep_indices, want["indices"].numpy())
    assert ex.sigma == np.float32(sigma)
    generated = ex.video_rows[ex.num_condition_rows:]
    assert generated.shape == ex.target.shape == (picture.shape[3] * picture.shape[4] // 4, CHANNELS * 4)
    if sigma == 1.0:  # the sampler's starting point for the same seed: noise, with the references in front
        assert same(generated, want["noise"]) and same(ex.audio_rows, want["audio_noise"])
        assert ex.timestep[0] == 0.0
    # the recipe writes the audio rows as sigma times the noise. Ours multiply by 1 - t, which is sigma itself down to
    # sigma 0.5 and below that sigma after the float32 rounding of t = 1 - sigma: within 6e-8 of it
    noise = want["audio_noise"].numpy()
    by_sigma = np.float32(sigma) * noise
    if sigma >= 0.5:
        assert same(ex.audio_rows, by_sigma)
    else:
        assert (np.abs(to_np(ex.audio_rows) - by_sigma) <= 6e-8 * np.abs(noise) + 1e-12).all()


def test_example_guards():
    """What the comparison above would not notice by itself: the references and the picture's noise are different
    draws, the references are not left clean, and a second example from the same seed is the same example."""
    picture, refs = latents_of(2)
    ex = ours(picture, refs, 0.5)
    again = ours(picture, refs, 0.5)
    assert same(ex.video_rows, again.video_rows) and same(ex.target, again.target)
    other = ours(picture, refs, 0.5, seed=12)
    assert not same(ex.video_rows, other.video_rows) and not same(ex.audio_rows, other.audio_rows)
    n = ex.num_condition_rows
    clean_refs = mx.concatenate([ops.patchify(to_mx(r), PATCH) for r in refs])
    assert not same(ex.video_rows[:n], clean_refs)
    assert_close(ex.video_rows[:n], clean_refs, atol=0.01, what="references sit close to clean: t = 0.999")
    # a level of its own for the audio rows moves them and their timestep, and nothing else
    apart = ours(picture, refs, 0.5, audio_sigma=0.9)
    want = theirs(picture, refs, 0.5, audio_sigma=0.9)
    assert same(apart.audio_rows, want["audio_rows"]) and same(apart.video_rows, ex.video_rows)
    assert np.array_equal(apart.timestep, want["timestep"].numpy()) and len(apart.timestep) == len(ex.timestep) + 1
    assert not same(apart.audio_rows, ex.audio_rows)
    # another conditioning level moves the references' rows and their timestep
    lower = ours(picture, refs, 0.5, condition_t=0.9)
    want = theirs(picture, refs, 0.5, level=0.9)
    assert same(lower.video_rows, want["video_rows"]) and np.array_equal(lower.timestep, want["timestep"].numpy())
    assert not same(lower.video_rows[:n], ex.video_rows[:n]) and same(lower.video_rows[n:], ex.video_rows[n:])


# --- the target's sign, at the level of an example -----------------------------------------------------------------


@pytest.mark.parametrize("sigma", [1.0, 0.7, 0.25, 0.0005])
def test_example_target_steps_back_to_the_picture(sigma):
    """From the example's noised rows, one Euler step along its target to sigma 0 gives the picture's clean rows:
    with the scheduler's own `step` and with `flow_step`. Along the flipped target it does not."""
    picture, refs = latents_of(1)
    ex = ours(picture, refs, sigma)
    generated = ex.video_rows[ex.num_condition_rows:]
    tol = 4e-6
    sch = MiniMaxH3Scheduler(shift_provided_sigmas=False)
    sch.set_timesteps(sigmas=torch.tensor([float(np.float32(sigma)), 0.0]))
    stepped = sch.step(torch.from_numpy(to_np(ex.target)), sch.timesteps[0], torch.from_numpy(to_np(generated)), return_dict=False)[0]
    assert_close(stepped, ex.clean_rows, atol=tol, what="the scheduler's step along the example's target")
    t = float(np.float32(1) - np.float32(sigma))
    landed = ops.flow_step(ex.target, t, generated, float(np.float32(sigma)), 0.0)
    assert_close(landed, ex.clean_rows, atol=tol, what="flow_step along the example's target")
    flipped = ops.flow_step(-ex.target, t, generated, float(np.float32(sigma)), 0.0)
    assert float(mx.abs(flipped - ex.clean_rows).max()) > 100 * tol, "2 * sigma * (x_0 - noise) away"


# --- the loss mask ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_refs", [0, 1, 2])
def test_loss_mask_is_the_generated_video_rows(n_refs):
    """The mask over the packed sequence, built here from diffusers' layout: True on the video rows after the
    condition rows and nowhere else. `Example.generated` picks the same rows out of the transformer's video output."""
    picture, refs = latents_of(n_refs)
    ex = ours(picture, refs, 0.5)
    want = theirs(picture, refs, 0.5)
    expected = torch.zeros(len(want["token_tags"]), dtype=torch.bool)
    expected[want["video_indices"][want["n_cond"]:]] = True
    mask = train.loss_mask(ex.layout)
    assert mask.dtype == bool and np.array_equal(mask, expected.numpy())
    n_generated = picture.shape[3] * picture.shape[4] // 4
    assert mask.sum() == n_generated and not mask[ex.layout.text_indices].any() and not mask[ex.layout.audio_indices].any()
    assert not mask[ex.layout.video_indices[:ex.num_condition_rows]].any()
    video_out = mx.arange(len(ex.layout.video_indices) * 3, dtype=mx.float32).reshape(1, -1, 3)  # stands for a prediction
    picked = ex.generated(video_out)
    assert picked.shape == (1, n_generated, 3) and same(picked, video_out[:, ex.num_condition_rows:])
    # the same rows, read off the packed sequence with the mask
    packed = np.zeros((len(mask), 3), np.float32)
    packed[ex.layout.video_indices] = to_np(video_out[0])
    assert np.array_equal(packed[mask], to_np(picked[0]))


def test_loss_mask_does_not_depend_on_timesteps():
    """At sigma 0.0005 the picture's timestep is 0.9995, above the conditioning level, so the references are held at
    0.9995 too: a generated row and a condition row carry the same timestep value, and the table of distinct
    timesteps has one entry. A mask read off timesteps would take the references' rows as rows to train on; the
    layout's mask is the same as at any other level."""
    picture, refs = latents_of(2)
    low, mid = ours(picture, refs, 0.0005), ours(picture, refs, 0.5)
    assert len(mid.timestep) == 2 and len(low.timestep) == 1, "the picture's (and audio's) level, and the references'"
    row_t = low.timestep[low.timestep_indices]
    cond_rows = low.layout.video_indices[:low.num_condition_rows]
    generated_rows = low.layout.video_indices[low.num_condition_rows:]
    assert row_t[cond_rows[0]] == row_t[generated_rows[0]] == np.float32(1) - np.float32(0.0005)
    by_timestep = (low.layout.token_tags == 0) & (row_t == row_t[generated_rows[0]])
    assert by_timestep[cond_rows].all(), "the counterexample: a timestep test lets the condition rows in"
    mask = train.loss_mask(low.layout)
    assert np.array_equal(mask, train.loss_mask(mid.layout)) and not mask[cond_rows].any() and mask[generated_rows].all()
    assert mask.sum() < by_timestep.sum()


# --- the reference pictures' encode ------------------------------------------------------------------------------------


def test_reference_encode_float16_rounding_is_a_choice(vae_pair):  # noqa: F811
    """`encode_condition` with the float16 round trip is the pipeline's (tested against diffusers elsewhere). Without
    it, the latents are the posterior sample itself: diffusers' function with that one line taken out, written out
    here. The two differ by float16's rounding, which the comparison with torch is too loose to see, so the rounding
    is checked directly: un-normalised, the rounded latents are float16 values and the others are not."""
    ref, vae = vae_pair
    pixels = torch.randint(0, 256, (1, 3, 1, 48, 40), dtype=torch.uint8)
    mean, std = torch.tensor(VAE["latents_mean"]).view(1, -1, 1, 1, 1), torch.tensor(VAE["latents_std"]).view(1, -1, 1, 1, 1)
    pm, ps = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1), torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1)
    with torch.no_grad():
        posterior = ref.encode((pixels.to(torch.float32).div(255.0) - pm) / ps, return_dict=False)[0]
        sample = posterior.sample(generator=torch.Generator().manual_seed(42))
    want = (sample - mean) / std  # encode_vae_condition without `latents.to(torch.float16).float()`

    def torch_noise(shape_cl):
        b, f, h, w, c = shape_cl
        return cl(torch.randn((b, c, f, h, w), generator=torch.Generator().manual_seed(42)))

    args = (vae, cl(pixels), PIXEL_MEAN, PIXEL_STD, VAE["latents_mean"], VAE["latents_std"], torch_noise)
    plain = encode_condition(*args, float16_round=False)
    rounded = encode_condition(*args, float16_round=True)
    assert_close(back(plain), want, atol=2e-3, rtol=2e-3, what="the posterior sample, normalised")
    assert same(rounded, encode_condition(*args)), "rounding is the default: the pipeline's path is unchanged"

    def unnormalised(x):
        return to_np(x) * np.float32(VAE["latents_std"]) + np.float32(VAE["latents_mean"])

    def off_float16(x):  # how far the un-normalised latents are from float16 values
        u = unnormalised(x)
        return float(np.abs(u - u.astype(np.float16).astype(np.float32)).max())

    assert off_float16(rounded) < 1e-6, "float16 values, up to the float32 rounding of normalising and back"
    assert off_float16(plain) > 1e-4, "not float16 values"
    apart = float(np.abs(unnormalised(rounded) - unnormalised(plain)).max())
    assert 1e-5 < apart < 2e-3, "the two differ by float16's rounding and no more"


def test_encode_references_needs_the_choice(vae_pair):  # noqa: F811
    """`train.encode_references` is `image.encode_pictures` with the rounding as a required choice: no default, and
    each answer reaches the encode."""
    from types import SimpleNamespace

    from h3mlx import image

    _, vae = vae_pair
    c = SimpleNamespace(vae=vae, vae_latents_mean=VAE["latents_mean"], vae_latents_std=VAE["latents_std"])
    pictures = [np.random.default_rng(i).integers(0, 256, (48, 40, 3), dtype=np.uint8) for i in range(2)]
    with pytest.raises(TypeError, match="float16_round"):
        train.encode_references(c, pictures)
    rounded = train.encode_references(c, pictures, float16_round=True)
    plain = train.encode_references(c, pictures, float16_round=False)
    pipeline = image.encode_pictures(c, pictures)
    assert len(rounded) == len(plain) == 2 and rounded[0].shape == (1, 1, 12, 10, 4)
    for r, p, want in zip(rounded, plain, pipeline):
        assert same(r, want), "True is the pipeline's encode"
        assert not same(p, want) and float(mx.abs(p - want).max()) < 2e-3, "False differs from it by the rounding"
    assert not same(rounded[0], rounded[1])


# --- what is refused ---------------------------------------------------------------------------------------------------


def test_example_refusals():
    picture, refs = latents_of(1)
    with pytest.raises(TypeError, match="condition_t"):  # no default conditioning level
        train.build_example(TINY, TAGS, cl(picture), 0.5, Noise(0))
    for bad in (0.0, -0.1, 1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="sigma must lie in"):
            ours(picture, refs, bad)
        with pytest.raises(ValueError, match="audio_sigma must lie in"):
            ours(picture, refs, 0.5, audio_sigma=bad)
    for bad in (-0.1, 1.5, float("nan")):
        with pytest.raises(ValueError, match="condition_t must lie in"):
            ours(picture, refs, 0.5, condition_t=bad)
    with pytest.raises(ValueError, match=r"latents must be \(1, 1, height, width, 4\)"):  # channels-first by mistake
        train.build_example(TINY, TAGS, to_mx(picture), 0.5, Noise(0), condition_t=LEVEL)
    with pytest.raises(ValueError, match=r"condition_latents\[0\] must be"):
        train.build_example(TINY, TAGS, cl(picture), 0.5, Noise(0), condition_t=LEVEL, condition_latents=[to_mx(refs[0])])
    two_frames = mx.zeros((1, 2, 8, 6, CHANNELS))
    with pytest.raises(ValueError, match="latents must be"):
        train.build_example(TINY, TAGS, two_frames, 0.5, Noise(0), condition_t=LEVEL)
