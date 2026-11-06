"""P-H3-Image against diffusers' own implementation (main at d3d8af89, `PH3Image*`), on tiny random models: the text
connector, the single-frame image VAE, the text-to-image and edit layouts, the edit resize rules, the text encoder's
several hidden states, and both workflows end to end through `PH3ImageBlocks`."""

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import RELEASE, assert_close, state_mx, to_mx
from PIL import Image

from h3mlx import image as I
from h3mlx import load
from h3mlx import vae_ops as V
from h3mlx.packing import Ref, build_packed_sequence, build_ref2va_packed_sequence
from h3mlx.vae import VaeConfig, VideoVAE, encode_condition

# Temporal strides only where the spatial stride is 2, as in the release: a temporal-only downsample pads no space, so
# (2, 2, 1) / (1, 2, 2) would shrink a 48x40 picture to a 10x8 latent instead of 12x10 (both sides agree, but no
# released VAE takes that path).
VAE = dict(latent_channels=4, block_out_channels=(8, 8, 8), layers_per_block=1, spatial_downsample_factors=(2, 2, 1),
           temporal_downsample_factors=(1, 2, 1), norm_num_groups=4, decoder_num_layers=2,
           decoder_num_attention_heads=2, decoder_attention_head_dim=16, decoder_num_register_tokens=2,
           latents_mean=tuple(0.1 * i for i in range(4)), latents_std=tuple(1.0 + 0.2 * i for i in range(4)))
TILE = dict(tile_h=32, tile_w=32, tile_overlap_h=8, tile_overlap_w=8)
PIXEL_MEAN, PIXEL_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NEEDS_PROCESSOR = pytest.mark.skipif(not (RELEASE / "processor").exists(),
                                     reason="release fixtures not present (scripts/fetch_test_fixtures.py)")


def cl(t):
    return to_mx(t.permute(0, 2, 3, 4, 1).contiguous())


def back(a):
    return a.transpose(0, 4, 1, 2, 3)


def randomize(module, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, prm in module.named_parameters():
            if name.endswith(("scale1", "scale2")):
                prm.normal_(0.0, 0.3)
            elif name.endswith("register_tokens") or ("norm" in name and prm.ndim == 1):
                prm.normal_(1.0, 0.2)
    return module


# -- connector -------------------------------------------------------------------------------------------------------


def make_connector(indices, hidden, out, seed=1):
    from diffusers.models.condition_embedders.condition_embedder_p_h3_image import PH3ImageTextConnector

    ref = PH3ImageTextConnector(hidden_state_indices=indices, text_hidden_size=hidden, out_channels=out).eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        ref.input_mean.normal_()
        ref.input_std.uniform_(0.5, 2.0)
        ref.proj.weight.normal_(0.0, 0.1)
        ref.output_mean.normal_()
        ref.output_std.uniform_(0.5, 2.0)
        ref.sink_embedding.normal_()
    return ref


def test_connector_matches_diffusers():
    ref = make_connector((1, 3, 4), 16, 24)
    states = tuple(torch.randn(1, 7, 16) for _ in range(5))
    with torch.no_grad():
        want = ref(states)
    got = I.connect(state_mx(ref), [to_mx(states[i]) for i in (1, 3, 4)], out_dtype=mx.float32)
    assert_close(got, want, atol=1e-5, rtol=1e-5, what="connector")
    assert_close(got[:, 0], ref.sink_embedding[None], atol=0, what="token 0 is the sink")


def test_edit_prompt_prefix():
    assert I.edit_prompt("make it red") == "Edit <Picture 1>. make it red"
    assert I.edit_prompt("  Edit <Picture 1>. Keep it.") == "  Edit <Picture 1>. Keep it."
    assert I.edit_prompt("subject_definitions:\n<Subject 1>") == "subject_definitions:\n<Subject 1>"


# -- image VAE -------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vae_pair():
    from diffusers.models.autoencoders.autoencoder_kl_p_h3_image import AutoencoderKLPH3Image

    ref = randomize(AutoencoderKLPH3Image(**VAE).eval(), 3)
    ref.enable_tiling(tile_sample_min_height=32, tile_sample_min_width=32, tile_sample_min_overlap_height=8,
                      tile_sample_min_overlap_width=8)
    p = {k: (V.conv3d_weight(to_mx(v)) if v.ndim == 5 else to_mx(v)) for k, v in ref.state_dict().items()}
    ours = VideoVAE(p, VaeConfig.from_diffusers({"_class_name": "AutoencoderKLPH3Image", **dict(ref.config)}, **TILE))
    assert ours.cfg.single_frame
    return ref, ours


@pytest.mark.parametrize("hw", [(10, 12), (8, 8), (25, 13)])
def test_image_decode_matches(vae_pair, hw):
    ref, ours = vae_pair
    z = torch.randn(1, 4, 1, *hw)
    with torch.no_grad():
        want = ref.decode(z).sample
    got = ours.decode(cl(z))
    assert got.shape[1] == 1
    assert_close(back(got), want, atol=3e-4, rtol=3e-4, what=f"image decode {hw} (tiled)")


def test_image_decode_untiled_and_frame_position(vae_pair):
    ref, ours = vae_pair
    z = torch.randn(1, 4, 1, 6, 5)
    ref.disable_tiling()
    try:
        with torch.no_grad():
            want = ref.decode(z).sample
    finally:
        ref.enable_tiling()
    untiled = VideoVAE(ours.p, VaeConfig.from_diffusers(dict(ref.config), single_frame=True, use_tiling=False))
    assert_close(back(untiled.decode(cl(z))), want, atol=3e-4, rtol=3e-4, what="image decode untiled")
    # guard: H3's video decoder positions (frame at 0, a temporal patch) are not this decoder
    pos = np.array(V.vit_positions(1, 2, 2, 0, 2.0 * (0.5 / 7) - 1.0))
    assert np.allclose(pos[:, 0], np.float32(2.0 * (0.5 / 7) - 1.0)) and not np.allclose(pos[:, 0], 0.0)


def test_image_encode_and_condition_recipe(vae_pair):
    from diffusers.modular_pipelines.p_h3_image.encoders import encode_vae_condition

    ref, ours = vae_pair
    x = torch.randn(1, 3, 1, 48, 40)
    with torch.no_grad():
        want = ref.encode(x).latent_dist.parameters
    assert_close(back(ours.encode_moments(cl(x))), want, atol=3e-4, rtol=3e-4, what="image moments (tiled)")

    pixels = torch.randint(0, 256, (1, 3, 1, 48, 40), dtype=torch.uint8)
    want = encode_vae_condition(ref, pixels, PIXEL_MEAN, PIXEL_STD, encode_seed=42)

    def torch_noise(shape_cl):
        b, f, h, w, c = shape_cl
        return cl(torch.randn((b, c, f, h, w), generator=torch.Generator().manual_seed(42)))

    got = encode_condition(ours, cl(pixels), PIXEL_MEAN, PIXEL_STD, VAE["latents_mean"], VAE["latents_std"], torch_noise)
    assert_close(back(got), want, atol=2e-3, rtol=2e-3, what="picture conditioning latents")


def test_image_vae_refuses_video(vae_pair):
    _, ours = vae_pair
    with pytest.raises(ValueError):
        ours.decode(mx.zeros((1, 2, 4, 4, 4)))
    with pytest.raises(ValueError):
        ours.encode_moments(mx.zeros((1, 5, 32, 32, 3)))


# -- layouts ---------------------------------------------------------------------------------------------------------


def _same_layout(ours, want):
    pos, tags, vid, aud, txt, ncv, nca = want
    assert np.array_equal(ours.position_ids, pos.numpy()), "position_ids differ"
    assert np.array_equal(ours.token_tags, tags.numpy())
    assert np.array_equal(ours.video_indices, vid.numpy())
    assert np.array_equal(ours.audio_indices, aud.numpy())
    assert np.array_equal(ours.text_indices, txt.numpy())
    assert (ours.num_condition_video_rows, ours.num_condition_audio_rows) == (ncv, nca)


@pytest.mark.parametrize("lh,lw,n_text", [(182, 182, 37), (136, 242, 5), (10, 14, 1)])
def test_text2image_layout_matches(lh, lw, n_text):
    from diffusers.modular_pipelines.p_h3_image.before_denoise import PH3ImagePrepareLayoutStep

    tags = np.ones(n_text, dtype=np.int64)
    want = PH3ImagePrepareLayoutStep.build_packed_sequence(torch.from_numpy(tags), 1, lh, lw, 2, (1, 2, 2), 2, 2, 0)
    _same_layout(build_packed_sequence(tags, 1, lh, lw, 2, (1, 2, 2), 2), want)


@pytest.mark.parametrize("shapes", [[(128, 128)], [(128, 96), (64, 64), (180, 100)], [(16, 8)] * 8])
def test_edit_layout_matches(shapes):
    from diffusers.modular_pipelines.p_h3_image.before_denoise import PH3ImageEditPrepareLayoutStep

    rng = np.random.default_rng(0)
    tags = rng.integers(0, 2, 41).astype(np.int64)  # text 1, vision blocks 0
    cond = [torch.zeros(1, 24, 1, h, w) for h, w in shapes]
    lh, lw = shapes[0]
    want = PH3ImageEditPrepareLayoutStep.build_edit_packed_sequence(torch.from_numpy(tags), cond, lh, lw, 2, (1, 2, 2),
                                                                    2, 2, 0)
    ours = build_ref2va_packed_sequence(tags, [Ref("image")] * len(shapes), [(1, h, w) for h, w in shapes], [], 1, lh,
                                        lw, 2, (1, 2, 2), 2)
    _same_layout(ours, want)


def test_schedule_is_the_grid_verbatim():
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    s = MiniMaxH3Scheduler(shift=12.0, shift_provided_sigmas=False)
    s.set_timesteps(sigmas=list(I.SAMPLE_SIGMAS))
    ours = I.schedules()
    assert np.array_equal(ours.video_sigmas, s.sigmas.numpy()) and np.array_equal(ours.audio_sigmas, s.sigmas.numpy())
    assert np.array_equal(ours.video_timesteps, s.timesteps.numpy())
    assert len(ours.video_timesteps) == 8 and not ours.two_time
    with pytest.raises(ValueError):
        I.schedules([1.0, 0.5, 0.6, 0.0])


# -- sizes -----------------------------------------------------------------------------------------------------------


def test_size_rules_match():
    from diffusers.modular_pipelines.p_h3_image import before_encoder as B

    rng = np.random.default_rng(1)
    for _ in range(400):
        w, h = (int(x) for x in rng.integers(17, 5000, 2))
        assert I.size_at_area(w, h, 1_000_000, 32) == B._size_at_area(w, h, 1_000_000, 32)
        assert I.size_in_area_range(w, h, 1024**2, 8 * 1024**2, 32) == B._size_in_area_range(w, h, 1024**2,
                                                                                               8 * 1024**2, 32)
        with _quiet():
            assert I.round_canvas(h, w, 32) == B.round_canvas(h, w, 32)
    assert I.size_at_area(1024, 992, 1_000_000, 32) == (1024, 992)  # on the multiple, within 20%: kept


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")

    def __exit__(self, *a):
        return self._w.__exit__(*a)


SMALL = dict(reference_min_pixels=48 * 48, reference_max_pixels=96 * 96, canvas_min_pixels=64 * 64,
             vision_pixels=40 * 40)


def _resize_pipe():
    from diffusers.modular_pipelines.p_h3_image.before_encoder import PH3ImageEditResizeStep

    pipe = PH3ImageEditResizeStep().init_pipeline()
    pipe.register_to_config(**SMALL)
    pipe.load_components()
    return pipe


class _Cfg:  # what prepare_edit reads of the components: the canvas multiple and the settings
    def __init__(self, multiple, settings):
        self.canvas_multiple, self.settings = multiple, settings


@pytest.mark.parametrize("canvas", [None, (70, 90)])
def test_prepare_edit_matches_resize_step(canvas):
    rng = np.random.default_rng(2)
    images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in
              ((50, 70), (130, 40), (64, 64))]
    pipe = _resize_pipe()
    kw = {} if canvas is None else dict(height=canvas[0], width=canvas[1])
    with _quiet():
        want = pipe(image=images, output=["height", "width", "reference_images", "vision_images"], **kw)
        h, w, refs, vis = I.prepare_edit(_Cfg(32, I.Settings(**SMALL)), images, *(canvas or (None, None)))
    assert (h, w) == (want["height"], want["width"])
    for got, exp in zip(refs + vis, want["reference_images"] + want["vision_images"]):
        assert got.size == exp.size and np.array_equal(np.asarray(got), np.asarray(exp))


# -- text encoder: several hidden states -----------------------------------------------------------------------------


@NEEDS_PROCESSOR
def test_hidden_states_and_connector_match_pipeline():
    """`get_p_h3_image_prompt_embeds` (transformers' output_hidden_states -> connector) against mlx-vlm with the
    layer capture, on an edit presentation with two pictures. Indices 1 and 2 sit where deepstack adds features, so
    they pin down that a state is the layer's output before that add."""
    from diffusers.modular_pipelines.p_h3_image.encoders import get_p_h3_image_prompt_embeds
    from test_text_encoder_tiny import mlx_model

    hf, proc = _qwen()
    indices = (1, 2, 3, 4)
    conn = make_connector(indices, 64, 48, seed=4)
    rng = np.random.default_rng(5)
    pics = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in ((96, 64), (64, 96))]
    comp = _Cfg(32, I.Settings())
    comp.processor, comp.tokenizer = proc, proc.tokenizer
    ids, tags, vision = I.edit_presentation(comp, "give her the hat", pics)
    tv = {k: torch.from_numpy(np.asarray(v)) for k, v in vision.items()}
    with torch.no_grad():
        want = get_p_h3_image_prompt_embeds(hf, conn, proc, ids, tv, dtype=torch.float32)
    model = mlx_model(hf)
    load.eval_each_layer(model.language_model.model.layers)
    states = load.hidden_states_at(model, ids, vision, indices, 4)
    got = I.connect(state_mx(conn), states, out_dtype=mx.float32)
    assert_close(got, want, atol=2e-3, rtol=2e-3, what="edit conditioning (states 1-4 through the connector)")
    assert tags.count(0) == sum(i in (151652, 151653, 151655) for i in ids)


def _qwen():
    from test_text_encoder_tiny import IDS, ROPE, TEXT, VISION
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    torch.manual_seed(5)
    cfg = Qwen3VLConfig(text_config=dict(TEXT, rope_parameters=dict(ROPE, rope_theta=5e6)), vision_config=VISION, **IDS)
    hf = Qwen3VLForConditionalGeneration(cfg).eval().float()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.normal_(1.0, 0.3)
    return hf, Qwen3VLProcessor.from_pretrained(str(RELEASE / "processor"))


# -- end to end ------------------------------------------------------------------------------------------------------


def build():
    from diffusers.models.autoencoders.autoencoder_kl_p_h3_image import AutoencoderKLPH3Image
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
    from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler

    hf, proc = _qwen()
    vae = randomize(AutoencoderKLPH3Image(**VAE).eval(), 7)
    vae.enable_tiling(tile_sample_min_height=32, tile_sample_min_width=32, tile_sample_min_overlap_height=8,
                      tile_sample_min_overlap_width=8)
    dit = randomize(MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=32, hidden_size=48, num_layers=2, num_refiner_layers=1, ffn_dim=96,
        in_channels=4, audio_in_channels=4, patch_size=(1, 2, 2), text_dim=64, freq_dim=32, time_embed_hidden_dim=40,
        time_embed_dim=24, rope_freq_dim=4,
    ).eval(), 8)
    conn = make_connector((1, 3, 4), 64, 64, seed=9)
    return dict(qwen=hf, proc=proc, vae=vae, dit=dit, conn=conn,
                schedulers=[MiniMaxH3Scheduler(shift=12.0, shift_provided_sigmas=False) for _ in range(2)])


E2E = dict(reference_min_pixels=32 * 32, reference_max_pixels=64 * 64, canvas_min_pixels=48 * 48,
           vision_pixels=48 * 48)


def run_diffusers(t, prompt, images, height, width, seed):
    from diffusers.modular_pipelines.p_h3_image.modular_blocks_p_h3_image import PH3ImageBlocks

    blocks = PH3ImageBlocks()
    blocks.sub_blocks.pop("prompt_upsampler")
    pipe = blocks.init_pipeline()
    pipe.update_components(text_encoder=t["qwen"], text_connector=t["conn"], tokenizer=t["proc"].tokenizer,
                           processor=t["proc"], vae=t["vae"], transformer=t["dit"], scheduler=t["schedulers"][0],
                           audio_scheduler=t["schedulers"][1])
    pipe.register_to_config(**E2E)
    pipe.load_components(names=["image_processor"])
    kw = dict(image=images) if images else {}
    if height is not None:
        kw.update(height=height, width=width)
    with _quiet():
        out = pipe(prompt=prompt, generator=torch.Generator().manual_seed(seed), output_type="np",
                   output=["images", "prompt_embeds", "text_token_tags"] + (["condition_rows"] if images else []), **kw)
    return np.asarray(out["images"])[0], out["prompt_embeds"], out["text_token_tags"], out.get("condition_rows")


def components(t):
    from h3mlx.vae import VaeConfig

    qm = __import__("test_text_encoder_tiny").mlx_model(t["qwen"])
    load.eval_each_layer(qm.language_model.model.layers)
    vae_p = {k: (V.conv3d_weight(to_mx(v)) if v.ndim == 5 else to_mx(v)) for k, v in t["vae"].state_dict().items()}
    vcfg = dict(t["vae"].config)
    c = I.Components(
        tokenizer=t["proc"].tokenizer, processor=t["proc"],
        encode_states=lambda ids, vision: load.hidden_states_at(qm, ids, vision, (1, 3, 4), 4),
        connector=state_mx(t["conn"]),
        vae=VideoVAE(vae_p, VaeConfig.from_diffusers(vcfg, single_frame=True, **TILE)),
        vae_latents_mean=tuple(vcfg["latents_mean"]), vae_latents_std=tuple(vcfg["latents_std"]),
        dit_params=state_mx(t["dit"]), dit_cfg=dict(t["dit"].config), settings=I.Settings(**E2E), text_dtype=mx.float32)
    c.dit_cfg["patch_size"] = tuple(c.dit_cfg["patch_size"])
    return c


def run_ours(t, prompt, images, height, width, seed, free=False, cache=False):
    c = components(t)
    dbg = {}
    with _quiet():
        img = I.generate(c, prompt, images=images, height=height, width=width, seed=seed, check_finite=True, debug=dbg,
                         free_transformer=free, modulation_cache=cache)
    dbg["components"] = c
    return img, dbg


@NEEDS_PROCESSOR
@pytest.mark.parametrize("workflow", ["text2image", "edit", "edit_canvas"])
def test_end_to_end_matches_diffusers(workflow):
    t = build()
    rng = np.random.default_rng(11)
    images, height, width = None, 40, 56
    prompt = "A risograph print of a lighthouse at dusk, two inks, no other text."
    if workflow.startswith("edit"):
        images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in ((40, 48), (56, 32))]
        prompt = "give the lighthouse in picture 1 the stripes from picture 2"
        height, width = (None, None) if workflow == "edit" else (48, 64)
    with torch.no_grad():
        want, want_embeds, want_tags, want_cond = run_diffusers(t, prompt, images, height, width, seed=7)
    got, dbg = run_ours(t, prompt, images, height, width, seed=7)
    print(workflow, got.shape, want.shape, f"max |diff| {np.abs(got - want).max():.2e}, std {want.std():.3f}")
    # the text and the pictures first, tightly: at this size a wholly different prompt moves the image by only
    # ~2e-3, and pictures held at t = 1.0 instead of 0.999 by less than 2e-5
    assert np.array_equal(dbg["text_token_tags"], want_tags.numpy())
    assert_close(dbg["prompt_embeds"], want_embeds, atol=1e-4, rtol=1e-4, what="prompt embeddings")
    if images:
        # the recipe rounds each picture's latents to float16, so an encoder difference of ~1e-6 can land one fp16
        # step (~2.4e-4 here) apart: a few elements may, every element may not (pictures held at t = 1.0 move all)
        ncv = dbg["layout"].num_condition_video_rows
        err = np.abs(np.array(dbg["initial_video_rows"][:ncv]) - want_cond.numpy())
        assert err.max() < 2e-3 and (err > 1e-5).mean() < 0.01, f"noised picture rows: max {err.max():.2e}, " \
                                                                 f"{(err > 1e-5).mean():.1%} of elements off"
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 2e-5
    assert want.std() > 0.01  # not a flat image that would match anything


@NEEDS_PROCESSOR
def test_single_use_components_free_their_weights_and_change_nothing():
    t = build()
    rng = np.random.default_rng(11)
    images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in ((40, 48), (56, 32))]
    keep, _ = run_ours(t, EDIT_PROMPT, images, None, None, seed=7)
    free, dbg = run_ours(t, EDIT_PROMPT, images, None, None, seed=7, free=True)
    c = dbg["components"]
    assert np.array_equal(keep, free)
    assert c.connector == {} and c.dit_params == {}
    assert not any(k.startswith(("encoder.", "quant_conv.")) for k in c.vae.p) and c.vae.enc == {}
    assert any(k.startswith("decoder.") for k in c.vae.p)


@NEEDS_PROCESSOR
@pytest.mark.parametrize("workflow", ["text2image", "edit"])
def test_modulation_cache_matches_diffusers_and_drops_adaln(workflow):
    """The cached AdaLN path against diffusers (which projects every call) and against our uncached path, with the
    projections dropped: the run's distinct timesteps are the 8 grid points, plus 0.999 for an edit's pictures."""
    t = build()
    rng = np.random.default_rng(11)
    images, height, width, prompt = None, 40, 56, "A risograph print of a lighthouse at dusk, two inks, no other text."
    if workflow == "edit":
        images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in ((40, 48), (56, 32))]
        height, width, prompt = None, None, EDIT_PROMPT
    with torch.no_grad():
        want, *_ = run_diffusers(t, prompt, images, height, width, seed=7)
    plain, _ = run_ours(t, prompt, images, height, width, seed=7)
    cached, dbg = run_ours(t, prompt, images, height, width, seed=7, free=True, cache=True)
    print(workflow, f"cached vs diffusers {np.abs(cached - want).max():.2e}, vs uncached {np.abs(cached - plain).max():.2e}")
    assert np.abs(cached - want).max() < 2e-5
    assert np.array_equal(cached, plain)  # each call's own tables: bit-identical
    assert len(dbg["modulation_timesteps"]) == (9 if workflow == "edit" else 8)
    assert not any("adaln_proj" in k or k.startswith("norm_out.linear") for k in dbg["components"].dit_params)


def test_text2image_sides_default_on_their_own():
    from diffusers.modular_pipelines.p_h3_image.before_denoise import PH3ImagePrepareLayoutStep
    from diffusers.modular_pipelines.p_h3_image.before_encoder import round_canvas

    defaults = {p.name: p.default for p in PH3ImagePrepareLayoutStep().inputs if p.name in ("height", "width")}
    assert defaults == {"height": I.DEFAULT_SIZE, "width": I.DEFAULT_SIZE}
    with _quiet():
        for h, w in ((None, None), (1000, None), (None, 1000), (2176, 3872)):
            want = round_canvas(h or defaults["height"], w or defaults["width"], 32)
            assert I.text2image_size(h, w, 32) == want


EDIT_PROMPT = "give the lighthouse in picture 1 the stripes from picture 2"


# -- the released configs --------------------------------------------------------------------------------------------


def test_release_configs_parse():
    from conftest import fixture

    root = fixture("P-H3-Image")
    import json

    vcfg = json.loads((root / "vae" / "config.json").read_text())
    assert VaeConfig.from_diffusers(vcfg).single_frame
    idx = json.loads((root / "modular_model_index.json").read_text())
    s = I.Settings.from_index(idx)
    assert s == I.Settings() and s.sample_sigmas == I.SAMPLE_SIGMAS
    conn = json.loads((root / "text_connector" / "config.json").read_text())
    te = json.loads((root / "text_encoder" / "config.json").read_text())
    assert tuple(conn["hidden_state_indices"]) == I.HIDDEN_STATE_INDICES
    assert conn["text_hidden_size"] == te["text_config"]["hidden_size"]
    assert max(conn["hidden_state_indices"]) == te["text_config"]["num_hidden_layers"]  # the last is the final norm
    t = {k: v for k, v in json.loads((root / "transformer" / "config.json").read_text()).items() if not k.startswith("_")}
    h3 = {k: v for k, v in json.loads((RELEASE / "transformer_ref" / "config.json").read_text()).items()
          if not k.startswith("_")}
    assert t == h3, "P-H3-Image's transformer is H3's architecture"
    assert conn["out_channels"] == t["text_dim"]
    names = set(json.loads((root / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"])
    h3_names = set(json.loads((RELEASE / "transformer_ref" / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"])
    assert names == h3_names
    sched = json.loads((root / "scheduler" / "scheduler_config.json").read_text())
    assert sched["shift"] == 12.0 and sched["shift_provided_sigmas"] is False  # the grid is used verbatim
