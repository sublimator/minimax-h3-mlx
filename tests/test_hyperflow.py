"""HyperFlow (8-step LoRA) on our MLX transformer, against torch.

The oracle is diffusers 0.42.1 + PEFT (the injection HyperFlow's loader performs: `add_adapter(LoraConfig(r, alpha,
dropout 0, bias none))`, adapter dtype = base layer dtype) plus two short torch transcriptions of HyperFlow's own code
at github.com/Video-Rebirth/hyperflow `1dd2f342` (read, not run): `TwoTimeEmbedder.forward` (embedder.py) and
`build_row_time_pairs` (schedule.py). The sigma grid goes through diffusers' own `MiniMaxH3Scheduler`.
"""

import copy
import json

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import assert_close, fixture, state_mx, to_mx, to_np
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from peft import LoraConfig, inject_adapter_in_model
from test_denoise import tiny_layout
from test_dit_mixed_precision import rel_err, to_mixed
from test_dit_tiny import TINY, packed_layout

from h3mlx import hyperflow, ops
from h3mlx.denoise import KEYFRAME_NOISE_AUG, Schedules, denoise, row_time_pairs, schedule_timesteps, shift_grid

RAW = hyperflow.DEFAULT_SIGMAS_8STEP
RANK = 4
BLOCK_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


# --- torch transcriptions of HyperFlow's code (the parts PEFT and diffusers don't already provide) ------------------


class TwoTimeEmbedder(torch.nn.Module):
    """embedder.py `TwoTimeEmbedder.forward`, endpoint set per forward."""

    def __init__(self, base, endpoint, time_proj, gate):
        super().__init__()
        self.base, self.endpoint, self.time_proj, self.gate = base, endpoint, time_proj, float(gate)
        self.r = None

    @property
    def linear_1(self):
        return self.base.linear_1

    def forward(self, sample):
        t_emb = self.base(sample)
        r_proj = self.time_proj(self.r.to(device=sample.device)).to(sample.dtype)
        r_emb = self.endpoint(r_proj)
        return t_emb + self.gate * (r_emb - t_emb)


def torch_row_time_pairs(video_indices, audio_indices, ncv, nca, n_text, vt, vr, at, ar, cvt, cat=1.0):
    """schedule.py `build_row_time_pairs`."""
    n = int(video_indices.numel() + audio_indices.numel() + n_text)
    t = torch.full((n,), float(vt), dtype=torch.float32)
    r = torch.full((n,), float(vr), dtype=torch.float32)
    t[video_indices[:ncv]] = float(cvt)
    r[video_indices[:ncv]] = float(cvt)
    t[audio_indices[nca:]] = float(at)
    r[audio_indices[nca:]] = float(ar)
    t[audio_indices[:nca]] = float(cat)
    r[audio_indices[:nca]] = float(cat)
    pairs, inv = torch.unique(torch.stack((t, r), dim=-1), dim=0, sorted=True, return_inverse=True)
    return pairs[:, 0].contiguous(), pairs[:, 1].contiguous(), inv


# --- a HyperFlow file for a tiny model, in the public key format, and both sides loaded from it ---------------------


def public_targets(cfg):
    out = [f"transformer_blocks.{i}.{n}" for i in range(cfg["num_layers"]) for n in BLOCK_TARGETS]
    out += [f"token_refiner.refiner_blocks.{j}.{n}" for j in range(cfg["num_refiner_layers"]) for n in BLOCK_TARGETS]
    return out + ["time_embedder.linear_1", "time_embedder.linear_2",
                  "endpoint_time_embedder.linear_1", "endpoint_time_embedder.linear_2"]


def make_file(model, cfg, gate=0.25, scale=0.3, seed=1):
    """Random LoRA A/B for every target (time embedder f32, the rest bf16, as the release), plus the header."""
    g = torch.Generator().manual_seed(seed)
    linears = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    tensors = {}
    for target in public_targets(cfg):
        lin = linears[target.replace("endpoint_time_embedder.", "time_embedder.")]
        dtype = torch.float32 if "time_embedder" in target else torch.bfloat16
        tensors[f"transformer.{target}.lora_A.weight"] = (torch.randn(RANK, lin.in_features, generator=g) * scale).to(dtype)
        tensors[f"transformer.{target}.lora_B.weight"] = (torch.randn(lin.out_features, RANK, generator=g) * scale).to(dtype)
    meta = {"hyperflow": "true", "hyperflow_version": "test", "hyperflow_gate": str(gate), "lora_alpha": str(RANK),
            "lora_rank": str(RANK), "base_model": "tiny", "hyperflow_sigmas": json.dumps(list(RAW)),
            "hyperflow_video_shift": "12.0", "hyperflow_audio_shift": "3.0",
            "compatible_subfolders": json.dumps(["transformer", "transformer_ref"])}
    return tensors, meta


def torch_hyperflow(model, tensors, meta):
    """PEFT injection as HyperFlow's `_load_into` does it, on a copy: endpoint embedder copied before any LoRA."""
    m = copy.deepcopy(model)
    endpoint = copy.deepcopy(m.time_embedder)
    cfg = dict(r=RANK, lora_alpha=float(meta["lora_alpha"]), lora_dropout=0.0, bias="none", init_lora_weights=True)
    main = sorted(t for t in public_targets(m.config) if not t.startswith("endpoint_"))
    m.add_adapter(LoraConfig(target_modules=main, **cfg), adapter_name="hyperflow")
    inject_adapter_in_model(LoraConfig(target_modules=["linear_1", "linear_2"], **cfg), endpoint, "hyperflow")
    params = dict(m.named_parameters())
    ep = dict(endpoint.named_parameters())
    with torch.no_grad():
        for key, value in tensors.items():
            module, ab = key[len("transformer."):].rsplit(".lora_", 1)
            ab = ab.split(".")[0]
            if module.startswith("endpoint_time_embedder."):
                dst = ep[f"{module[len('endpoint_time_embedder.'):]}.lora_{ab}.hyperflow.weight"]
            else:
                dst = params[f"{module}.lora_{ab}.hyperflow.weight"]
            dst.copy_(value.to(dst.dtype))
    m.time_embedder = TwoTimeEmbedder(m.time_embedder, endpoint, copy.deepcopy(m.time_proj), float(meta["hyperflow_gate"]))
    return m


def mlx_hyperflow(model, tensors, meta):
    params, cfg = hyperflow.attach(state_mx(model), dict(model.config), {k: to_mx(v) for k, v in tensors.items()},
                                   hyperflow.HyperFlowMeta.from_header(meta))
    return params, cfg


def randomised(cfg, seed):
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(**cfg).eval()
    with torch.no_grad():
        for name, prm in model.named_parameters():
            if "norm" in name and prm.ndim == 1:
                prm.normal_()
    return model


def pair_layout(gen):
    """packed_layout's shuffled rows, with distinct (t, r) pairs instead of distinct t."""
    lay = packed_layout(gen=gen)
    lay["timestep"] = torch.tensor([0.0, 0.37, 0.999])
    endpoint = torch.tensor([0.21, 0.52, 0.999])  # a conditioning pair keeps r = t
    return lay, endpoint


# --- the sigma grid --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_grid_matches_diffusers_scheduler(shift):
    base = torch.as_tensor(RAW, dtype=torch.float32)
    want = shift * base / (1.0 + (shift - 1.0) * base)  # schedule.py shift_sigmas == diffusers set_timesteps' formula
    sch = MiniMaxH3Scheduler(shift=shift)
    sch.set_timesteps(sigmas=want)
    got = shift_grid(RAW, shift)
    assert np.array_equal(got, want.numpy()) and np.array_equal(got, sch.sigmas.numpy())
    s = Schedules.from_grid(RAW, 12.0, 3.0)
    ts = s.video_timesteps if shift == 12.0 else s.audio_timesteps
    ends = s.video_endpoints if shift == 12.0 else s.audio_endpoints
    assert np.array_equal(ts, sch.timesteps.numpy()), "t_i = 1 - sigma_i, as the scheduler exposes it"
    assert np.array_equal(ends, (1.0 - want[1:]).numpy()), "r_i = 1 - sigma_{i+1} (endpoints_from_sigmas)"
    assert len(ts) == 8


@pytest.mark.parametrize("bad", [[1.0, 0.5, 0.6, 0.0], [1.2, 0.5, 0.0], [1.0, 0.5, 0.1], [1.0]])
def test_grid_rejects_invalid(bad):
    with pytest.raises(ValueError):
        shift_grid(bad, 12.0)


@pytest.mark.parametrize("vt,vr,at,ar", [(0.31, 0.47, 0.12, 0.2), (0.999, 0.9995, 0.5, 0.6), (0.0, 0.42, 0.0, 0.1),
                                          (0.5, 0.6, 0.5, 0.6)])
def test_row_time_pairs_match(vt, vr, at, ar):
    lay = tiny_layout()
    cvt = max(vt, KEYFRAME_NOISE_AUG)
    want = torch_row_time_pairs(torch.from_numpy(lay.video_indices), torch.from_numpy(lay.audio_indices),
                                lay.num_condition_video_rows, lay.num_condition_audio_rows, len(lay.text_indices),
                                vt, vr, at, ar, cvt)
    got = row_time_pairs(lay, vt, vr, at, ar, cvt, 1.0)
    for g, w, what in zip(got, want, ("timesteps", "endpoints", "indices")):
        assert np.array_equal(g, w.numpy()), what


# --- the forward -----------------------------------------------------------------------------------------------------


def test_forward_matches_peft():
    gen = torch.Generator().manual_seed(3)
    model = randomised(TINY, 2)
    tensors, meta = make_file(model, TINY)
    tensors = {k: v.float() for k, v in tensors.items()}  # f32 everywhere: a tight check of the wiring
    lay, endpoint = pair_layout(gen)
    video, audio, text = torch.randn(1, 12, 16, generator=gen), torch.randn(1, 7, 6, generator=gen), torch.randn(1, 5, 20, generator=gen)
    tm = torch_hyperflow(model, tensors, meta)
    tm.time_embedder.r = endpoint
    with torch.no_grad():
        want = tm(video, audio, text, **lay, return_dict=False)
        plain = model(video, audio, text, **lay, return_dict=False)
    params, cfg = mlx_hyperflow(model, tensors, meta)
    lay_mx = {k: to_mx(v) for k, v in lay.items()}
    got = ops.dit_forward(params, cfg, to_mx(video), to_mx(audio), to_mx(text), **lay_mx, endpoint=to_mx(endpoint))
    assert_close(got[0], want[0], atol=2e-4, rtol=2e-4, what="video velocity")
    assert_close(got[1], want[1], atol=2e-4, rtol=2e-4, what="audio velocity")
    # guards: the adapter and the endpoint must both move the output, or this test could not fail
    assert rel_err(want[0], plain[0]) > 0.05
    moved = ops.dit_forward(params, cfg, to_mx(video), to_mx(audio), to_mx(text), **lay_mx,
                            endpoint=to_mx(endpoint + 0.2))
    assert rel_err(moved[0], got[0]) > 1e-3


def test_mixed_precision_tracks_peft():
    """bf16 blocks with bf16 LoRA, f32 time embedders with f32 LoRA (the release): the MLX error against the f32 run
    stays within 1.5x of torch's own bf16 error, as for the base model."""
    gen = torch.Generator().manual_seed(5)
    cfg = dict(TINY, num_layers=4)
    model = randomised(cfg, 4)
    tensors, meta = make_file(model, cfg)
    lay, endpoint = pair_layout(gen)
    video, audio, text = torch.randn(1, 12, 16, generator=gen), torch.randn(1, 7, 6, generator=gen), torch.randn(1, 5, 20, generator=gen)
    truth_m = torch_hyperflow(model, {k: v.float() for k, v in tensors.items()}, meta)
    mixed = to_mixed(model)
    bf_m = torch_hyperflow(mixed, tensors, meta)
    assert bf_m.transformer_blocks[0].attn.to_q.lora_A["hyperflow"].weight.dtype == torch.bfloat16
    assert bf_m.time_embedder.base.linear_1.lora_A["hyperflow"].weight.dtype == torch.float32
    truth_m.time_embedder.r = bf_m.time_embedder.r = endpoint
    with torch.no_grad():
        truth = truth_m(video, audio, text, **lay, return_dict=False)
        t_bf = bf_m(video, audio, text, **lay, return_dict=False)
    params, mcfg = mlx_hyperflow(mixed, tensors, meta)
    assert params["transformer_blocks.0.attn.to_q.lora_A.weight"].dtype == mx.bfloat16
    assert params["endpoint_time_embedder.linear_1.lora_A.weight"].dtype == mx.float32
    lay_mx = {k: to_mx(v) for k, v in lay.items()}
    m_bf = ops.dit_forward(params, mcfg, to_mx(video), to_mx(audio), to_mx(text), **lay_mx, endpoint=to_mx(endpoint))
    for i, name in enumerate(("video", "audio")):
        e_torch, e_mlx = rel_err(t_bf[i], truth[i]), rel_err(m_bf[i], truth[i])
        print(f"{name}: torch bf16 rel err {e_torch:.4f}, mlx bf16 rel err {e_mlx:.4f}")
        assert e_mlx <= 1.5 * e_torch + 1e-3, f"{name}: mlx {e_mlx:.4f} vs torch {e_torch:.4f}"


# --- the loop --------------------------------------------------------------------------------------------------------


def test_denoise_two_time_matches():
    """8 calls on the file's grid: HyperFlowSetTimestepsStep's plan + HyperFlowLoopDenoiser's endpoint hand-off +
    the official Euler update (MiniMaxH3Scheduler.step on the installed grid), against ours."""
    model = randomised(TINY, 7)
    tensors, meta = make_file(model, TINY, scale=0.1)
    tensors = {k: v.float() for k, v in tensors.items()}
    tm = torch_hyperflow(model, tensors, meta)
    lay = tiny_layout()
    n_v, n_a, n_t = len(lay.video_indices), len(lay.audio_indices), len(lay.text_indices)
    video, audio, text = torch.randn(n_v, 16), torch.randn(n_a, 6), torch.randn(1, n_t, 20)

    base = torch.as_tensor(RAW, dtype=torch.float32)
    vs, as_ = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    vsig, asig = 12.0 * base / (1.0 + 11.0 * base), 3.0 * base / (1.0 + 2.0 * base)
    vs.set_timesteps(sigmas=vsig)
    as_.set_timesteps(sigmas=asig)
    v_end, a_end = 1.0 - vsig[1:], 1.0 - asig[1:]
    v, a = video.clone(), audio.clone()
    ncv, nca = lay.num_condition_video_rows, lay.num_condition_audio_rows
    t_lay = {k: torch.from_numpy(getattr(lay, k)) for k in ("token_tags", "video_indices", "audio_indices", "text_indices")}
    t_lay["position_ids"] = torch.from_numpy(lay.position_ids)
    with torch.no_grad():
        for i, (t, ta) in enumerate(zip(vs.timesteps, as_.timesteps)):
            ts, ends, inv = torch_row_time_pairs(t_lay["video_indices"], t_lay["audio_indices"], ncv, nca, n_t,
                                                 float(t), float(v_end[i]), float(ta), float(a_end[i]),
                                                 max(float(t), 0.999))
            tm.time_embedder.r = ends
            vp, ap = tm(v[None], a[None], text, ts, inv, **t_lay, return_dict=False)
            v[ncv:] = vs.step(vp[0, ncv:].float(), t, v[ncv:], return_dict=False)[0]
            a[nca:] = as_.step(ap[0, nca:].float(), as_.timesteps[i], a[nca:], return_dict=False)[0]

    params, cfg = mlx_hyperflow(model, tensors, meta)
    sched = hyperflow.HyperFlowMeta.from_header(meta).schedules()
    calls = []
    got_v, got_a = denoise(params, cfg, lay, to_mx(text), to_mx(video), to_mx(audio), sched,
                           on_step=lambda i, t: calls.append(i))
    assert calls == list(range(8))
    assert_close(got_v[:ncv], video[:ncv], atol=0, what="conditioning rows untouched")
    assert_close(got_v, v, atol=5e-4, rtol=5e-4, what="video rows after 8 calls")
    assert_close(got_a, a, atol=5e-4, rtol=5e-4, what="audio rows after 8 calls")


# --- the real file's header and the refusals -------------------------------------------------------------------------


def test_real_header_fits_transformer_ref():
    """The released v1.0 header (read by HTTP range, 2026-11-04): 316 modules, rank 256, gate 0.25, the 9-point grid,
    and every LoRA fits a Linear of transformer_ref at its real shape (model built on the meta device)."""
    header = json.loads(fixture("hyperflow_v1.0.header.json").read_text())
    md = header.pop("__metadata__")
    meta = hyperflow.HyperFlowMeta.from_header(md)
    shapes = {k: tuple(v["shape"]) for k, v in header.items()}
    assert (meta.gate, meta.lora_alpha, meta.lora_rank, meta.model_calls) == (0.25, 256.0, 256, 8)
    assert meta.sigmas == RAW and (meta.video_shift, meta.audio_shift) == (12.0, 3.0)
    assert "transformer_ref" in meta.compatible_subfolders
    cfg = json.loads(fixture("MiniMax-H3", "transformer_ref", "config.json").read_text())
    cfg = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if not k.startswith("_")}
    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel(**cfg)
    base = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    released = set(json.loads(fixture("transformer_ref.dtypes.json").read_text()))
    plan = hyperflow.plan_from_keys(shapes)
    assert len(plan) == 316
    assert {hyperflow._base_name(m) + ".weight" for m in plan} <= released, "every target is a released tensor"
    wired = hyperflow.wired_shapes(cfg)
    assert hyperflow.check_plan(plan, shapes, base, meta, wired=wired) == 256
    assert set(plan) == set(wired), "the released file covers exactly the linears ops.plinear serves"
    for m, shape in wired.items():  # the config-derived shapes check_file uses = the real model's
        assert base[hyperflow._base_name(m) + ".weight"] == shape, m
    assert hyperflow.check_file(meta, shapes, cfg) == 256
    assert not any("adaln_proj" in m for m in plan), "the adapter leaves the AdaLN projections alone"


def test_read_header_from_a_file(tmp_path):
    from safetensors.torch import save_file

    model = randomised(TINY, 0)
    tensors, md = make_file(model, TINY)
    path = tmp_path / "hf.safetensors"
    save_file(tensors, str(path), metadata=md)
    meta, shapes = hyperflow.read_header(path)
    assert meta.gate == 0.25 and len(shapes) == len(tensors)
    params, cfg, meta2 = hyperflow.load(path, state_mx(model), dict(model.config))
    assert cfg["hyperflow_gate"] == 0.25 and "endpoint_time_embedder.linear_2.bias" in params


def test_refusals():
    model = randomised(TINY, 0)
    tensors, md = make_file(model, TINY)
    t_mx = {k: to_mx(v) for k, v in tensors.items()}
    meta = hyperflow.HyperFlowMeta.from_header(md)
    base = state_mx(model)
    params, cfg = hyperflow.attach(base, dict(model.config), t_mx, meta)
    assert "endpoint_time_embedder.linear_1.weight" not in base, "attach does not modify its input"
    lay = {k: to_mx(v) for k, v in packed_layout().items()}
    args = (to_mx(torch.randn(1, 12, 16)), to_mx(torch.randn(1, 7, 6)), to_mx(torch.randn(1, 5, 20)))
    with pytest.raises(ValueError, match="endpoints"):
        ops.dit_forward(params, cfg, *args, **lay)  # two-time weights, no endpoints
    with pytest.raises(ValueError, match="endpoints"):
        ops.dit_forward(base, dict(model.config), *args, **lay, endpoint=mx.zeros(3))  # endpoints, no HyperFlow
    with pytest.raises(ValueError, match="one endpoint per timestep"):
        ops.dit_forward(params, cfg, *args, **lay, endpoint=mx.zeros(2))  # 3 distinct timesteps, 2 endpoints
    with pytest.raises(NotImplementedError):
        schedule_timesteps(tiny_layout(), meta.schedules())
    norm_lora = {**t_mx, "transformer.transformer_blocks.0.norm1.lora_A.weight": mx.zeros((4, 48)),
                 "transformer.transformer_blocks.0.norm1.lora_B.weight": mx.zeros((48, 4))}
    with pytest.raises(ValueError, match="not run through ops.plinear"):
        hyperflow.attach(base, dict(model.config), norm_lora, meta)
    with pytest.raises(ValueError, match="2-D"):  # without a wiring list, the 2-D check still stands
        hyperflow.check_plan(hyperflow.plan_from_keys(norm_lora), {k: v.shape for k, v in norm_lora.items()},
                             {k: v.shape for k, v in base.items()}, meta)
    shapes = {k: tuple(v.shape) for k, v in tensors.items()}
    no_rank = hyperflow.HyperFlowMeta.from_header({k: v for k, v in md.items() if k != "lora_rank"})
    assert no_rank.lora_rank is None and hyperflow.check_file(no_rank, shapes, dict(model.config)) == RANK
    with pytest.raises(ValueError, match="not run through ops.plinear"):  # a LoRA the port would silently skip
        hyperflow.attach(base, dict(model.config), {**t_mx, "transformer.proj_in.lora_A.weight": mx.zeros((4, 16)),
                                                    "transformer.proj_in.lora_B.weight": mx.zeros((48, 4))}, meta)
    with pytest.raises(ValueError, match="not run through ops.plinear"):
        hyperflow.check_file(meta, {**shapes, "transformer.transformer_blocks.0.adaln_proj.linear.lora_A.weight": (4, 24),
                                    "transformer.transformer_blocks.0.adaln_proj.linear.lora_B.weight": (864, 4)},
                             dict(model.config))
    with pytest.raises(ValueError, match="sigma grid"):  # an explicit empty grid is refused, not replaced by the default
        hyperflow.check_file(hyperflow.HyperFlowMeta.from_header({**md, "hyperflow_sigmas": "[]"}), shapes,
                             dict(model.config))
    no_grid = {k: v for k, v in md.items() if k != "hyperflow_sigmas"}
    assert hyperflow.HyperFlowMeta.from_header(no_grid).sigmas == RAW
    zero = {k: ((0, v[1]) if k.endswith("lora_A.weight") else (v[0], 0)) for k, v in shapes.items()}
    with pytest.raises(ValueError, match="rank must be positive"):
        hyperflow.check_file(no_rank, zero, dict(model.config))
    with pytest.raises(ValueError, match="no LoRA tensors"):
        hyperflow.check_file(meta, {}, dict(model.config))
    with pytest.raises(NotImplementedError):
        ops.ModulationCache(params, cfg, [0.0, 0.5])
    with pytest.raises(ValueError, match="already attached"):
        hyperflow.attach(params, cfg, t_mx, meta)
    with pytest.raises(ValueError, match="not a HyperFlow"):
        hyperflow.HyperFlowMeta.from_header({"format": "pt"})
    with pytest.raises(NotImplementedError, match="scaling"):
        hyperflow.attach(base, dict(model.config), t_mx, hyperflow.HyperFlowMeta.from_header({**md, "lora_alpha": "8"}))
    with pytest.raises(ValueError, match="lists"):
        hyperflow.attach(base, dict(model.config), t_mx, meta, subfolder="something_else")
    nowhere = {**t_mx, "transformer.nowhere.lora_A.weight": mx.zeros((4, 4)),
               "transformer.nowhere.lora_B.weight": mx.zeros((4, 4))}
    with pytest.raises(ValueError, match="not run through ops.plinear"):
        hyperflow.attach(base, dict(model.config), nowhere, meta)
    with pytest.raises(KeyError, match="no Linear"):  # without a wiring list
        hyperflow.check_plan(hyperflow.plan_from_keys(nowhere), {k: v.shape for k, v in nowhere.items()},
                             {k: v.shape for k, v in base.items()}, meta)
    with pytest.raises(ValueError, match="do not fit"):
        bad = dict(t_mx)
        bad["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"] = mx.zeros((RANK, 7))
        hyperflow.attach(base, dict(model.config), bad, meta)
    with pytest.warns(UserWarning, match="trained with shifts"):
        hyperflow.HyperFlowMeta.from_header({**md, "hyperflow_video_shift": "5.0"}).schedules()


def test_provenance_files_back_the_constants():
    """The published manifest's sha256 is the one the run script enforces, and HyperFlow's base revision has the same
    transformer_ref files as the release we run (both fetched from the HF API, 2026-11-04)."""
    manifest = json.loads(fixture("hyperflow.manifest-5bde94d7.json").read_text())
    name = manifest["default"]
    assert hyperflow.KNOWN_SHA256[name] == manifest["weights"][name]["sha256"]
    oids = json.loads(fixture("transformer_ref.lfs-42ed227-vs-83db0c0.json").read_text())
    ours, theirs = oids["revisions"].values()
    assert ours == theirs and len(ours) == 16
    header = json.loads(fixture("hyperflow_v1.0.header.json").read_text())
    assert header["__metadata__"]["base_model_revision"] in oids["revisions"]
