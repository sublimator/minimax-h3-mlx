"""The training cache (`h3mlx.cache`): its keys, its files, what a reader refuses, the two sizing rules for reference
pictures, and the encode on tiny encoders.

Weight-free. What a cache holds on real weights is the real encoders' output, which nothing here can vouch for: these
tests show that an entry is found only by the request that made it, that it comes back as it went in, and that with
tiny random encoders an entry is what the encoders give directly.
"""

import inspect
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch
from conftest import state_mx, to_np
from PIL import Image
from test_dit_tiny import TINY
from test_p_h3_image import E2E, NEEDS_PROCESSOR, PIXEL_MEAN, PIXEL_STD, VAE, make_connector, vae_pair  # noqa: F401

from h3mlx import cache, files, train
from h3mlx import image as I
from h3mlx.pipeline import Noise
from h3mlx.vae import encode_condition

SETTINGS = dict(reference_min_pixels=1024**2, reference_max_pixels=8 * 1024**2, vision_pixels=1_000_000, canvas_multiple=32)
SHA = {name: files.sha256_json(name) for name in ("picture", "first reference", "second reference", "another")}


def ask(**changes) -> dict:
    """A request for an edit with two references; `changes` replace arguments."""
    arguments = dict(
        picture=SHA["picture"], references=[SHA["first reference"], SHA["second reference"]],
        prompt="put the hat from picture 2 on the dog", height=256, width=192, latent_seed=7, latent_float16_round=False,
        reference_sizing="recipe", reference_seed=42, reference_float16_round=False, vae_dtype="float32",
        text_dtype="bfloat16", settings=dict(SETTINGS), encoders={name: f"identity of the {name}" for name in cache.ENCODERS})
    arguments.update(changes)
    return cache.request(**arguments)


def arrays_for(request: dict, seed=0, channels=4, text_rows=10, text_dim=20) -> dict:
    """Synthetic tensors of the right names and shapes for a request (a VAE of ratio 8 is assumed for the sizes)."""
    g = torch.Generator().manual_seed(seed)
    h, w = request["target"]["height"] // 8, request["target"]["width"] // 8
    out = {"latents": mx.array(torch.randn(1, 1, h, w, channels, generator=g).numpy())}
    for i in range(len(request["inputs"]["references"])):
        out[f"condition_latents.{i}"] = mx.array(torch.randn(1, 1, 4 + 2 * i, 6, channels, generator=g).numpy())
    out["text"] = mx.array(torch.randn(1, text_rows, text_dim, generator=g).numpy()).astype(cache.DTYPES[request["text"]["dtype"]])
    out["text_token_tags"] = mx.array(np.array([1] * (text_rows - 3) + [0] * 3, dtype=np.int64))
    return out


def same(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(to_np(a.astype(mx.float32)), to_np(b.astype(mx.float32)))


def listing(directory) -> list[str]:
    return sorted(p.name for p in Path(directory).iterdir())


# --- round trip --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("references", [0, 2])
@pytest.mark.parametrize("text_dtype", ["bfloat16", "float32"])
def test_an_entry_comes_back_as_it_went_in(tmp_path, references, text_dtype):
    """Two files per entry, named by the request's key. Every array comes back with its bits, its dtype and its
    shape, the tags as integers, and the record carries the whole request. The entry feeds `train.build_example`."""
    request = ask(references=[SHA["first reference"], SHA["second reference"]][:references], text_dtype=text_dtype,
                  height=64, width=48, settings=dict(SETTINGS, canvas_multiple=16))
    tensors = arrays_for(request)
    name = cache.write(tmp_path, request, tensors)
    assert name == cache.key(request) and len(name) == 64
    assert listing(tmp_path) == [f"{name}.json", f"{name}.safetensors"]
    entry = cache.read(tmp_path, request)
    assert same(entry.latents, tensors["latents"]) and same(entry.text, tensors["text"])
    assert entry.text.dtype == cache.DTYPES[text_dtype] and entry.latents.dtype == mx.float32
    assert entry.text_token_tags.dtype == np.int64 and entry.text_token_tags.tolist() == [1] * 7 + [0] * 3
    assert len(entry.condition_latents) == references
    assert all(same(got, tensors[f"condition_latents.{i}"]) for i, got in enumerate(entry.condition_latents))
    assert entry.record["request"] == request == json.loads((tmp_path / f"{name}.json").read_text())["request"]
    header = files.safetensors_header(tmp_path / f"{name}.safetensors")
    assert set(header) - {"__metadata__"} == set(tensors), "plain safetensors: a header and the arrays"

    example = train.build_example(TINY, entry.text_token_tags, entry.latents, 0.5, Noise(3), condition_t=0.999,
                                  condition_latents=entry.condition_latents)
    direct = train.build_example(TINY, np.array(tensors["text_token_tags"]), tensors["latents"], 0.5, Noise(3),
                                 condition_t=0.999, condition_latents=[tensors[f"condition_latents.{i}"] for i in range(references)])
    assert same(example.video_rows, direct.video_rows) and same(example.target, direct.target)


# --- a changed input, choice or version is another key -----------------------------------------------------------------


CHANGES = {
    "the picture": dict(picture=SHA["another"]),
    "a reference": dict(references=[SHA["first reference"], SHA["another"]]),
    "the references' order": dict(references=[SHA["second reference"], SHA["first reference"]]),
    "one reference fewer": dict(references=[SHA["first reference"]]),
    "the prompt": dict(prompt="put the hat from picture 2 on the cat"),
    "the height": dict(height=288),
    "the width": dict(width=224),
    "the latent seed": dict(latent_seed=8),
    "the latent float16 rounding": dict(latent_float16_round=True),
    "the reference sizing": dict(reference_sizing="pipeline"),
    "the reference seed": dict(reference_seed=43),
    "the reference float16 rounding": dict(reference_float16_round=True),
    "the VAE's dtype": dict(vae_dtype="bfloat16"),
    "the text dtype": dict(text_dtype="float32"),
    "the reference bounds": dict(settings=dict(SETTINGS, reference_max_pixels=4 * 1024**2)),
    "the vision size": dict(settings=dict(SETTINGS, vision_pixels=500_000)),
    "the canvas multiple": dict(settings=dict(SETTINGS, canvas_multiple=16)),
    **{f"the {name}'s identity": dict(encoders={n: ("another" if n == name else f"identity of the {n}") for n in cache.ENCODERS})
       for name in cache.ENCODERS},
}


@pytest.mark.parametrize("what", list(CHANGES))
def test_a_changed_input_or_choice_is_not_served_the_old_entry(tmp_path, what):
    """An entry is written for one request. Asked with one thing changed, the cache has nothing: the key is another
    one, so the old entry is not found, let alone served. Pointed at the old entry's record on purpose, the reader
    refuses it and says what differs."""
    request = ask()
    cache.write(tmp_path, request, arrays_for(request))
    changed = ask(**CHANGES[what])
    assert cache.key(changed) != cache.key(request)
    assert cache.read(tmp_path, changed) is None
    assert cache.read(tmp_path, request) is not None, "the entry is still there for its own request"
    with pytest.raises(cache.CacheMismatch, match="is not the entry asked for: .*: asked "):
        cache.load(tmp_path / f"{cache.key(request)}.json", changed)


def test_a_changed_code_or_library_version_is_another_key(tmp_path, monkeypatch):
    """The request carries a digest of this package's source and the versions of the libraries the encode runs
    through. After a change to either, yesterday's entries are not found."""
    request = ask()
    cache.write(tmp_path, request, arrays_for(request))
    assert request["versions"]["h3mlx"] == cache.code_digest() and request["versions"]["packages"]["mlx"]
    assert set(request["versions"]["packages"]) == set(cache.PACKAGES)

    with monkeypatch.context() as patch:
        patch.setattr(cache, "code_digest", lambda package=None: "0" * 64)
        changed = ask()
        assert cache.key(changed) != cache.key(request) and cache.read(tmp_path, changed) is None
        with pytest.raises(cache.CacheMismatch, match="versions.h3mlx: asked"):
            cache.load(tmp_path / f"{cache.key(request)}.json", changed)
    for package in cache.PACKAGES:
        with monkeypatch.context() as patch:
            real = cache.metadata.version
            patch.setattr(cache.metadata, "version", lambda name, package=package: "0.0.0" if name == package else real(name))
            assert cache.read(tmp_path, ask()) is None, package
    assert cache.read(tmp_path, ask()) is not None


def test_an_example_without_references_ignores_the_reference_choices():
    """The reference choices are not part of a text-to-image request: changing them leaves its key alone, so such
    entries are not encoded again for a setting they never used."""
    plain = ask(references=[])
    assert plain["references"] is None and plain["text"]["workflow"] == "text2image" and ask()["text"]["workflow"] == "edit"
    for change in (dict(reference_sizing="pipeline"), dict(reference_seed=1), dict(reference_float16_round=True),
                   dict(settings=dict(SETTINGS, vision_pixels=500_000)),
                   dict(reference_sizing=None, reference_seed=None, reference_float16_round=None)):
        assert cache.key(ask(references=[], **change)) == cache.key(plain), change
    for change in (dict(reference_sizing=None), dict(reference_seed=None), dict(reference_float16_round=None)):
        with pytest.raises(ValueError, match="reference"):  # with references they are needed
            ask(**change)
    assert cache.key(ask(references=[], latent_seed=8)) != cache.key(plain)


def test_code_digest_follows_the_source(tmp_path):
    """The digest covers every `.py` file of the package, by name and by content."""
    package = Path(cache.__file__).parent
    copy = tmp_path / "h3mlx"
    shutil.copytree(package, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert cache.code_digest(copy) == cache.code_digest() == cache.code_digest(package)
    sources = sorted(p.name for p in copy.glob("*.py"))
    assert {"cache.py", "vae.py", "image.py", "train.py", "ops.py"} <= set(sources)
    for name in sources:  # one byte more in any file
        original = (copy / name).read_bytes()
        (copy / name).write_bytes(original + b"\n")
        assert cache.code_digest(copy) != cache.code_digest(), name
        (copy / name).write_bytes(original)
    assert cache.code_digest(copy) == cache.code_digest()
    (copy / "train.py").rename(copy / "trains.py")
    assert cache.code_digest(copy) != cache.code_digest()
    (copy / "trains.py").rename(copy / "train.py")
    (copy / "new.py").write_text("")
    assert cache.code_digest(copy) != cache.code_digest()
    with pytest.raises(ValueError, match="no source files"):
        cache.code_digest(tmp_path / "nowhere")


def test_identity_reads_no_weights(tmp_path):
    """A component is identified by its small files' contents and its weight files' names and sizes. The weights are
    not opened: here they cannot be. The stated limit: other bytes in a weight file of the same size are not seen."""
    component = tmp_path / "vae"
    component.mkdir()
    (component / "config.json").write_text('{"latent_channels": 4}')
    weights = component / "diffusion_pytorch_model.safetensors"
    weights.write_bytes(b"\0" * 1000)
    weights.chmod(0)
    try:
        with pytest.raises(PermissionError):
            weights.read_bytes()
        first = cache.identity(component)
        assert len(first) == 64 and cache.identity(component) == first
    finally:
        weights.chmod(0o644)
    weights.write_bytes(b"\1" * 1000)
    assert cache.identity(component) == first, "same name and size: not noticed, as documented"
    weights.write_bytes(b"\0" * 1001)
    assert cache.identity(component) != first
    weights.write_bytes(b"\0" * 1000)
    weights.rename(component / "model.safetensors")
    assert cache.identity(component) != first
    (component / "model.safetensors").rename(weights)
    assert cache.identity(component) == first
    (component / "config.json").write_text('{"latent_channels": 8}')
    assert cache.identity(component) != first
    with pytest.raises(ValueError, match="nothing to identify"):
        cache.identity(tmp_path / "missing")


# --- what a reader refuses ---------------------------------------------------------------------------------------------


def two_entries(tmp_path):
    first, second = ask(), ask(latent_float16_round=True)
    cache.write(tmp_path, first, arrays_for(first, seed=1))
    cache.write(tmp_path, second, arrays_for(second, seed=2))
    return first, second


def test_an_entry_put_under_another_requests_name_is_refused(tmp_path):
    """The stale-entry case made by hand: one entry's two files copied to the names another request would look up.
    The reader compares the record's request with what it was asked, and names the choice that differs."""
    first, second = two_entries(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    for suffix in (".json", ".safetensors"):
        shutil.copy(tmp_path / f"{cache.key(first)}{suffix}", other / f"{cache.key(second)}{suffix}")
    with pytest.raises(cache.CacheMismatch, match=r"target\.float16_round: asked True, the entry has False"):
        cache.read(other, second)
    # and with the record's request edited to match, the key the record states gives it away
    record = json.loads((other / f"{cache.key(second)}.json").read_text())
    record["request"] = second
    (other / f"{cache.key(second)}.json").write_text(json.dumps(record))
    with pytest.raises(cache.CacheMismatch, match="the record's key is not its request's"):
        cache.read(other, second)


def test_differences_names_what_differs():
    """What the reader's refusal is made from: dotted paths, both values, keys on one side only, and a 1 that is not
    a True (JSON keeps the two apart, and so does the comparison)."""
    request = ask()
    assert cache.differences(request, json.loads(json.dumps(request))) == []
    assert cache.differences({"a": {"b": 1, "c": [1, 2]}}, {"a": {"b": 2, "c": [1, 2]}}) == ["a.b: asked 1, the entry has 2"]
    assert cache.differences({"a": True}, {"a": 1}) == ["a: asked True, the entry has 1"]
    assert cache.differences({"a": None}, {"a": False}) == ["a: asked None, the entry has False"]
    assert cache.differences({"a": 1, "b": 2}, {"a": 1}) == ["b: missing from the entry"]
    assert cache.differences({"a": 1}, {"a": 1, "b": 2}) == ["b: not asked for"]
    assert cache.differences({"a": [1, 2]}, {"a": [2, 1]}) == ["a: asked [1, 2], the entry has [2, 1]"]
    assert cache.differences(request, None)[0].startswith(": asked ")


def damage(tmp_path, what: str, first: dict, second: dict) -> None:
    record_path, tensors_path = tmp_path / f"{cache.key(first)}.json", tmp_path / f"{cache.key(first)}.safetensors"
    record = json.loads(record_path.read_text())
    if what == "another entry's tensors":
        shutil.copy(tmp_path / f"{cache.key(second)}.safetensors", tensors_path)
    elif what == "truncated tensors":
        tensors_path.write_bytes(tensors_path.read_bytes()[:-16])
    elif what == "no tensors":
        tensors_path.unlink()
    elif what == "a record that is not JSON":
        record_path.write_text(record_path.read_text()[:-20])
    elif what == "an empty record":
        record_path.write_text("")
    elif what == "another format":
        record_path.write_text(json.dumps({**record, "format": "h3mlx-training-example-0"}))
    elif what == "a record that is a list":
        record_path.write_text("[]")
    elif what == "a record without its tensors' description":
        record_path.write_text(json.dumps({k: v for k, v in record.items() if k != "tensors"}))
    elif what in ("a pickle with the record's blessing", "an npz with the record's blessing"):
        if what.startswith("a pickle"):
            import pickle

            tensors_path.write_bytes(pickle.dumps({"latents": np.zeros(3)}))
        else:
            np.savez(tensors_path.with_suffix(".npz"), latents=np.zeros(3))
            tensors_path.with_suffix(".npz").rename(tensors_path)
        record["tensors"]["sha256"] = files.sha256_file(tensors_path)
        record_path.write_text(json.dumps(record))
    elif what == "arrays the record does not list":
        arrays = dict(mx.load(str(tensors_path), format="safetensors"))
        arrays["text"] = arrays["text"].astype(mx.float32)
        mx.save_safetensors(str(tensors_path), arrays)
        record["tensors"]["sha256"] = files.sha256_file(tensors_path)
        record_path.write_text(json.dumps(record))
    else:
        raise AssertionError(what)


@pytest.mark.parametrize("what,message", [
    ("another entry's tensors", "sha256 differs"),
    ("truncated tensors", "sha256 differs"),
    ("no tensors", "are missing"),
    ("a record that is not JSON", "the record cannot be read"),
    ("an empty record", "the record cannot be read"),
    ("another format", "not a h3mlx-training-example-1 record"),
    ("a record that is a list", "not a h3mlx-training-example-1 record"),
    ("a record without its tensors' description", "the record's key is not its request's"),
    ("a pickle with the record's blessing", "cannot be read as safetensors"),
    ("an npz with the record's blessing", "cannot be read as safetensors"),
    ("arrays the record does not list", "its arrays are not the ones its record lists"),
])
def test_a_damaged_entry_is_an_error_not_a_miss(tmp_path, what, message):
    """An entry that is there and wrong is refused with a `CacheMismatch`: it is never returned, and never taken for
    a miss, which would let a trainer overwrite the evidence. The tensors are read as safetensors and as nothing else:
    a pickle or an npz under the entry's name is refused even when the record's digest has been made to agree."""
    first, second = two_entries(tmp_path)
    damage(tmp_path, what, first, second)
    with pytest.raises(cache.CacheMismatch, match=message):
        cache.read(tmp_path, first)
    assert cache.read(tmp_path, second) is not None, "the other entry is untouched"


def test_no_pickle_in_the_module():
    source = inspect.getsource(cache) + inspect.getsource(files)
    assert "import pickle" not in source and "pickle." not in source and "np.load" not in source
    assert 'format="safetensors"' in inspect.getsource(cache.load)


# --- writing -----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("fails", ["saving the tensors", "renaming the tensors", "writing the record", "renaming the record"])
def test_a_write_that_fails_leaves_no_entry(tmp_path, monkeypatch, fails):
    """Each file is written under a temporary name and renamed, the record last. Whatever step fails, the directory
    is left without an entry and without a temporary file, and a later write succeeds."""
    request = ask()
    tensors = arrays_for(request)
    real_save, real_replace, real_text = mx.save_safetensors, os.replace, Path.write_text
    renames = []

    def save(path, arrays, *a, **kw):
        real_save(path, arrays, *a, **kw)
        raise OSError("disk full")

    def replace(src, dst):
        renames.append(Path(dst).suffix)
        if (fails, Path(dst).suffix) in (("renaming the tensors", ".safetensors"), ("renaming the record", ".json")):
            raise OSError("no rename today")
        return real_replace(src, dst)

    def write_text(self, text, *a, **kw):
        real_text(self, text[: len(text) // 2], *a, **kw)
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        if fails == "saving the tensors":
            patch.setattr(mx, "save_safetensors", save)
        elif fails == "writing the record":
            patch.setattr(Path, "write_text", write_text)
        else:
            patch.setattr(os, "replace", replace)
        with pytest.raises(OSError):
            cache.write(tmp_path, request, tensors)
    assert listing(tmp_path) == [], "nothing left behind"
    assert cache.read(tmp_path, request) is None
    cache.write(tmp_path, request, tensors)
    assert same(cache.read(tmp_path, request).latents, tensors["latents"])


def test_files_a_killed_writer_left_are_ignored(tmp_path):
    """A writer killed mid-write leaves files under temporary names, or the tensors without their record. Neither is
    an entry: the reader reports a miss, and the next write replaces the orphan."""
    request = ask()
    tensors = arrays_for(request)
    name = cache.key(request)
    (tmp_path / f"{files.TMP}123-abcd1234-{name}.safetensors").write_bytes(b"half a file")
    (tmp_path / f"{files.TMP}123-abcd1234-{name}.json").write_text('{"format": "h3mlx-training-ex')
    (tmp_path / f"{name}.safetensors").write_bytes(b"tensors whose record never landed")
    assert cache.read(tmp_path, request) is None
    cache.write(tmp_path, request, tensors)
    assert same(cache.read(tmp_path, request).text, tensors["text"])


def test_an_entry_is_written_once(tmp_path):
    request = ask()
    first = arrays_for(request, seed=1)
    cache.write(tmp_path, request, first)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError, match="already has an entry"):
        cache.write(tmp_path, request, arrays_for(request, seed=2))
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert same(cache.read(tmp_path, request).latents, first["latents"])


def test_atomic_write_keeps_the_old_file_when_the_new_one_fails(tmp_path):
    path = tmp_path / "run.json"
    files.atomic_json(path, {"step": 1})

    def half(tmp):
        tmp.write_text('{"step": ')
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        files.atomic_write(path, half)
    assert json.loads(path.read_text()) == {"step": 1} and listing(tmp_path) == ["run.json"]
    with pytest.raises(ValueError):
        files.atomic_json(path, {"loss": float("nan")})
    assert json.loads(path.read_text()) == {"step": 1} and listing(tmp_path) == ["run.json"]
    files.atomic_json(path, {"step": 2})
    assert json.loads(path.read_text()) == {"step": 2}


@pytest.mark.parametrize("what,message", [
    ("a name missing", "has exactly"), ("a name too many", "has exactly"), ("bf16 latents", "must be float32"),
    ("latents without their axes", "must be float32"), ("latents of another shape", "are not a picture of"),
    ("text in another dtype", "text must be bfloat16"), ("tags of another length", "text_token_tags must be int64"),
    ("a tag of 2", "text_token_tags must be int64"), ("float tags", "text_token_tags must be int64"),
    ("a NaN in the latents", "latents holds values that are not finite"),
    ("an infinity in the text", "text holds values that are not finite"),
    ("a NaN in a reference", r"condition_latents\.1 holds values that are not finite"),
])
def test_tensors_that_are_not_an_entry_are_not_written(tmp_path, what, message):
    request = ask()
    t = arrays_for(request)
    if what == "a name missing":
        del t["condition_latents.1"]
    elif what == "a name too many":
        t["condition_latents.2"] = t["condition_latents.1"]
    elif what == "bf16 latents":
        t["latents"] = t["latents"].astype(mx.bfloat16)
    elif what == "latents without their axes":
        t["latents"] = t["latents"][0, 0]
    elif what == "latents of another shape":
        t["latents"] = t["latents"][:, :, :, :-1]
    elif what == "text in another dtype":
        t["text"] = t["text"].astype(mx.float32)
    elif what == "tags of another length":
        t["text_token_tags"] = t["text_token_tags"][:-1]
    elif what == "a tag of 2":
        t["text_token_tags"] = mx.array(np.array([2] * 10, dtype=np.int64))
    elif what == "float tags":
        t["text_token_tags"] = t["text_token_tags"].astype(mx.float32)
    else:
        name = {"a NaN in the latents": "latents", "an infinity in the text": "text", "a NaN in a reference": "condition_latents.1"}[what]
        bad = np.array(t[name].astype(mx.float32))
        bad.flat[3] = np.inf if "infinity" in what else np.nan
        t[name] = mx.array(bad).astype(t[name].dtype)
    with pytest.raises(ValueError, match=message):
        cache.write(tmp_path, request, t)
    assert not tmp_path.exists() or listing(tmp_path) == []


# --- the request -------------------------------------------------------------------------------------------------------


def test_a_request_has_no_defaults():
    parameters = inspect.signature(cache.request).parameters
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty for p in parameters.values())
    assert len(parameters) == 14
    complete = dict(picture=SHA["picture"], references=[], prompt="a dog", height=64, width=64, latent_seed=0,
                    latent_float16_round=False, reference_sizing="recipe", reference_seed=42, reference_float16_round=False,
                    vae_dtype="float32", text_dtype="bfloat16", settings=SETTINGS, encoders={n: "x" for n in cache.ENCODERS})
    cache.request(**complete)
    for name in parameters:
        with pytest.raises(TypeError, match=name):
            cache.request(**{k: v for k, v in complete.items() if k != name})


@pytest.mark.parametrize("change,message", [
    (dict(picture="a.png"), "picture must be a sha256"), (dict(references=["a.png"]), r"references\[0\] must be a sha256"),
    (dict(prompt="  "), "prompt must be a string with something in it"), (dict(prompt=None), "prompt must be a string"),
    (dict(height=250), "height must be a positive multiple of 32"), (dict(width=0), "width must be a positive multiple"),
    (dict(height=256.0), "height must be a positive multiple"), (dict(latent_seed=-1), "latent_seed must be an integer seed"),
    (dict(latent_seed=1.0), "latent_seed must be an integer seed"), (dict(reference_seed=True), "reference_seed must be an integer seed"),
    (dict(latent_float16_round=1), "latent_float16_round must be True or False"),
    (dict(reference_float16_round=None), "reference_float16_round must be True or False"),
    (dict(reference_sizing="diffusers"), "reference sizing is one of"), (dict(vae_dtype="float64"), "vae_dtype is one of"),
    (dict(text_dtype="bf16"), "text_dtype is one of"), (dict(settings={k: v for k, v in SETTINGS.items() if k != "vision_pixels"}), "settings must hold"),
    (dict(settings=dict(SETTINGS, extra=1)), "settings must hold"), (dict(settings=dict(SETTINGS, vision_pixels=1e6)), "settings must hold"),
    (dict(encoders={"vae": "x"}), "encoders must name an identity"),
    (dict(encoders={n: "" for n in cache.ENCODERS}), "encoders must name an identity"),
])
def test_request_refusals(change, message):
    with pytest.raises(ValueError, match=message):
        ask(**change)


# --- sizing a reference ------------------------------------------------------------------------------------------------


def sizes(sizing, width, height, target=(1024, 1024), **settings):
    s = dict(min_pixels=1024**2, max_pixels=8 * 1024**2, vision_pixels=1_000_000, multiple=32)
    s.update(settings)
    return cache.reference_sizes(sizing, width, height, target_height=target[0], target_width=target[1], **s)


def test_pipeline_sizing_is_diffusers():
    """`"pipeline"` is the rule diffusers' edit pipeline applies at sampling, from its own two functions."""
    from diffusers.modular_pipelines.p_h3_image import before_encoder as B

    rng = np.random.default_rng(3)
    for _ in range(400):
        w, h = (int(x) for x in rng.integers(17, 6000, 2))
        target = tuple(int(x) * 32 for x in rng.integers(4, 90, 2))
        assert sizes("pipeline", w, h, target) == (B._size_in_area_range(w, h, 1024**2, 8 * 1024**2, 32),
                                                   B._size_at_area(w, h, 1_000_000, 32))
    assert sizes("pipeline", 4000, 3000, (64, 64)) == sizes("pipeline", 4000, 3000, (2048, 2048)), "the target plays no part"


def test_recipe_sizing_scales_down_to_the_target_and_never_up():
    """`"recipe"` has no runnable original here: the trainer was read, not run. These are its stated properties and
    values worked by hand from the rule (scale by min(1, sqrt(target area / own area)), round each side to 32)."""
    # 4000x3000 into a 1024x1024 target: scale 0.2956 -> 1182.4 x 886.8 -> 37 and 28 thirty-twos
    assert sizes("recipe", 4000, 3000) == ((1184, 896), (1152, 864))
    # the vision size: scale sqrt(1e6 / 12e6) = 0.2887 -> 1154.7 x 866.0 -> 36 and 27 thirty-twos
    assert cache.scale_down_to_area(4000, 3000, 1_000_000, 32) == (1152, 864)
    # smaller than the target: kept at its own size, on the grid; the pipeline would bring it up to its lower bound
    assert sizes("recipe", 640, 480) == ((640, 480), (640, 480))
    assert sizes("pipeline", 640, 480)[0] == (1184, 896)
    # the target's own shape and larger: exactly the target
    assert sizes("recipe", 4096, 2048, (512, 1024))[0] == (1024, 512)
    # a tie rounds to even, as Python's round does on both sides: 1040 / 32 = 32.5 -> 32
    assert cache.scale_down_to_area(1040, 1040, 10**9, 32) == (1024, 1024)
    assert cache.scale_down_to_area(5, 3, 10**9, 32) == (32, 32), "never below one multiple"
    assert cache.scale_down_to_area(1001, 999, 1000 * (1_000_999 // 1000), 8) == cache.scale_down_to_area(1001, 999, 1_000_000, 8)

    rng = np.random.default_rng(4)
    for _ in range(400):
        w, h = (int(x) for x in rng.integers(40, 6000, 2))
        target = tuple(int(x) * 32 for x in rng.integers(4, 90, 2))
        (vw, vh), (qw, qh) = sizes("recipe", w, h, target)
        assert vw % 32 == vh % 32 == qw % 32 == qh % 32 == 0
        assert vw <= w + 16 and vh <= h + 16 and qw <= w + 16 and qh <= h + 16, "never scaled up, but for the rounding"
        slack = 32 * (vw + vh) + 32 * 32  # what rounding both sides can add to the area
        assert vw * vh <= max(target[0] * target[1], 0) + slack or (vw <= w + 16 and w * h <= target[0] * target[1])
        if w * h > target[0] * target[1] and min(vw, vh) >= 256:
            assert abs(vw / vh - w / h) < 0.2 * w / h and abs(vw * vh / (target[0] * target[1]) - 1) < 0.25
    with pytest.raises(ValueError, match="reference sizing is one of"):
        sizes("comfy", 640, 480)


# --- the encode, on tiny encoders --------------------------------------------------------------------------------------


def tiny_components(vae) -> SimpleNamespace:
    """A tiny VAE with a stub text side: a tokenizer that maps characters to ids and a fixed 'text encoder'. Enough
    for a text-to-image entry; an edit needs the release's processor (the last test)."""
    connector = state_mx(make_connector((1, 3, 4), 16, 20, seed=9))

    def tokenizer(prompt, add_special_tokens=False):
        return {"input_ids": [ord(ch) % 97 for ch in prompt]}

    def encode_states(ids, vision):
        g = torch.Generator().manual_seed(sum(ids))
        return [mx.array(torch.randn(1, len(ids), 16, generator=g).numpy()) for _ in range(3)]

    return SimpleNamespace(canvas_multiple=8, vae=vae, vae_latents_mean=VAE["latents_mean"], vae_latents_std=VAE["latents_std"],
                           tokenizer=tokenizer, encode_states=encode_states, connector=connector)


TINY_SETTINGS = dict(reference_min_pixels=32 * 32, reference_max_pixels=64 * 64, vision_pixels=48 * 48, canvas_multiple=8)


def direct_latents(vae, picture, seed, float16_round):
    """The posterior sample of one picture through `vae.encode_condition`, with torch's draws for `seed`."""
    def noise(shape_cl):
        b, f, h, w, c = shape_cl
        draw = torch.randn((b, c, f, h, w), generator=torch.Generator().manual_seed(seed))
        return mx.array(draw.permute(0, 2, 3, 4, 1).contiguous().numpy())

    pixels = mx.array(np.asarray(picture.convert("RGB"))[None, None])
    return encode_condition(vae, pixels, PIXEL_MEAN, PIXEL_STD, VAE["latents_mean"], VAE["latents_std"], noise,
                            float16_round=float16_round)


def test_a_text_to_image_entry_is_the_direct_encode(tmp_path, vae_pair):  # noqa: F811
    """The picture's latents in an entry are `vae.encode_condition`'s for the request's seed and rounding, bit for
    bit; the text rows are the connector's on the tokenizer's ids. The seed and the rounding both change the latents,
    a picture of the stated shape at another size is resized first, and one of another shape is refused."""
    _, vae = vae_pair
    c = tiny_components(vae)
    rng = np.random.default_rng(5)
    picture = Image.fromarray(rng.integers(0, 256, (48, 40, 3), dtype=np.uint8))
    digest = files.sha256_json("the picture")

    def entry_for(**changes):
        arguments = dict(picture=digest, references=[], prompt="a dog", height=48, width=40, latent_seed=7,
                         latent_float16_round=False, reference_sizing="recipe", reference_seed=42,
                         reference_float16_round=False, vae_dtype="float32", text_dtype="float32", settings=TINY_SETTINGS,
                         encoders={n: "tiny" for n in cache.ENCODERS})
        arguments.update(changes)
        request = cache.request(**arguments)
        return request, cache.encode(c, request, picture, [])

    request, tensors = entry_for()
    assert sorted(tensors) == ["latents", "text", "text_token_tags"]
    assert same(tensors["latents"], direct_latents(vae, picture, 7, False)) and tensors["latents"].shape == (1, 1, 12, 10, 4)
    ids = c.tokenizer("a dog")["input_ids"]
    assert same(tensors["text"], I.connect(c.connector, c.encode_states(ids, {}), mx.float32))
    assert to_np(tensors["text_token_tags"]).tolist() == [1] * len(ids)

    assert same(entry_for(latent_float16_round=True)[1]["latents"], direct_latents(vae, picture, 7, True))
    assert same(entry_for(latent_seed=8)[1]["latents"], direct_latents(vae, picture, 8, False))
    assert not same(direct_latents(vae, picture, 8, False), tensors["latents"])
    assert not same(direct_latents(vae, picture, 7, True), tensors["latents"])
    assert entry_for(text_dtype="bfloat16")[1]["text"].dtype == mx.bfloat16

    # written and read back, the entry builds the example the direct encode builds
    cache.write(tmp_path, request, tensors)
    entry = cache.read(tmp_path, request)
    cfg = dict(TINY, text_dim=20)
    got = train.build_example(cfg, entry.text_token_tags, entry.latents, 0.5, Noise(1), condition_t=0.999)
    want = train.build_example(cfg, np.array([1] * len(ids)), direct_latents(vae, picture, 7, False), 0.5, Noise(1), condition_t=0.999)
    assert same(got.video_rows, want.video_rows) and same(got.target, want.target) and same(got.clean_rows, want.clean_rows)

    # the same shape at twice the size: resized with Lanczos to the stated size, then encoded
    large = picture.resize((80, 96), Image.Resampling.BICUBIC)
    resized_request = cache.request(**dict(picture=digest, references=[], prompt="a dog", height=48, width=40, latent_seed=7,
                                           latent_float16_round=False, reference_sizing="recipe", reference_seed=42,
                                           reference_float16_round=False, vae_dtype="float32", text_dtype="float32",
                                           settings=TINY_SETTINGS, encoders={n: "tiny" for n in cache.ENCODERS}))
    resized = cache.encode(c, resized_request, large, [])["latents"]
    assert same(resized, direct_latents(vae, large.resize((40, 48), Image.Resampling.LANCZOS), 7, False))
    assert cache.target_picture(large, 48, 40, 8).size == (40, 48) and cache.target_picture(picture, 48, 40, 8).size == (40, 48)
    for shape in ((64, 48), (48, 40), (96, 40)):  # (width, height) pictures that are not 40x48's shape
        with pytest.raises(ValueError, match="does not have the shape of the stated 40x48"):
            cache.target_picture(Image.new("RGB", shape), 48, 40, 8)
    cache.target_picture(Image.new("RGB", (83, 96)), 48, 40, 8)  # within one multiple a side
    with pytest.raises(ValueError, match="canvas multiple of 8; these components have 16"):
        cache.encode(SimpleNamespace(canvas_multiple=16), request, picture, [])
    with pytest.raises(ValueError, match="names 0 references, got 1"):
        cache.encode(c, request, picture, [picture])


@NEEDS_PROCESSOR
@pytest.mark.parametrize("sizing", cache.SIZING)
def test_an_edit_entry_is_the_pipelines_conditioning(sizing):
    """With the tiny release's text encoder, processor and VAE. Under the pipeline's choices (its sizing, seed 42,
    float16 rounding) an edit entry's reference latents, text rows and tags are the ones the sampling pipeline itself
    makes for the same pictures and instruction (`image.generate`, which is tested against diffusers). Under the
    recipe's sizing the references are encoded at the recipe's sizes and the text encoder sees them at its sizes."""
    import test_p_h3_image as T

    t = T.build()
    c = T.components(t)
    rng = np.random.default_rng(11)
    pictures = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)) for h, w in ((40, 48), (56, 32), (96, 120))]
    target = Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    prompt = "give the lighthouse in picture 1 the stripes from picture 2"
    settings = dict(reference_min_pixels=E2E["reference_min_pixels"], reference_max_pixels=E2E["reference_max_pixels"],
                    vision_pixels=E2E["vision_pixels"], canvas_multiple=c.canvas_multiple)
    request = cache.request(picture=files.sha256_json("target"), references=[files.sha256_json(i) for i in range(3)],
                            prompt=prompt, height=48, width=64, latent_seed=5, latent_float16_round=False,
                            reference_sizing=sizing, reference_seed=42, reference_float16_round=True, vae_dtype="float32",
                            text_dtype="float32", settings=settings, encoders={n: "tiny" for n in cache.ENCODERS})
    tensors = cache.check_tensors(request, cache.encode(c, request, target, pictures))
    multiple = c.canvas_multiple
    want_sizes = [cache.reference_sizes(sizing, *p.size, target_height=48, target_width=64, min_pixels=settings["reference_min_pixels"],
                                        max_pixels=settings["reference_max_pixels"], vision_pixels=settings["vision_pixels"],
                                        multiple=multiple) for p in pictures]
    ratio = c.vae.cfg.spatial_ratio
    for i, (vae_size, _) in enumerate(want_sizes):
        assert tensors[f"condition_latents.{i}"].shape[2:4] == (vae_size[1] // ratio, vae_size[0] // ratio)
    direct = I.encode_pictures(c, [I.resize(p, s[0]) for p, s in zip(pictures, want_sizes)])
    assert all(same(tensors[f"condition_latents.{i}"], x) for i, x in enumerate(direct))
    ids, tags, vision = I.edit_presentation(c, prompt, [I.resize(p, s[1]) for p, s in zip(pictures, want_sizes)])
    assert to_np(tensors["text_token_tags"]).tolist() == list(tags) and 0 in tags and 1 in tags
    assert same(tensors["text"], I.connect(c.connector, c.encode_states(ids, vision), mx.float32))

    debug = {}
    with T._quiet():
        I.generate(T.components(t), prompt, images=pictures, height=48, width=64, seed=1, debug=debug)
    pipeline_latents = debug["condition_latents"]
    if sizing == "pipeline":
        assert all(same(tensors[f"condition_latents.{i}"], x) for i, x in enumerate(pipeline_latents))
        assert same(tensors["text"], debug["prompt_embeds"]) and to_np(tensors["text_token_tags"]).tolist() == debug["text_token_tags"].tolist()
    else:
        assert tensors["condition_latents.2"].shape != pipeline_latents[2].shape, "the large reference is scaled down to the target"
