# MangaJaNai × Suwayomi — Step 1: Official 2× Model Inventory

**Investigation date:** 2026-07-20
**Status:** Complete for the current public, stable MangaJaNai release

## Result

The current public MangaJaNai V1 release contains seven black-and-white 2×
models, each trained for a nominal input page height. Direct inspection of all
seven official `.pth` files establishes that:

- every file is a PyTorch ZIP checkpoint whose top-level object is a plain
  `collections.OrderedDict` state dictionary;
- none is a complete pickled model or a training-checkpoint wrapper;
- every state dictionary contains exactly 702 tensors;
- every tensor is stored as PyTorch `HalfStorage` (FP16);
- every model contains 16,703,171 parameters, including biases;
- every model has the same ordered key, shape, and dtype signature;
- the seven files therefore share one graph and differ only in parameter
  values;
- the effective graph is a three-channel, 2× ESRGAN/RRDBNet model.

This removes the main uncertainty around export: one architecture
implementation and one export procedure can be reused for all seven weights.

## Authoritative model set

The official [MangaJaNai 1.0.0 release](https://github.com/the-database/MangaJaNai/releases/tag/1.0.0)
lists the following seven 2× models. The current project README still identifies
these seven heights as the MangaJaNai black-and-white model family:
[1200, 1300, 1400, 1500, 1600, 1920, and 2048](https://github.com/the-database/MangaJaNai/blob/d2f8f92dc75303c557ca2d7a507913eff512ff32/README.md).

| Nominal input height | Official filename | File size (bytes) | SHA-256 |
|---:|---|---:|---|
| 1200 | `2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth` | 33,709,234 | `43b784f674bdbf89886a62a64cd5f8d8df92caf4d861bdf4d47dad249ede0267` |
| 1300 | `2x_MangaJaNai_1300p_V1_ESRGAN_75k.pth` | 33,708,528 | `15ca3c0f75f97f7bf52065bf7c9b8d602de94ce9e3b078ac58793855eed18589` |
| 1400 | `2x_MangaJaNai_1400p_V1_ESRGAN_70k.pth` | 33,708,528 | `a940ad8ebcf6bea5580f2f59df67deb009f054c9b87dbbc58c2e452722f34858` |
| 1500 | `2x_MangaJaNai_1500p_V1_ESRGAN_90k.pth` | 33,708,528 | `d91f2d247fa61144c1634a2ba46926acd3956ae90d281a5bed6655f8364a5b2c` |
| 1600 | `2x_MangaJaNai_1600p_V1_ESRGAN_90k.pth` | 33,708,528 | `6f5923f812dbc5d6aeed727635a21e74cacddce595afe6135cbd95078f6eee44` |
| 1920 | `2x_MangaJaNai_1920p_V1_ESRGAN_70k.pth` | 33,708,528 | `1ad4aa6f64684baa430da1bb472489bff2a02473b14859015884a3852339c005` |
| 2048 | `2x_MangaJaNai_2048p_V1_ESRGAN_95k.pth` | 33,708,528 | `146cd009b9589203a8444fe0aa7195709bb5b9fdeaca3808b7fbbd5538f94c41` |

The official `MangaJaNai_V1_ModelsOnly.zip` asset is 433,383,160 bytes and has
SHA-256:

```text
5156f4167875bba51a8ed52bd1c794b0d7277f7103f99b397518066e4dda7e55
```

The small size difference of the 1200p file is serialization-container
overhead. Its tensor count, parameter count, dtype, keys, and shapes are
identical to the other six.

## Checkpoint structure

Each checkpoint has this logical structure:

```text
PyTorch ZIP archive
└── data.pkl
    └── OrderedDict<string, Tensor>
```

There is no outer `state_dict`, `params`, `params_ema`, optimizer, scheduler,
epoch, or model-object entry. The first and last tensors in all seven files are:

```text
model.0.weight  Half [64, 12, 3, 3]
model.0.bias    Half [64]
...
model.10.weight Half [3, 64, 3, 3]
model.10.bias   Half [3]
```

The storage location recorded by the serializer is `cuda:0`. That is only
checkpoint metadata; a loader with device remapping can load it on CPU or any
chosen CUDA device.

## Reconstructed architecture

The current MangaJaNaiConverterGui pins `spandrel==0.4.1` in its
[backend dependencies](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/pyproject.toml)
and loads files through Spandrel's architecture-detecting `ModelLoader` in
[`load_model.py`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/packages/chaiNNer_pytorch/pytorch/io/load_model.py).

Spandrel 0.4.1 identifies this exact old-style key layout as ESRGAN, derives
the graph parameters from tensor shapes, and detects the pixel-unshuffle
variant. Its pinned implementation is visible in the
[ESRGAN detector](https://github.com/chaiNNer-org/spandrel/blob/v0.4.1/libs/spandrel/spandrel/architectures/ESRGAN/__init__.py)
and [RRDBNet implementation](https://github.com/chaiNNer-org/spandrel/blob/v0.4.1/libs/spandrel/spandrel/architectures/ESRGAN/__arch/RRDB.py).

The resulting architecture is:

| Property | Verified value |
|---|---:|
| Architecture family | ESRGAN RRDBNet, old-style state-dict layout |
| External input channels | 3 |
| External output channels | 3 |
| Effective scale | 2× |
| Feature channels | 64 |
| RRDB blocks | 23 |
| Residual dense blocks per RRDB | 3 |
| Dense convolutions per residual dense block | 5 |
| Growth channels | 32 |
| Kernel size | 3×3 |
| ESRGAN+ | No |
| Pixel-unshuffle factor | 2 |
| Internal first-convolution input channels | 12 |
| Internal upsampling factor | 4× |
| Parameter count | 16,703,171 |
| Stored parameter dtype | FP16 |

The apparent 12-channel input and 4× internal upsampling are not contradictions.
For an RGB input, a factor-2 pixel unshuffle transforms 3 channels into 12
channels while halving each spatial dimension. The RRDBNet then upsamples that
representation by 4×, producing an effective 2× result relative to the original
image. Spandrel reflect-pads odd input dimensions before pixel unshuffle and
crops the final tensor back to exactly `2H × 2W`.

## Current GUI selection bands

The current default workflow in
[`MainWindowViewModel.cs`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/ViewModels/MainWindowViewModel.cs)
uses these exact ranges for the 2× black-and-white chains:

| Original page height | Selected model |
|---:|---:|
| 0–1250 | 1200p |
| 1251–1350 | 1300p |
| 1351–1450 | 1400p |
| 1451–1550 | 1500p |
| 1551–1760 | 1600p |
| 1761–1984 | 1920p |
| 1985 and above | 2048p |

The backend's
[`should_chain_activate_for_image`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/run_upscale.py)
compares the original width and height independently. Because the default
MangaJaNai chains specify `0xHEIGHT`, selection is based on original image
height only. It does not use width, longest side, DPI, or orientation, and it
does not resize the page before inference in the default workflow.

This confirms that the thresholds are midpoints between the nominal training
heights, with open-ended clamping to 1200p and 2048p at the extremes.

## Licensing and redistribution

The model repository's authoritative
[`LICENSE`](https://github.com/the-database/MangaJaNai/blob/d2f8f92dc75303c557ca2d7a507913eff512ff32/LICENSE)
is **Creative Commons Attribution-NonCommercial 4.0 International
(CC BY-NC 4.0)**. It is not the ShareAlike variant sometimes attributed to
MangaJaNai by third-party model indexes.

Practical consequences for this project:

- local noncommercial use is within the granted scope;
- conversion to ONNX is an allowed technical format modification;
- distributing the original or converted model files must remain
  noncommercial;
- redistribution must preserve appropriate creator attribution, a license
  reference/link, relevant notices, and an indication that conversion or other
  modification occurred;
- no implication of endorsement is permitted;
- downstream recipients may not be subjected to additional legal or
  technological restrictions on the licensed model material;
- commercial use or commercial redistribution requires separate permission
  from the rights holder.

The service's own source code can be licensed separately from the model files,
provided the model artifacts and notices remain clearly separated and their
license is honored. This is an engineering interpretation of the published
license, not legal advice.

## Inspection method

The official release asset was downloaded directly from the GitHub release.
All seven 2× files were extracted and inspected without executing their pickle
payloads. A restricted metadata-only unpickler accepted only:

- `collections.OrderedDict`;
- PyTorch `_rebuild_tensor_v2` records;
- PyTorch storage descriptors.

It did not import PyTorch, instantiate arbitrary checkpoint classes, or read
tensor values into a runtime. For every model it recorded the ordered key list,
tensor shapes, dtypes, storage metadata, file size, and SHA-256. A hash of the
complete ordered `(key, shape, dtype)` sequence was identical for all seven:

```text
011f1e16fd5840400f61d725b5792f548cc589cd8ad3c4d09f76a9621ad95a3f
```

## Remaining caveats

- This inventory covers the current public stable V1 GitHub release. The
  project README mentions experimental/pre-release models distributed through
  Discord; those are not part of the stable production baseline and were not
  treated as authoritative deployment inputs.
- Numerical weight validation and actual PyTorch-to-ONNX execution belong to
  Step 4. Step 1 proves graph compatibility, not output equivalence.
- Exact decoding, normalization, grayscale conversion, tiling, and output
  processing belong to Step 2 and remain intentionally unresolved here.

## Step 1 decision

Use the seven official V1 2× `.pth` files as the baseline model set. Implement
one RRDBNet exporter parameterized by the input file, with the verified graph:

```text
RGB input
→ reflect-pad for pixel unshuffle when required
→ pixel-unshuffle ×2
→ 64-feature / 23-block RRDBNet
→ internal 4× upsampling
→ RGB output cropped to 2H × 2W
```

Do not write seven architecture variants. Store model-specific nominal height,
selection range, source SHA-256, converted ONNX SHA-256, and license metadata
outside the graph in `models.json`.
