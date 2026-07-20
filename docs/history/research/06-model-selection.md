# MangaJaNai × Suwayomi — Step 6: Model Selection

Investigation date: 2026-07-20
Status: complete; production routing rule frozen

## Outcome

Select the model from the page's **source height in pixels after EXIF orientation
has been applied**. Do not use width, longest side, DPI, aspect ratio, file size,
or a resized dimension.

The exact production bands are:

| EXIF-oriented source height | Model |
|---:|---:|
| 1–1250 | 1200p |
| 1251–1350 | 1300p |
| 1351–1450 | 1400p |
| 1451–1550 | 1500p |
| 1551–1760 | 1600p |
| 1761–1984 | 1920p |
| 1985 and above | 2048p |

These bands are exactly equivalent to choosing the nearest nominal height from
`1200, 1300, 1400, 1500, 1600, 1920, 2048`, assigning an exact midpoint tie to
the lower model, and clamping below and above the model family. The explicit
bands in `mangajanai-step-4-models.json` are the production authority; the
nearest-height description is the semantic invariant used to validate them.

No resize occurs before selection or model inference in the normal 2× path.

## Source behavior and deliberate production correction

The built-in 2× workflows are hardcoded in
[`MainWindowViewModel.cs`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/ViewModels/MainWindowViewModel.cs).
They declare the seven ranges above as `0xHEIGHT` constraints. The backend
function
[`should_chain_activate_for_image`](https://github.com/the-database/MangaJaNaiConverterGui/blob/e63e7843ba45e2a2d9fd9007e3ea33aec9b9e222/MangaJaNaiConverterGui/backend/src/run_upscale.py)
compares width and height constraints independently. Because width is zero in
these workflows, only height participates.

The current GUI reads the raw decoded dimensions and does not autorotate EXIF
images. Step 2 identified that as a decoder defect. Production deliberately
corrects it:

```text
decode and convert embedded profile to sRGB
→ apply EXIF orientation to pixels
→ validate the oriented dimensions
→ apply any bypass policy
→ select from oriented height
→ preprocess, tile, and infer
```

For images without a rotation-bearing EXIF orientation—which covers the normal
case—this is byte-for-byte the same routing decision as the GUI. For EXIF
orientations 5–8, selection follows the height readers actually see rather than
the unrotated storage height.

## Edge-case policy

| Case | Frozen behavior |
|---|---|
| Height below 1200 | Select 1200p; do not extrapolate to a nonexistent smaller model |
| Height above 2048 | Select 2048p; do not reject or invent a larger model |
| Exact midpoint | Select the lower model: 1250→1200p, 1350→1300p, 1450→1400p, 1550→1500p, 1760→1600p, 1984→1920p |
| Landscape/double-page spread | Use oriented height exactly as for a portrait page; width affects tiling only |
| Very wide spread | Do not switch to 2048p merely because the width or longest side is large |
| Tiny image | If it is not bypassed by a separate API policy, route it to 1200p |
| Invalid decoded height (`<= 0`) | Reject as an invalid image before model selection and return the original valid body where possible |
| No matching configured band | Treat as a startup/configuration error; never silently choose a model |

Size-based bypassing is intentionally separate from model selection. This step
does not introduce a new thumbnail cutoff because the original application has
no model-selection cutoff and no quality evidence supports one yet. A future
bypass threshold must be independently configured and included in the pipeline
version/cache key.

## Representative tests performed

The checked `models.json` bands are continuous, non-overlapping,
midpoint-derived, and open-ended at 2048p. Twenty-two scalar boundary and clamp
vectors passed, including every boundary on both sides:

| Input heights | Expected model |
|---|---:|
| 1, 64, 1200, 1250 | 1200p |
| 1251, 1300, 1350 | 1300p |
| 1351, 1400, 1450 | 1400p |
| 1451, 1500, 1550 | 1500p |
| 1551, 1600, 1760 | 1600p |
| 1761, 1920, 1984 | 1920p |
| 1985, 2048, 10000 | 2048p |

Representative geometry cases also preserve the intended dimension:

| Stored dimensions and orientation | Oriented dimensions | Selected model | Reason |
|---|---|---:|---|
| 1800×1200, normal | 1800×1200 | 1200p | Height is 1200 |
| 1800×1200, EXIF rotate 90° | 1200×1800 | 1920p | Selection occurs after orientation |
| 1100×1600 portrait | 1100×1600 | 1600p | Width is irrelevant |
| 3200×1600 spread | 3200×1600 | 1600p | Same height gives the same model |
| 5000×1900 wide spread | 5000×1900 | 1920p | Longest side is not used |
| 903×1761 odd portrait | 903×1761 | 1920p | First pixel above the 1600p band |
| 900×2500 tall scan | 900×2500 | 2048p | Upper clamp |
| 64×64 thumbnail | 64×64 | 1200p | Lower clamp if not separately bypassed |

Step 5 already exercised every one of the seven models on all thirteen content
fixtures, including portrait/odd images, a landscape double-page surrogate,
high-contrast lines, noisy JPEG material, gradients, and screentones. The FP32
CUDA path passed all 91 model/fixture pairs. This establishes that routing any
of those representative page types to its selected model does not expose a
model-specific runtime or graph failure.

The official repositories do not provide a redistributable set of full pages at
every boundary height, so this step does not claim a subjective cross-model
quality study on such a corpus. Step 7's target-host full-page tests will include
user-owned pages below, at, and above the bands and may reopen routing only if
they provide concrete visual evidence against the official midpoint rule.

## Production implementation contract

Use the explicit ranges from `models.json`, not duplicated constants scattered
through the API and worker. Validate the metadata once at startup:

1. nominal heights are strictly increasing;
2. the first band begins at zero;
3. adjacent bands are continuous and non-overlapping;
4. each finite maximum is the integer midpoint of adjacent nominal heights;
5. exact midpoint ties therefore remain in the lower band;
6. only the last band is open-ended;
7. every referenced ONNX artifact and SHA-256 is present.

Minimal C# selector shape:

```csharp
public sealed record ModelBand(
    int NominalHeight,
    int MinimumHeight,
    int? MaximumHeight,
    string ModelId);

public static ModelBand SelectModel(
    int exifOrientedSourceHeight,
    IReadOnlyList<ModelBand> validatedBands)
{
    ArgumentOutOfRangeException.ThrowIfNegativeOrZero(
        exifOrientedSourceHeight);

    foreach (var band in validatedBands)
    {
        if (exifOrientedSourceHeight >= band.MinimumHeight &&
            (band.MaximumHeight is null ||
             exifOrientedSourceHeight <= band.MaximumHeight.Value))
        {
            return band;
        }
    }

    throw new InvalidOperationException(
        "Validated model bands did not cover the decoded image height.");
}
```

Record at least `orientedWidth`, `orientedHeight`, `selectedModelId`,
`nominalHeight`, and `selectionPolicyVersion` in structured request logs. The
selected model ID, ONNX SHA-256, and selection-policy version must participate
in the persistent cache key.

## Worker consequence

All seven FP32 ONNX sessions remain resident in the one GPU worker established
in Step 5. A page-height change selects another already-loaded session; it does
not terminate or replace the worker. GPU inference remains one serialized lane.
The worker is terminated only for the configured idle timeout, controlled
shutdown/restart, or fault recovery, which preserves the requirement that at
most one GPU worker process is active.

## Step 6 decision

Freeze policy version `mangajanai-v1-height-midpoint-oriented-v1`:

```text
EXIF-oriented source height
→ explicit official midpoint band
→ lower model on exact tie
→ 1200p/2048p clamps at the extremes
→ one of seven already-resident FP32 sessions
```

Step 7 can now measure tile size, context/overlap, stitching, seam visibility,
peak VRAM, and full-page latency without leaving model routing ambiguous.
