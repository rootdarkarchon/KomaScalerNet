# Optimization 4: lossless output encoding

> **Historical record:** This document preserves dated measurements or acceptance evidence. It is not current operator guidance; use `README.md` and `docs/operations/DEPLOYMENT.md`.

## Decision

Production now defaults to **single-band lossless PNG at compression level 3**.
On the target host it reduced the median output-stage reader proxy (localhost
transfer plus decode) from 164.664 ms for the former WebP effort-2 policy to
50.088 ms. Median encoding fell from 2,445.254 ms to 221.115 ms. The PNG was
294,940 bytes (6.66%) larger, but its transfer time was slightly lower in this
loopback sample and its much faster decode dominates end-to-end reader latency.

PNG level 0 was fastest but doubled the payload to 9.31 MB. Level 3 is the
balanced production choice: it retained most of PNG's latency advantage with a
4.72 MB payload. WebP is still supported as a configurable option; it is no
longer the default merely because it was implemented first.

## Scope and method

Measured on 2026-07-20 with .NET SDK 10.0.302, libvips 8.16.1, Linux
6.12.74+deb13+1-amd64. The source was the exact 3320×2800 output used by the
earlier target benchmark. The test extracted band zero once and retained one
9,296,000-byte grayscale pixel buffer for every encoder case.

The benchmark is an opt-in xUnit test in
`tests/KomaScaler.UnitTests/OutputEncodingBenchmarkTests.cs`. Each case used one
warm-up, five encode samples, five decode samples, and seven complete localhost
HTTP transfers through a fresh loopback TCP server. Reported times are medians.
Every HTTP response was read completely and compared byte-for-byte with the
encoded payload. Every decoded pixel was compared with the common source
buffer. `reader` is the sum of median localhost transfer and median decode; it
does not include GPU inference, browser painting, or Wi-Fi latency.

```bash
KOMASCALER_ENCODING_SOURCE=/devtmp/komascaler-acceptance/parallel-final/first.webp \
KOMASCALER_ENCODING_OUTPUT=/devtmp/komascaler-acceptance/output-encoding \
dotnet test tests/KomaScaler.UnitTests/KomaScaler.UnitTests.csproj \
  -c Release --no-build \
  --filter FullyQualifiedName~LosslessGrayscaleEncodingMatrix \
  --logger 'console;verbosity=detailed'
```

Raw JSON and encoded artifacts are in
`/devtmp/komascaler-acceptance/output-encoding`. They are deliberately outside
the repository.

## Measured results

| Encoder | Encode ms | Bytes | Decode bands | Decode ms | Localhost HTTP ms | Reader ms | Exact pixels |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| WebP effort 0 | 2,418.826 | 4,866,766 | 3 | 174.469 | 3.936 | 178.404 | yes |
| WebP effort 1 | 3,046.909 | 4,467,564 | 3 | 181.623 | 3.768 | 185.391 | yes |
| WebP effort 2 | 2,445.254 | 4,427,946 | 3 | 161.751 | 2.913 | 164.664 | yes |
| PNG level 0 | 19.722 | 9,313,909 | 1 | 19.810 | 5.499 | 25.309 | yes |
| PNG level 1 | 199.338 | 4,776,579 | 1 | 71.535 | 2.898 | 74.433 | yes |
| **PNG level 3** | **221.115** | **4,722,886** | **1** | **47.336** | **2.752** | **50.088** | **yes** |

The input to every encoder was one band. libvips decoded all three PNG outputs
as one band, proving there is no RGB expansion in the PNG path. It decoded the
three WebP outputs as three equal RGB bands. Pixel equality passed for all
formats; for WebP, each expanded channel was checked against the grayscale
source.

The isolated timings also identify the former output bottleneck. WebP effort 2
consumed about 2.45 seconds before the response could begin. PNG level 3 removes
about 2.22 seconds from that stage. GPU tiled inference remains the dominant
whole-conversion cost, but this change materially improves deadline margin.

## Application changes

- `Upscaling:Output:Format` accepts `png` and `webp`; the default is `png`.
- `Upscaling:Output:PngCompression` accepts 0 through 9; the default is 3.
- WebP remains configurable through `Quality` (1 through 100) and `Effort` (0
  through 6). Output must remain lossless.
- PNG is encoded directly from the one-band grayscale buffer with palette
  disabled and an eight-bit depth. No RGB image is constructed.
- Format, lossless mode, WebP quality, WebP effort, and PNG compression are all
  included in the pipeline/cache identity. Cached files use the neutral
  `.image` suffix, are signature-validated as PNG or WebP, and their response
  MIME is derived from the bytes.

## Suwayomi compatibility

The live Suwayomi instance was inspected read-only. Its Web UI and GraphQL API
were reachable at `127.0.0.1:4568`; `fetchChapterPages(chapterId: 1)` returned 54
reader page URLs, and a real request to
`/api/v1/manga/61/chapter/1/page/0` returned HTTP 200 `image/jpeg` (850×1200,
798,307 bytes). This confirms the page route used for a future conversion test.

An actual serve-conversion test of the new PNG response was intentionally not
performed after the operator deferred modification/restart of the live
Suwayomi instance. Consequently, **Suwayomi PNG compatibility is an
operator-deferred result, not a measured pass**. The isolated PNG encode,
libvips decode, byte-exact localhost HTTP transfer, and application tests are
measured passes. To complete the live check during a maintenance window, point
`server.serveConversions.default.target` at KomaScaler, restart Suwayomi,
request one of the page URLs above, and verify HTTP 200, `Content-Type:
image/png`, single-band 2× dimensions, and successful display in the actual
reader; then restore the prior conversion configuration if this is only a
test.
