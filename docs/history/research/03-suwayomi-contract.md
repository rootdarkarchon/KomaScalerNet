# MangaJaNai × Suwayomi — Step 3: Serve-Conversion Contract

Status: complete
Suwayomi Server source snapshot: `1d583ca4a718646e17a4c62d23fe8e5edccaf774`
Suwayomi WebUI source snapshot: `147ecdcd7b2de0c7d46304a98430d181aed84f2c`

## Outcome

Suwayomi's current external serve-conversion interface is a synchronous HTTP multipart request. It sends only the image bytes, a generated filename, and the source MIME type. It supplies no manga, chapter, page, source URL, or cache identifier. The upscaler must therefore identify work and cache entries from the image contents plus its own pipeline/model version.

The production endpoint must return a valid image body for every handled request, including bypasses and recoverable failures. A non-2xx status is not a reliable way to ask Suwayomi to use the original image: the current implementation does not check HTTP status before consuming the response body.

## Minimal configuration

`server.conf` uses HOCON. A minimal same-host configuration is:

```hocon
server.serveConversions = {
  default = {
    target = "http://127.0.0.1:9999/convert"
    callTimeout = 2m
    connectTimeout = 5s
  }
}
```

With a static shared secret:

```hocon
server.serveConversions = {
  default = {
    target = "http://127.0.0.1:9999/convert"
    callTimeout = 2m
    connectTimeout = 5s
    headers = {
      "X-Upscaler-Token" = "replace-with-a-long-random-value"
    }
  }
}
```

If Suwayomi and the upscaler run in different containers, `127.0.0.1` is wrong; use the upscaler's container/service name on their shared network.

The official configuration documentation defines [`server.serveConversions`](https://github.com/Suwayomi/Suwayomi-Server/blob/1d583ca4a718646e17a4c62d23fe8e5edccaf774/docs/Configuring-Suwayomi%E2%80%90Server.md) as having the same map format as `server.downloadConversions`.

## Conversion selection

The current selection logic is in [`Page.getPageImageServe`](https://github.com/Suwayomi/Suwayomi-Server/blob/1d583ca4a718646e17a4c62d23fe8e5edccaf774/server/src/main/kotlin/suwayomi/tachidesk/manga/impl/Page.kt):

1. If the page request has a `format` query parameter, Suwayomi creates an internal format conversion and it takes precedence over `serveConversions`.
2. Otherwise, an exact source-MIME entry such as `"image/jpeg"` is selected.
3. Otherwise, the `default` entry is selected.
4. With no match, the original stream is served unchanged.

To send all page formats to the upscaler, configure `default`. Specific formats can be excluded by mapping them to `target = "none"`, as described by the configuration documentation.

## Exact outbound request

[`ConversionUtil.imageHttpPostProcess`](https://github.com/Suwayomi/Suwayomi-Server/blob/1d583ca4a718646e17a4c62d23fe8e5edccaf774/server/src/main/kotlin/suwayomi/tachidesk/util/ConversionUtil.kt) constructs the request.

| Property | Current behavior |
|---|---|
| Method | `POST` |
| Body type | `multipart/form-data` with an automatically generated boundary |
| Form field | Exactly one part named `image` |
| Additional fields | None |
| Part content type | The MIME type determined for the source page |
| Part filename | A generated temporary filename such as `conversion123456789.jpg`; it is not the original page filename |
| Body length | File-backed multipart body; OkHttp can normally send a known content length |
| Configurable headers | Every entry in `headers` is added with `set(name, value)` |
| Other headers | Normal OkHttp transport headers plus the shared Suwayomi user-agent/cookie interceptors |

Equivalent diagnostic request:

```bash
curl --fail-with-body \
  -X POST 'http://127.0.0.1:9999/convert' \
  -F 'image=@page.jpg;type=image/jpeg'
```

The service must not depend on the multipart filename. It should read the `image` part, enforce upload limits, sniff/decode the bytes, and hash the bytes for cache identity.

## Timeout behavior

The conversion-specific configuration type exposes only:

- `callTimeout`, covering the complete OkHttp call including dispatcher queueing, connection, upload, server processing, and response transfer;
- `connectTimeout`, covering connection establishment.

The fields are parsed by [`DownloadConversionType`](https://github.com/Suwayomi/Suwayomi-Server/blob/1d583ca4a718646e17a4c62d23fe8e5edccaf774/server/server-config/src/main/kotlin/suwayomi/tachidesk/server/util/DownloadConversionType.kt).

When omitted, the shared client in [`NetworkHelper`](https://github.com/Suwayomi/Suwayomi-Server/blob/1d583ca4a718646e17a4c62d23fe8e5edccaf774/server/src/main/kotlin/eu/kanade/tachiyomi/network/NetworkHelper.kt) uses:

| Timeout | Value |
|---|---:|
| Connect | 30 seconds |
| Read | 30 seconds |
| Complete call | 2 minutes |
| Write | Inherited OkHttp default, 10 seconds in the pinned OkHttp 5.4 line |

Setting a long `callTimeout` does **not** replace the 30-second socket read timeout. In the normal request/response pattern, the converter does not send response headers or body bytes until processing finishes. Queue wait after the request reaches the upscaler, cold worker startup, and inference must therefore stay safely below 30 seconds to the first response bytes.

Production consequence: apply an internal deadline of roughly 25 seconds. If an upscale cannot complete in time, return the original valid image with HTTP 200 and let any worker startup or cache warming continue only when it is safe and intentionally designed. Do not allow an unbounded GPU queue to hold Suwayomi connections open.

## Response interpretation

Suwayomi does not trust or use the converter's response `Content-Type`. It buffers the response stream, inspects its first 12 bytes with [`ImageUtil.findImageType`](https://github.com/Suwayomi/Suwayomi-Server/blob/1d583ca4a718646e17a4c62d23fe8e5edccaf774/server/src/main/kotlin/suwayomi/tachidesk/manga/impl/util/storage/ImageUtil.kt), and chooses the MIME type from the detected signature.

Recognized output signatures are JPEG, PNG, GIF, WebP, AVIF, HEIF, and JPEG XL. If detection fails, Suwayomi labels the bytes `image/jpeg` without validating that they are actually a JPEG.

| Converter response | Suwayomi result |
|---|---|
| `200` + valid upscaled image | Serves the detected image format |
| `200` + original image | Serves the original; this is the correct bypass/fallback response |
| Valid image in a different format | Accepted; route `Content-Type` follows magic-byte detection |
| Response `Content-Type` disagrees with bytes | Header is ignored; detected bytes win |
| Response filename or `Content-Disposition` | Ignored |
| Empty body / `204` | Empty bytes are treated as `image/jpeg`; the reader receives a broken image |
| JSON/text error body | Labeled `image/jpeg` and served; the reader receives a broken image |
| Non-2xx with a body | Status is not checked; the body is still consumed and served according to signature/fallback |
| Connection exception or timeout | Conversion returns no stream; outer page handling reopens and serves the original page |

The non-2xx behavior follows from the use of `await()` rather than `awaitSuccess()` in [`OkHttpExtensions.kt`](https://github.com/Suwayomi/Suwayomi-Server/blob/1d583ca4a718646e17a4c62d23fe8e5edccaf774/server/src/main/kotlin/eu/kanade/tachiyomi/network/OkHttpExtensions.kt).

The upscaler contract should therefore be stricter than Suwayomi's consumer:

- return HTTP 200 with valid image bytes for success, bypass, and recoverable failure;
- never return `204` for “unchanged”;
- validate the final encoded image before sending it;
- preserve or deliberately change format, but always make the body self-consistent;
- keep diagnostic details in service logs and response headers, not a JSON error body.

Useful optional response headers such as `X-Upscaler-Result: upscaled|bypass|cache|fallback` will be ignored by Suwayomi but remain valuable for direct testing and logs.

## Downloaded-page behavior

Serve conversions do apply to downloaded pages. `getPageImage()` first checks whether a chapter is downloaded and opens that stored page; `getPageImageServe()` then applies `serveConversions` to the returned stream in the same way as an online/cached page.

`server.downloadConversions` is separate:

- `downloadConversions` converts while writing chapter downloads;
- `serveConversions` converts whenever a page is served to a reader;
- if both are configured with broad `default` entries, a downloaded page can be converted once at download time and then sent through the serve converter again.

For this on-demand architecture, leave `downloadConversions` empty unless pre-conversion of downloads is explicitly wanted.

## Caching behavior

Suwayomi caches the fetched/original page before it calls the serve converter. [`ImageResponse.getImageResponse`](https://github.com/Suwayomi/Suwayomi-Server/blob/1d583ca4a718646e17a4c62d23fe8e5edccaf774/server/src/main/kotlin/suwayomi/tachidesk/manga/impl/util/storage/ImageResponse.kt) handles that source cache.

The converted serve result is **not written to a Suwayomi server-side cache**. A new request that reaches `getPageImageServe()` invokes the converter again. The page route does send `Cache-Control: max-age=86400`, so a browser or downstream HTTP cache may reuse that page URL for one day.

Consequences:

- the upscaler needs its own persistent content-addressed cache;
- the cache key must include a digest of source bytes plus model/pipeline/output-format versions;
- identical simultaneous requests must be deduplicated before entering the GPU queue;
- changing models or settings does not immediately invalidate a browser's one-day cached page URL.

## Prefetch and concurrency

Current Suwayomi WebUI settings define an image preload range of 1–20 with a default of **5** in [`ReaderSettings.constants.tsx`](https://github.com/Suwayomi/Suwayomi-WebUI/blob/147ecdcd7b2de0c7d46304a98430d181aed84f2c/src/features/reader/settings/ReaderSettings.constants.tsx).

[`getPageIndexesToLoad`](https://github.com/Suwayomi/Suwayomi-WebUI/blob/147ecdcd7b2de0c7d46304a98430d181aed84f2c/src/features/reader/viewer/pager/ReaderPager.utils.tsx) normally renders:

- the current page slot;
- five page slots in the reading direction by default;
- up to two previously visited page slots in non-continuous modes;
- preload slots from adjacent chapters near chapter boundaries.

Double-page slots may contain two image requests. The browser, network protocol, and Suwayomi's shared OkHttp dispatcher determine how many run simultaneously; there is no converter-specific single-request gate in Suwayomi.

The API must therefore accept a burst of concurrent uploads and keep those HTTP requests asynchronous, while the GPU stage remains exactly one serialized execution lane. Cache hits, bypasses, and duplicate followers should not occupy the GPU lane.

## Verified application contract

The ASP.NET endpoint can be specified as:

```text
POST /convert
Content-Type: multipart/form-data

required part:
  name: image
  filename: arbitrary/untrusted
  content-type: advisory source MIME
  body: encoded source image

success or bypass response:
  status: 200
  content-type: accurate image MIME (helpful, though Suwayomi sniffs bytes)
  body: complete valid JPEG, PNG, WebP, AVIF, HEIF, GIF, or JXL image
```

Recommended first implementation:

- accept JPEG, PNG, WebP, BMP, and AVIF input after independent sniff/decode validation;
- return lossless WebP for upscaled monochrome pages;
- return the exact original bytes for color/unsupported/bypass and deadline fallback;
- enforce a 25-second response deadline;
- use source-byte SHA-256 plus pipeline version as the cache and in-flight deduplication key;
- expose bounded request-body and decoded-pixel limits;
- accept concurrent HTTP requests but serialize GPU inference in the single worker;
- treat the random multipart filename as metadata only.

## Step 3 acceptance result

The configuration key, request method and multipart shape, filename behavior, headers, timeout behavior, response MIME detection, response-filename irrelevance, all practical failure paths, downloaded-page handling, cache behavior, and current WebUI preload burst have been verified from implementation.

The next investigation step is proving the `.pth` → ONNX export and reference-validation path.
