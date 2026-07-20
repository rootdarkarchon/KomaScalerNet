using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using KomaScaler.Cache;
using KomaScaler.Concurrency;
using KomaScaler.Configuration;
using KomaScaler.Images;
using KomaScaler.Inference;
using KomaScaler.Models;
using KomaScaler.Observability;
using KomaScaler.Pipeline;
using KomaScaler.Tiling;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Options;

var builder = WebApplication.CreateBuilder(args);
builder.Configuration.AddKeyPerFile("/run/credentials/komascaler", optional: true);
builder.Services.Configure<UpscalingOptions>(builder.Configuration.GetSection(UpscalingOptions.SectionName));
builder.Services.Configure<TilingOptions>(builder.Configuration.GetSection(UpscalingOptions.SectionName + ":Tiling"));
var startupOptions = builder.Configuration.GetSection(UpscalingOptions.SectionName).Get<UpscalingOptions>() ?? new UpscalingOptions();
ValidateStartupOptions(startupOptions, builder.Environment.IsEnvironment("Testing"));
builder.Services.Configure<FormOptions>(value => value.MultipartBodyLengthLimit = startupOptions.Input.MaxUploadBytes + 1_048_576);
builder.WebHost.ConfigureKestrel(value => value.Limits.MaxRequestBodySize = startupOptions.Input.MaxUploadBytes + 1_048_576);

builder.Services.AddSingleton(startupOptions);
builder.Services.AddSingleton<InventoryState>();
builder.Services.AddSingleton<ITilingPolicyProvider, TilingPolicyProvider>();
builder.Services.AddSingleton<IImageProcessor, VipsImageProcessor>();
builder.Services.AddSingleton<IResultCache>(_ => new FileResultCache(startupOptions.Cache.Directory));
builder.Services.AddSingleton<InFlightRegistry>();
builder.Services.AddSingleton<KomaMetrics>();
builder.Services.AddSingleton<IGpuUpscaler, GpuUpscaler>();
builder.Services.AddSingleton<ITileInferenceBackend>(provider => provider.GetRequiredService<IGpuUpscaler>());
builder.Services.AddSingleton<ConversionQueue>();
builder.Services.AddSingleton<IConversionQueue>(provider => provider.GetRequiredService<ConversionQueue>());
builder.Services.AddHostedService(provider => provider.GetRequiredService<ConversionQueue>());
builder.Services.AddHostedService<GpuIdleService>();
builder.Services.AddHostedService<CacheCleanupService>();
builder.Services.AddHostedService<InventoryRefreshService>();

var app = builder.Build();
var inventoryState = app.Services.GetRequiredService<InventoryState>();
await inventoryState.RefreshAsync(app.Lifetime.ApplicationStopping).ConfigureAwait(false);
if (startupOptions.Models.PreloadOnStartup && inventoryState.IsReady)
    await app.Services.GetRequiredService<IGpuUpscaler>().WarmAsync(app.Lifetime.ApplicationStopping).ConfigureAwait(false);

app.MapGet("/health/live", () => Results.Ok(new { status = "live" }));
app.MapGet("/health/ready", (InventoryState inventory, ITilingPolicyProvider tiling, IGpuUpscaler gpu, IResultCache cache) =>
{
    var failures = new List<string>();
    failures.AddRange(inventory.Result.Errors);
    if (tiling.ValidationError is not null) failures.Add(tiling.ValidationError);
    if (gpu.State == GpuServiceState.Faulted) failures.Add("GPU service is faulted.");
    try { Directory.CreateDirectory(Path.GetDirectoryName(cache.PathFor(new string('0', 64)))!); }
    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { failures.Add("Cache path is not writable: " + ex.Message); }
    try { _ = NetVips.NetVips.Version(0); }
    catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException) { failures.Add("libvips is unavailable: " + ex.Message); }
    return failures.Count == 0
        ? Results.Ok(new { status = "ready", gpu = gpu.State.ToString(), tiling = tiling.Snapshot() })
        : Results.Json(new { status = "not-ready", gpu = gpu.State.ToString(), errors = failures }, statusCode: StatusCodes.Status503ServiceUnavailable);
});
app.MapGet("/metrics", (KomaMetrics metrics, IConversionQueue queue, IGpuUpscaler gpu, InFlightRegistry inFlight) =>
    Results.Text(metrics.Render(queue.Depth, gpu.State.ToString(), inFlight.Count), "text/plain; version=0.0.4"));

app.MapPost("/convert", ConvertAsync).DisableAntiforgery();
app.Run();

static async Task<IResult> ConvertAsync(
    HttpContext context, UpscalingOptions options, InventoryState inventoryState,
    ITilingPolicyProvider tilingPolicies, IImageProcessor images, IResultCache cache,
    InFlightRegistry inFlight, IConversionQueue queue, KomaMetrics metrics, ILogger<Program> logger)
{
    var started = Stopwatch.StartNew();
    context.Response.Headers["X-Correlation-ID"] = context.TraceIdentifier;
    if (!Authenticate(context, options.Security)) return Results.Unauthorized();
    if (!context.Request.HasFormContentType) return Results.BadRequest(new { error = "multipart/form-data is required" });

    IFormCollection form;
    try { form = await context.Request.ReadFormAsync(context.RequestAborted).ConfigureAwait(false); }
    catch (Exception ex) when (ex is BadHttpRequestException or InvalidDataException or InvalidOperationException)
    {
        return Results.BadRequest(new { error = "invalid multipart body" });
    }
    var files = form.Files.Where(x => string.Equals(x.Name, "image", StringComparison.Ordinal)).ToArray();
    if (files.Length != 1 || form.Files.Count != 1) return Results.BadRequest(new { error = "exactly one image part is required" });
    if (files[0].Length <= 0 || files[0].Length > options.Input.MaxUploadBytes) return Results.StatusCode(StatusCodes.Status413PayloadTooLarge);

    byte[] original;
    var input = files[0].OpenReadStream();
    await using (input.ConfigureAwait(false))
    {
        original = await ReadBoundedAsync(input, options.Input.MaxUploadBytes, context.RequestAborted).ConfigureAwait(false);
    }
    var originalMime = ImageSignatures.DetectMime(original) ?? "application/octet-stream";
    try
    {
        var inspection = images.Inspect(original, options.Input.MaxDecodedPixels, options.Input.MonochromeThreshold);
        if (!inspection.Supported || !inspection.IsMonochrome || inspection.PreprocessedLuminance is null || !inventoryState.IsReady)
            return Page(original, originalMime, "bypass", context, metrics, started.Elapsed);
        var inventory = inventoryState.Inventory!;
        var model = inventory.Select(inspection.Height);
        var tiling = tilingPolicies.Snapshot();
        var sourceHash = SHA256.HashData(original);
        var key = PipelineKey.Create(new(sourceHash, model, inventory.SelectionPolicyVersion, tiling,
            TensorRtPolicy.CacheIdentity(options.Models, inventory.Runtime.OnnxRuntimeVersion),
            options.Output.Format, options.Output.Lossless, options.Output.Quality, options.Output.Effort,
            options.Output.PngCompression));
        var cached = await cache.TryReadAsync(key, context.RequestAborted).ConfigureAwait(false);
        if (cached is not null) return Page(cached, ImageSignatures.DetectMime(cached)!, "cache", context, metrics, started.Elapsed);

        var registration = inFlight.GetOrAdd(key, () => queue.EnqueueAsync(key, model, inspection.PreprocessedLuminance, inspection.Width, inspection.Height, tiling));
        using var deadline = new CancellationTokenSource(options.Queue.ResponseDeadline);
        using var wait = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted, deadline.Token);
        try
        {
            var result = await registration.Task.WaitAsync(wait.Token).ConfigureAwait(false);
            return Page(result.Bytes, result.ContentType, registration.IsProducer ? "upscaled" : "deduplicated", context, metrics, started.Elapsed);
        }
        catch (OperationCanceledException) when (deadline.IsCancellationRequested || context.RequestAborted.IsCancellationRequested)
        {
            return Page(original, originalMime, "fallback", context, metrics, started.Elapsed);
        }
        catch (QueueAdmissionException)
        {
            return Page(original, originalMime, "fallback", context, metrics, started.Elapsed);
        }
    }
    catch (Exception ex) when (ex is not OperationCanceledException || !context.RequestAborted.IsCancellationRequested)
    {
        logger.LogWarning(ex, "Recoverable conversion failure for correlation {CorrelationId}", context.TraceIdentifier);
        return Page(original, originalMime, "fallback", context, metrics, started.Elapsed);
    }
}

static IResult Page(byte[] bytes, string contentType, string result, HttpContext context, KomaMetrics metrics, TimeSpan elapsed)
{
    context.Response.Headers["X-Upscaler-Result"] = result;
    metrics.Record(result, elapsed);
    return Results.Bytes(bytes, contentType);
}

static bool Authenticate(HttpContext context, SecurityOptions security)
{
    if (string.IsNullOrEmpty(security.Token)) return true;
    if (!context.Request.Headers.TryGetValue(security.TokenHeader, out var supplied) || supplied.Count != 1) return false;
    var expectedBytes = Encoding.UTF8.GetBytes(security.Token);
    var suppliedBytes = Encoding.UTF8.GetBytes(supplied[0] ?? string.Empty);
    return expectedBytes.Length == suppliedBytes.Length && CryptographicOperations.FixedTimeEquals(expectedBytes, suppliedBytes);
}

static async Task<byte[]> ReadBoundedAsync(Stream source, long maximum, CancellationToken ct)
{
    using var destination = new MemoryStream((int)Math.Min(maximum, 1_048_576));
    var buffer = new byte[81_920];
    long total = 0;
    while (true)
    {
        var read = await source.ReadAsync(buffer, ct).ConfigureAwait(false);
        if (read == 0) break;
        total += read;
        if (total > maximum) throw new BadHttpRequestException("Image part is too large.", StatusCodes.Status413PayloadTooLarge);
        await destination.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
    }
    return destination.ToArray();
}

static void ValidateStartupOptions(UpscalingOptions options, bool nativeIndependentTestHost)
{
    var tilingError = OptionsValidation.ValidateTiling(options.Tiling);
    if (tilingError is not null) throw new InvalidOperationException(tilingError);
    if (options.Input.MaxUploadBytes <= 0 || options.Input.MaxDecodedPixels <= 0) throw new InvalidOperationException("Input limits must be positive.");
    var providerError = TensorRtPolicy.Validate(options.Models);
    if (providerError is not null) throw new InvalidOperationException(providerError);
    if (!nativeIndependentTestHost && !string.Equals(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"), "0", StringComparison.Ordinal))
        throw new InvalidOperationException("TensorRT requires NVIDIA_TF32_OVERRIDE=0 before process startup.");
    if (!options.Models.PreloadOnStartup)
        throw new InvalidOperationException("Production TensorRT requires PreloadOnStartup=true so every compatible engine exists before HTTP readiness.");
    if (options.Tiling.MaximumCoreSize + (2 * options.Tiling.ContextPixelsPerSide) > options.Models.TensorRtProfileMaximumExtent)
        throw new InvalidOperationException("The configured tile extent exceeds the TensorRT maximum profile extent.");
    if (options.Queue.Capacity <= 0 || options.Queue.AdmissionTimeout <= TimeSpan.Zero || options.Queue.ResponseDeadline <= TimeSpan.Zero) throw new InvalidOperationException("Queue settings must be positive.");
    if (!string.Equals(options.Output.Format, "webp", StringComparison.OrdinalIgnoreCase) && !string.Equals(options.Output.Format, "png", StringComparison.OrdinalIgnoreCase))
        throw new InvalidOperationException("Output format must be webp or png.");
    if (!options.Output.Lossless || options.Output.Quality is < 1 or > 100 || options.Output.Effort is < 0 or > 6 || options.Output.PngCompression is < 0 or > 9)
        throw new InvalidOperationException("Output must be lossless; WebP quality must be 1..100 and effort 0..6; PNG compression must be 0..9.");
}

public partial class Program;
