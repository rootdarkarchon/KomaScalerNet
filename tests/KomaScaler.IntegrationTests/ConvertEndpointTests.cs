using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using KomaScaler.Images;
using KomaScaler.Inference;
using KomaScaler.Models;
using KomaScaler.Cache;
using KomaScaler.Configuration;
using KomaScaler.Tiling;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Hosting;

namespace KomaScaler.IntegrationTests;

public sealed class ConvertEndpointTests : IClassFixture<KomaFactory>
{
    private readonly KomaFactory _factory;
    public ConvertEndpointTests(KomaFactory factory) => _factory = factory;

    [Fact]
    public async Task Convert_AcceptsArbitraryFilename_AndReturnsExact2xFakeResult()
    {
        using var client = AuthenticatedClient();
        var inventory = _factory.Services.GetRequiredService<InventoryState>();
        Assert.True(inventory.IsReady, string.Join(" | ", inventory.Result.Errors));
        using var content = Multipart(PngBytes(), "totally-untrusted.exe", "text/plain");
        using var response = await client.PostAsync("/convert", content);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("upscaled", response.Headers.GetValues("X-Upscaler-Result").Single());
        var bytes = await response.Content.ReadAsByteArrayAsync();
        Assert.True(KomaScaler.Cache.ImageSignatures.IsWebP(bytes));
        Assert.NotEmpty(response.Headers.GetValues("X-Correlation-ID").Single());
    }

    [Fact]
    public async Task Convert_RepeatedRequest_IsCacheHitWithoutSecondInference()
    {
        using var client = AuthenticatedClient(); var unique = PngBytes().Append((byte)99).ToArray();
        var before = _factory.Backend.Calls;
        using var first = await client.PostAsync("/convert", Multipart(unique, "a", "image/jpeg"));
        using var second = await client.PostAsync("/convert", Multipart(unique, "b", "application/octet-stream"));
        Assert.Equal("upscaled", first.Headers.GetValues("X-Upscaler-Result").Single());
        Assert.Equal("cache", second.Headers.GetValues("X-Upscaler-Result").Single());
        Assert.Equal(before + 1, _factory.Backend.Calls);
    }

    [Fact]
    public async Task Convert_UnsupportedBytesBypassExactly()
    {
        using var client = AuthenticatedClient(); var original = "unknown-image"u8.ToArray();
        using var response = await client.PostAsync("/convert", Multipart(original, "x.jpg", "image/jpeg"));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("bypass", response.Headers.GetValues("X-Upscaler-Result").Single());
        Assert.Equal(original, await response.Content.ReadAsByteArrayAsync());
    }

    [Fact]
    public async Task Convert_ConcurrentIdenticalRequestsShareOneConversion()
    {
        using var client = AuthenticatedClient();
        var unique = PngBytes().Append((byte)77).ToArray();
        var before = _factory.Backend.Calls;
        _factory.Backend.DelayMilliseconds = 25;
        try
        {
            var firstTask = client.PostAsync("/convert", Multipart(unique, "first", "image/png"));
            var secondTask = client.PostAsync("/convert", Multipart(unique, "second", "image/png"));
            using var first = await firstTask; using var second = await secondTask;
            var results = new[] { first, second }.Select(x => x.Headers.GetValues("X-Upscaler-Result").Single()).Order(StringComparer.Ordinal).ToArray();
            Assert.Equal(["deduplicated", "upscaled"], results);
            Assert.Equal(before + 1, _factory.Backend.Calls);
        }
        finally { _factory.Backend.DelayMilliseconds = 0; }
    }

    [Fact]
    public async Task Convert_DifferentImagesRemainSingleReaderWhileTilesMayParallelize()
    {
        using var client = AuthenticatedClient();
        _factory.Backend.DelayMilliseconds = 100;
        try
        {
            var firstTask = client.PostAsync("/convert", Multipart(PngBytes().Append((byte)201).ToArray(), "one", "image/png"));
            var secondTask = client.PostAsync("/convert", Multipart(PngBytes().Append((byte)202).ToArray(), "two", "image/png"));
            using var first = await firstTask;
            using var second = await secondTask;
            Assert.Equal("upscaled", first.Headers.GetValues("X-Upscaler-Result").Single());
            Assert.Equal("upscaled", second.Headers.GetValues("X-Upscaler-Result").Single());
            Assert.Equal(1, _factory.Backend.MaximumObservedActiveRuns);
        }
        finally { _factory.Backend.DelayMilliseconds = 0; }
    }

    [Fact]
    public async Task Convert_ResponseDeadlineReturnsExactOriginalWhileWorkCanFinish()
    {
        using var client = AuthenticatedClient();
        var original = PngBytes().Append((byte)55).ToArray();
        _factory.Backend.DelayMilliseconds = 1500;
        try
        {
            using var response = await client.PostAsync("/convert", Multipart(original, "slow", "image/png"));
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
            Assert.Equal("fallback", response.Headers.GetValues("X-Upscaler-Result").Single());
            Assert.Equal(original, await response.Content.ReadAsByteArrayAsync());
        }
        finally
        {
            await Task.Delay(600);
            _factory.Backend.DelayMilliseconds = 0;
        }
    }

    [Fact]
    public async Task Convert_MissingOrWrongPart_IsControlled()
    {
        using var client = AuthenticatedClient();
        using var empty = new MultipartFormDataContent();
        using var missing = await client.PostAsync("/convert", empty);
        Assert.Equal(HttpStatusCode.BadRequest, missing.StatusCode);
        using var wrong = new MultipartFormDataContent(); wrong.Add(new ByteArrayContent(PngBytes()), "page", "x.png");
        using var wrongResponse = await client.PostAsync("/convert", wrong);
        Assert.Equal(HttpStatusCode.BadRequest, wrongResponse.StatusCode);
    }

    [Fact]
    public async Task Health_LiveIsIndependentFromGpuLoad()
    {
        using var client = _factory.CreateClient();
        using var response = await client.GetAsync("/health/live");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    [Fact]
    public async Task Startup_PrewarmsGpuOnceBeforeReadiness()
    {
        using var client = _factory.CreateClient();
        using var response = await client.GetAsync("/health/ready");
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal(1, _factory.Backend.WarmCalls);
    }

    [Fact]
    public async Task Convert_MissingOrIncorrectTokenIsUnauthorized()
    {
        using var client = _factory.CreateClient();
        using var missing = await client.PostAsync("/convert", Multipart(PngBytes(), "x", "image/png"));
        Assert.Equal(HttpStatusCode.Unauthorized, missing.StatusCode);
        client.DefaultRequestHeaders.Add("X-Upscaler-Token", "incorrect");
        using var incorrect = await client.PostAsync("/convert", Multipart(PngBytes(), "x", "image/png"));
        Assert.Equal(HttpStatusCode.Unauthorized, incorrect.StatusCode);
    }

    [Fact]
    public async Task Convert_RecoverableInferenceFailureReturnsExactOriginal()
    {
        using var client = AuthenticatedClient();
        var original = PngBytes().Append((byte)33).ToArray();
        _factory.Backend.FailNext = true;
        using var response = await client.PostAsync("/convert", Multipart(original, "fault", "image/png"));
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("fallback", response.Headers.GetValues("X-Upscaler-Result").Single());
        Assert.Equal(original, await response.Content.ReadAsByteArrayAsync());
    }

    private HttpClient AuthenticatedClient()
    {
        var client = _factory.CreateClient();
        client.DefaultRequestHeaders.Add("X-Upscaler-Token", "integration-secret");
        return client;
    }

    private static MultipartFormDataContent Multipart(byte[] bytes, string filename, string declaredMime)
    {
        var result = new MultipartFormDataContent();
        var image = new ByteArrayContent(bytes); image.Headers.ContentType = MediaTypeHeaderValue.Parse(declaredMime);
        result.Add(image, "image", filename); return result;
    }

    private static byte[] PngBytes() => [137, 80, 78, 71, 13, 10, 26, 10, 1, 2, 3, 4];
}

public sealed class KomaFactory : WebApplicationFactory<Program>, IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "komascaler-integration-" + Guid.NewGuid().ToString("N"));
    public FakeGpu Backend { get; } = new();

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Testing");
        Directory.CreateDirectory(_root);
        CreateInventory();
        builder.ConfigureAppConfiguration((_, configuration) => configuration.AddInMemoryCollection(new Dictionary<string, string?>(StringComparer.Ordinal)
        {
            ["Upscaling:Models:Directory"] = Path.Combine(_root, "models"),
            ["Upscaling:Models:InventoryFile"] = "models.production.json",
            ["Upscaling:Cache:Directory"] = Path.Combine(_root, "cache"),
            ["Upscaling:Queue:ResponseDeadline"] = "00:00:05"
        }));
        builder.ConfigureServices(services =>
        {
            services.RemoveAll<UpscalingOptions>();
            services.AddSingleton(new UpscalingOptions
            {
                Models = new ModelOptions { Directory = Path.Combine(_root, "models"), InventoryFile = "models.production.json" },
                Cache = new CacheOptions { Directory = Path.Combine(_root, "cache") },
                Queue = new QueueOptions { Capacity = 8, AdmissionTimeout = TimeSpan.FromMilliseconds(100), ResponseDeadline = TimeSpan.FromSeconds(1) },
                Security = new SecurityOptions { Token = "integration-secret" }
            });
            services.RemoveAll<IResultCache>();
            services.AddSingleton<IResultCache>(_ => new FileResultCache(Path.Combine(_root, "cache")));
            services.RemoveAll<IImageProcessor>(); services.AddSingleton<IImageProcessor, FakeImages>();
            services.RemoveAll<IGpuUpscaler>(); services.AddSingleton<IGpuUpscaler>(Backend);
            services.RemoveAll<ITileInferenceBackend>(); services.AddSingleton<ITileInferenceBackend>(Backend);
            var idle = services.FirstOrDefault(x => x.ServiceType == typeof(IHostedService) && x.ImplementationType == typeof(GpuIdleService));
            if (idle is not null) services.Remove(idle);
        });
    }

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);
        if (disposing && Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private void CreateInventory()
    {
        var modelDirectory = Path.Combine(_root, "models"); Directory.CreateDirectory(modelDirectory);
        var models = new List<object>();
        var bands = new[] { (1200, 1, (int?)1250), (1300, 1251, 1350), (1400, 1351, 1450), (1500, 1451, 1550), (1600, 1551, 1760), (1920, 1761, 1984), (2048, 1985, null) };
        foreach (var (nominal, minimum, maximum) in bands)
        {
            var file = nominal + ".onnx"; var bytes = BitConverter.GetBytes(nominal); File.WriteAllBytes(Path.Combine(modelDirectory, file), bytes);
            models.Add(new { id = "model-" + nominal, nominalHeight = nominal, minimumHeight = minimum, maximumHeight = maximum, file, sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(), sourceFile = nominal + ".pth", sourceSha256 = new string('b', 64) });
        }
        var manifest = new
        {
            schemaVersion = 1,
            selectionPolicyVersion = "selection-v1",
            graphContract = new { opset = 17, batch = 1, inputName = "input", inputType = "float32", inputLayout = "NCHW", inputRange = new[] { 0d, 1d }, outputName = "output", outputType = "float32", outputLayout = "NCHW", scale = 2, minimumHeight = 4, minimumWidth = 4, dynamicSpatialDimensions = true, padding = "graph-owned" },
            runtime = new { onnxRuntimeVersion = "1.26.0", cudaMajor = 12, cudnnMajor = 9, precision = "fp32", tf32 = false, requiredExecutionProvider = "TensorrtExecutionProvider" },
            models
        };
        File.WriteAllText(Path.Combine(modelDirectory, "models.production.json"), JsonSerializer.Serialize(manifest));
    }
}

public sealed class FakeImages : IImageProcessor
{
    public ImageInspection Inspect(ReadOnlySpan<byte> encoded, long maxDecodedPixels, int monochromeThreshold)
    {
        var mime = KomaScaler.Cache.ImageSignatures.DetectMime(encoded);
        return mime is null ? new(false, false, "application/octet-stream", 0, 0, null, "unsupported") : new(true, true, mime, 2, 3, [0, 64, 128, 192, 224, 255], null);
    }
    public EncodedImage EncodeLossless(ReadOnlyMemory<byte> grayscale, int width, int height, OutputOptions options)
    {
        var bytes = "RIFFxxxxWEBP"u8.ToArray().Concat(BitConverter.GetBytes(width)).Concat(BitConverter.GetBytes(height)).Concat(grayscale.ToArray()).ToArray();
        return new(bytes, "image/webp");
    }
}

public sealed class FakeGpu : IGpuUpscaler
{
    private int _calls; private int _active; private int _maximum; private int _warmCalls;
    public int DelayMilliseconds { get; set; }
    public bool FailNext { get; set; }
    public int Calls => Volatile.Read(ref _calls);
    public int WarmCalls => Volatile.Read(ref _warmCalls);
    public GpuServiceState State => GpuServiceState.Ready;
    public DateTimeOffset LastActivity => DateTimeOffset.UtcNow;
    public int ActiveRuns => Volatile.Read(ref _active);
    public int MaximumObservedActiveRuns => Volatile.Read(ref _maximum);
    public Task WarmAsync(CancellationToken ct) { Interlocked.Increment(ref _warmCalls); return Task.CompletedTask; }
    public Task DrainIfIdleAsync(TimeSpan idleFor, CancellationToken ct) => Task.CompletedTask;
    public ValueTask DisposeAsync() => ValueTask.CompletedTask;

    public async Task<float[]> RunAsync(string modelId, float[] input, int height, int width, CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref _calls); var active = Interlocked.Increment(ref _active); _maximum = Math.Max(_maximum, active);
        try
        {
            if (FailNext) { FailNext = false; throw new InvalidOperationException("Injected recoverable inference failure."); }
            var delay = DelayMilliseconds;
            if (delay > 0) await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            var ow = width * 2; var oh = height * 2; var plane = ow * oh; var output = new float[plane * 3];
            for (var y = 0; y < oh; y++) for (var x = 0; x < ow; x++) { var value = input[((y / 2) * width) + (x / 2)]; var i = y * ow + x; output[i] = output[plane + i] = output[2 * plane + i] = value; }
            return output;
        }
        finally { Interlocked.Decrement(ref _active); }
    }
}
