namespace KomaScaler.Configuration;

public sealed class UpscalingOptions
{
    public const string SectionName = "Upscaling";
    public ModelOptions Models { get; init; } = new();
    public TilingOptions Tiling { get; init; } = new();
    public QueueOptions Queue { get; init; } = new();
    public InputOptions Input { get; init; } = new();
    public OutputOptions Output { get; init; } = new();
    public CacheOptions Cache { get; init; } = new();
    public SecurityOptions Security { get; init; } = new();
}

public sealed record ModelOptions
{
    public string Directory { get; init; } = "models";
    public string InventoryFile { get; init; } = "models.production.json";
    public int CudaDeviceId { get; init; }
    public string TensorRtCacheDirectory { get; init; } = "/var/cache/komascaler/tensorrt";
    public string TensorRtVersion { get; init; } = "10.14";
    public ulong TensorRtMaxWorkspaceBytes { get; init; } = 2_147_483_648;
    public int TensorRtAuxiliaryStreams { get; init; }
    public bool TensorRtContextMemorySharing { get; init; } = true;
    public bool TensorRtTimingCache { get; init; } = true;
    public int TensorRtProfileOptimumExtent { get; init; } = 960;
    public int TensorRtProfileMaximumExtent { get; init; } = 960;
    public bool PreloadOnStartup { get; init; } = true;
    public TimeSpan IdleSessionUnloadAfter { get; init; } = TimeSpan.FromMinutes(10);
}

public static class TensorRtPolicy
{
    public static string CacheIdentity(ModelOptions options, string onnxRuntimeVersion) => string.Join('|',
            "tensorrt-only-v2", $"ort={onnxRuntimeVersion}", $"trt={options.TensorRtVersion}", $"device={options.CudaDeviceId}",
            "fp16=0", "int8=0", "tf32=0", $"workspace={options.TensorRtMaxWorkspaceBytes}",
            $"aux={options.TensorRtAuxiliaryStreams}", $"contextSharing={options.TensorRtContextMemorySharing}",
            $"timingCache={options.TensorRtTimingCache}", "sessions=1", "runs=1", "cudaFallback=0",
            $"profile=input:1x3x8x8-1x3x{options.TensorRtProfileOptimumExtent}x{options.TensorRtProfileOptimumExtent}-1x3x{options.TensorRtProfileMaximumExtent}x{options.TensorRtProfileMaximumExtent}");

    public static string EngineCacheIdentity(ModelOptions options, string onnxRuntimeVersion) =>
        CacheIdentity(options, onnxRuntimeVersion);

    public static string? Validate(ModelOptions options)
    {
        if (options.TensorRtMaxWorkspaceBytes == 0) return "TensorRtMaxWorkspaceBytes must be positive.";
        if (options.TensorRtAuxiliaryStreams is < 0 or > 16) return "TensorRtAuxiliaryStreams must be between 0 and 16.";
        if (options.TensorRtProfileOptimumExtent is < 8 or > OptionsValidation.MaximumTileExtent ||
            options.TensorRtProfileMaximumExtent is < 8 or > OptionsValidation.MaximumTileExtent ||
            options.TensorRtProfileOptimumExtent > options.TensorRtProfileMaximumExtent)
            return $"TensorRT profile extents must satisfy 8 <= optimum <= maximum <= {OptionsValidation.MaximumTileExtent}.";
        if (string.IsNullOrWhiteSpace(options.TensorRtCacheDirectory) || string.IsNullOrWhiteSpace(options.TensorRtVersion))
            return "TensorRT cache directory and version identity are required.";
        return null;
    }
}

public sealed class TilingOptions
{
    public int MaximumCoreSize { get; init; } = 832;
    public int ContextPixelsPerSide { get; init; } = 64;
    public int LowMemoryMaximumCoreSize { get; init; } = 192;
    public int LowMemoryContextPixelsPerSide { get; init; } = 16;
}

public sealed class QueueOptions
{
    public int Capacity { get; init; } = 8;
    public TimeSpan AdmissionTimeout { get; init; } = TimeSpan.FromMilliseconds(100);
    public TimeSpan ResponseDeadline { get; init; } = TimeSpan.FromSeconds(28);
}

public sealed class InputOptions
{
    public long MaxUploadBytes { get; init; } = 67_108_864;
    public long MaxDecodedPixels { get; init; } = 80_000_000;
    public int MonochromeThreshold { get; init; } = 12;
}

public sealed class OutputOptions
{
    public string Format { get; init; } = "png";
    public bool Lossless { get; init; } = true;
    public int Quality { get; init; } = 100;
    public int Effort { get; init; } = 2;
    public int PngCompression { get; init; } = 3;
}

public sealed class CacheOptions
{
    public string Directory { get; init; } = "cache";
    public long MaximumBytes { get; init; } = 107_374_182_400;
    public TimeSpan MaximumAge { get; init; } = TimeSpan.FromDays(90);
    public TimeSpan CleanupInterval { get; init; } = TimeSpan.FromHours(1);
}

public sealed class SecurityOptions
{
    public string TokenHeader { get; init; } = "X-Upscaler-Token";
    public string Token { get; init; } = string.Empty;
}
