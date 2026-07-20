using Microsoft.Extensions.Options;
using Microsoft.Extensions.Logging;

namespace KomaScaler.Configuration;

public sealed record TilingPolicy(int MaximumCoreSize, int ContextPixelsPerSide)
{
    public const string PartitionVersion = "balanced-v1";
    public const string BlendVersion = "halfsine-v1";
    public int MaximumTileExtent => checked(MaximumCoreSize + (2 * ContextPixelsPerSide));
}

public static class OptionsValidation
{
    public const int MaximumTileExtent = 960;

    public static string? ValidateTiling(TilingOptions value)
    {
        if (value.MaximumCoreSize <= 0 || value.ContextPixelsPerSide <= 0 ||
            value.LowMemoryMaximumCoreSize <= 0 || value.LowMemoryContextPixelsPerSide <= 0)
            return "All tiling values must be positive.";
        if (value.ContextPixelsPerSide >= value.MaximumCoreSize ||
            value.LowMemoryContextPixelsPerSide >= value.LowMemoryMaximumCoreSize)
            return "Context must be smaller than its corresponding maximum core size.";
        if (value.MaximumCoreSize + (2 * value.ContextPixelsPerSide) > MaximumTileExtent ||
            value.LowMemoryMaximumCoreSize + (2 * value.LowMemoryContextPixelsPerSide) > MaximumTileExtent)
            return $"The maximum tile extent must not exceed {MaximumTileExtent} pixels.";
        return null;
    }
}

public interface ITilingPolicyProvider
{
    TilingPolicy Snapshot();
    string? ValidationError { get; }
}

public sealed class TilingPolicyProvider : ITilingPolicyProvider, IDisposable
{
    private readonly IDisposable? _changeRegistration;
    private TilingPolicy _current;
    private string? _validationError;

    private readonly ILogger<TilingPolicyProvider> _logger;

    public TilingPolicyProvider(IOptionsMonitor<TilingOptions> monitor, ILogger<TilingPolicyProvider> logger)
    {
        _logger = logger;
        var error = OptionsValidation.ValidateTiling(monitor.CurrentValue);
        if (error is not null) throw new OptionsValidationException(nameof(TilingOptions), typeof(TilingOptions), [error]);
        _current = Create(monitor.CurrentValue);
        _changeRegistration = monitor.OnChange(Publish);
    }

    public string? ValidationError => Volatile.Read(ref _validationError);
    public TilingPolicy Snapshot() => Volatile.Read(ref _current);
    public void Dispose() => _changeRegistration?.Dispose();

    private void Publish(TilingOptions options)
    {
        var error = OptionsValidation.ValidateTiling(options);
        if (error is not null)
        {
            Volatile.Write(ref _validationError, error);
            _logger.LogWarning("Rejected invalid tiling policy reload: {ValidationError}", error);
            return;
        }
        Volatile.Write(ref _current, Create(options));
        Volatile.Write(ref _validationError, null);
        _logger.LogInformation("Published tiling policy core={MaximumCoreSize} context={ContextPixelsPerSide}", options.MaximumCoreSize, options.ContextPixelsPerSide);
    }

    private static TilingPolicy Create(TilingOptions value) => new(value.MaximumCoreSize, value.ContextPixelsPerSide);
}
