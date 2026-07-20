using KomaScaler.Configuration;
using KomaScaler.Pipeline;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace KomaScaler.UnitTests;

public sealed class PolicyAndKeyTests
{
    [Theory]
    [InlineData(320, 32, true)]
    [InlineData(192, 16, true)]
    [InlineData(576, 32, true)]
    [InlineData(576, 64, true)]
    [InlineData(704, 64, true)]
    [InlineData(832, 64, true)]
    [InlineData(833, 64, false)]
    [InlineData(320, 0, false)]
    [InlineData(16, 16, false)]
    public void TilingValidation_EnforcesHardwareEnvelope(int core, int context, bool valid)
    {
        var options = new TilingOptions { MaximumCoreSize = core, ContextPixelsPerSide = context };
        Assert.Equal(valid, OptionsValidation.ValidateTiling(options) is null);
    }

    [Fact]
    public void PipelineKey_ChangesForEveryMaterialPolicyInput()
    {
        var model = TestInventory.Bands()[0];
        var baseline = new PipelineIdentity(new byte[32], model, "selection-v1", new(832, 64), "tensorrt-only-v2|tf32=0", "png", true, 100, 2, 3);
        var key = PipelineKey.Create(baseline);
        Assert.Equal(64, key.Length);
        Assert.NotEqual(key, PipelineKey.Create(baseline with { Tiling = new(192, 16) }));
        Assert.NotEqual(key, PipelineKey.Create(baseline with { SelectionPolicyVersion = "selection-v2" }));
        Assert.NotEqual(key, PipelineKey.Create(baseline with { InferenceIdentity = "tensorrt-only-v2|profile=8/704/704" }));
        Assert.NotEqual(key, PipelineKey.Create(baseline with { EncoderFormat = "webp" }));
        Assert.NotEqual(key, PipelineKey.Create(baseline with { EncoderQuality = 99 }));
        Assert.NotEqual(key, PipelineKey.Create(baseline with { EncoderEffort = 3 }));
        Assert.NotEqual(key, PipelineKey.Create(baseline with { EncoderPngCompression = 1 }));
        Assert.NotEqual(key, PipelineKey.Create(baseline with { SourceSha256 = Enumerable.Repeat((byte)1, 32).ToArray() }));
        Assert.NotEqual(key, PipelineKey.Create(baseline with { Model = model with { Sha256 = new string('c', 64) } }));
    }

    [Fact]
    public void TensorRtEngineOptions_ChangeProviderAndApplicationCacheIdentity()
    {
        var baseline = new ModelOptions();
        var identity = TensorRtPolicy.CacheIdentity(baseline, "1.26.0");
        Assert.NotEqual(identity, TensorRtPolicy.CacheIdentity(baseline with { TensorRtAuxiliaryStreams = 1 }, "1.26.0"));
        Assert.NotEqual(identity, TensorRtPolicy.CacheIdentity(baseline with { TensorRtContextMemorySharing = false }, "1.26.0"));
        Assert.NotEqual(identity, TensorRtPolicy.CacheIdentity(baseline with { TensorRtMaxWorkspaceBytes = 1_073_741_824 }, "1.26.0"));
        Assert.NotEqual(identity, TensorRtPolicy.CacheIdentity(baseline with { TensorRtVersion = "10.15" }, "1.26.0"));
        Assert.NotEqual(identity, TensorRtPolicy.CacheIdentity(baseline with { TensorRtProfileOptimumExtent = 640 }, "1.26.0"));
        Assert.NotEqual(identity, TensorRtPolicy.CacheIdentity(baseline with { TensorRtProfileMaximumExtent = 704 }, "1.26.0"));
    }

    [Fact]
    public void PolicyProvider_PublishesValidReloadAndRetainsLastGoodAfterInvalidReload()
    {
        var monitor = new MutableMonitor(new TilingOptions());
        using var provider = new TilingPolicyProvider(monitor, NullLogger<TilingPolicyProvider>.Instance);
        var pageSnapshot = provider.Snapshot();
        monitor.Publish(new TilingOptions { MaximumCoreSize = 192, ContextPixelsPerSide = 16 });
        Assert.Equal(new TilingPolicy(192, 16), provider.Snapshot());
        Assert.Equal(new TilingPolicy(832, 64), pageSnapshot);
        monitor.Publish(new TilingOptions { MaximumCoreSize = 929, ContextPixelsPerSide = 16 });
        Assert.Equal(new TilingPolicy(192, 16), provider.Snapshot());
        Assert.NotNull(provider.ValidationError);
    }

    private sealed class MutableMonitor(TilingOptions current) : IOptionsMonitor<TilingOptions>
    {
        private Action<TilingOptions, string?>? _listener;
        public TilingOptions CurrentValue { get; private set; } = current;
        public TilingOptions Get(string? name) => CurrentValue;
        public IDisposable? OnChange(Action<TilingOptions, string?> listener) { _listener = listener; return new Registration(() => _listener = null); }
        public void Publish(TilingOptions value) { CurrentValue = value; _listener?.Invoke(value, null); }
        private sealed class Registration(Action dispose) : IDisposable { public void Dispose() => dispose(); }
    }
}
