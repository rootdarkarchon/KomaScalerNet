using KomaScaler.Cache;
using KomaScaler.Concurrency;

namespace KomaScaler.UnitTests;

public sealed class CacheAndConcurrencyTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "komascaler-tests-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task Cache_RoundTripsSupportedOutputsAndTreatsCorruptionAsMiss()
    {
        var cache = new FileResultCache(_directory); var key = new string('a', 64);
        var webp = "RIFFxxxxWEBPpayload"u8.ToArray();
        await cache.WriteAsync(key, webp, CancellationToken.None);
        Assert.Equal(webp, await cache.TryReadAsync(key, CancellationToken.None));
        await File.WriteAllBytesAsync(cache.PathFor(key), "not an image"u8.ToArray());
        Assert.Null(await cache.TryReadAsync(key, CancellationToken.None));
        var pngKey = new string('b', 64);
        var png = new byte[] { 137, 80, 78, 71, 13, 10, 26, 10, 1 };
        await cache.WriteAsync(pngKey, png, CancellationToken.None);
        Assert.Equal(png, await cache.TryReadAsync(pngKey, CancellationToken.None));
    }

    [Fact]
    public async Task InFlight_ConcurrentCallersExecuteOneFactory()
    {
        var registry = new InFlightRegistry(); var calls = 0;
        var gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task<SharedResult> Factory()
        {
            Interlocked.Increment(ref calls);
            return FinishAsync();
        }
        async Task<SharedResult> FinishAsync() { await gate.Task.ConfigureAwait(false); return new("RIFFxxxxWEBP"u8.ToArray(), "image/webp"); }
        var registrations = Enumerable.Range(0, 20).Select(_ => registry.GetOrAdd("key", Factory)).ToArray();
        gate.SetResult();
        await Task.WhenAll(registrations.Select(x => x.Task));
        Assert.Equal(1, calls);
        Assert.Single(registrations, x => x.IsProducer);
        Assert.Equal(0, registry.Count);
    }

    [Fact]
    public async Task RunPermits_DrainWaitsForEveryActiveLeaseAndBlocksNewWork()
    {
        using var permits = new RunPermitSet(2);
        using var first = await permits.EnterAsync(CancellationToken.None);
        using var second = await permits.EnterAsync(CancellationToken.None);
        var drain = permits.DrainAsync(CancellationToken.None).AsTask();
        await Task.Delay(20);
        Assert.False(drain.IsCompleted);
        first.Dispose();
        await Task.Delay(20);
        Assert.False(drain.IsCompleted);
        second.Dispose();
        using var drained = await drain.WaitAsync(TimeSpan.FromSeconds(1));
        var waiting = permits.EnterAsync(CancellationToken.None).AsTask();
        await Task.Delay(20);
        Assert.False(waiting.IsCompleted);
        drained.Dispose();
        using var resumed = await waiting.WaitAsync(TimeSpan.FromSeconds(1));
    }

    [Fact]
    public async Task RunPermits_CancelledPartialDrainRestoresEveryPermitAndDoubleReleaseIsSafe()
    {
        using var permits = new RunPermitSet(2);
        using var active = await permits.EnterAsync(CancellationToken.None);
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(30));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => permits.DrainAsync(cancellation.Token).AsTask());
        active.Dispose();
        using var first = await permits.EnterAsync(CancellationToken.None);
        using var second = await permits.EnterAsync(CancellationToken.None);
        first.Dispose();
        first.Dispose();
    }

    [Fact]
    public void RunPermits_DoubleDisposeIsSafe()
    {
        var permits = new RunPermitSet(1);
        permits.Dispose();
        permits.Dispose();
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
        GC.SuppressFinalize(this);
    }
}
