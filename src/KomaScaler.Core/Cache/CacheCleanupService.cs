using KomaScaler.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;

namespace KomaScaler.Cache;

public sealed class CacheCleanupService(IOptions<UpscalingOptions> options) : BackgroundService
{
    private readonly CacheOptions _options = options.Value.Cache;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(_options.CleanupInterval);
        do { await CleanupAsync(stoppingToken).ConfigureAwait(false); }
        while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false));
    }

    internal Task CleanupAsync(CancellationToken ct) => Task.Run(() => Cleanup(ct), ct);

    private void Cleanup(CancellationToken ct)
    {
        if (!Directory.Exists(_options.Directory)) return;
        var now = DateTime.UtcNow;
        var files = Directory.EnumerateFiles(_options.Directory, "*", SearchOption.AllDirectories)
            .Where(path => path.EndsWith(".image", StringComparison.Ordinal) || path.EndsWith(".webp", StringComparison.Ordinal))
            .Select(path => new FileInfo(path)).OrderBy(file => file.LastWriteTimeUtc).ToList();
        long total = files.Sum(file => file.Length);
        foreach (var file in files)
        {
            ct.ThrowIfCancellationRequested();
            if (now - file.LastWriteTimeUtc <= _options.MaximumAge && total <= _options.MaximumBytes) continue;
            try { total -= file.Length; file.Delete(); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }
}
