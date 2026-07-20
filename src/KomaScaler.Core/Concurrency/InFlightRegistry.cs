using System.Collections.Concurrent;

namespace KomaScaler.Concurrency;

public sealed record SharedResult(byte[] Bytes, string ContentType);
public sealed record Registration(Task<SharedResult> Task, bool IsProducer);

public sealed class InFlightRegistry
{
    private readonly ConcurrentDictionary<string, Lazy<Task<SharedResult>>> _work = new(StringComparer.Ordinal);

    public Registration GetOrAdd(string key, Func<Task<SharedResult>> factory)
    {
        var candidate = new Lazy<Task<SharedResult>>(() => RunAndRemoveAsync(key, factory), LazyThreadSafetyMode.ExecutionAndPublication);
        var actual = _work.GetOrAdd(key, candidate);
        return new(actual.Value, ReferenceEquals(candidate, actual));
    }

    public int Count => _work.Count;

    private async Task<SharedResult> RunAndRemoveAsync(string key, Func<Task<SharedResult>> factory)
    {
        try { return await factory().ConfigureAwait(false); }
        finally
        {
            if (_work.TryGetValue(key, out var current))
                _work.TryRemove(new KeyValuePair<string, Lazy<Task<SharedResult>>>(key, current));
        }
    }
}
