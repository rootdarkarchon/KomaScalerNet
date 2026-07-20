namespace KomaScaler.Concurrency;

public sealed class RunPermitSet : IDisposable
{
    private readonly SemaphoreSlim _semaphore;
    private readonly int _capacity;
    private int _disposed;

    public RunPermitSet(int capacity)
    {
        if (capacity <= 0) throw new ArgumentOutOfRangeException(nameof(capacity));
        _capacity = capacity;
        _semaphore = new(capacity, capacity);
    }

    public async ValueTask<IDisposable> EnterAsync(CancellationToken ct)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        await _semaphore.WaitAsync(ct).ConfigureAwait(false);
        return new Lease(_semaphore, 1);
    }

    public async ValueTask<IDisposable> DrainAsync(CancellationToken ct)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        var acquired = 0;
        try
        {
            for (; acquired < _capacity; acquired++) await _semaphore.WaitAsync(ct).ConfigureAwait(false);
            return new Lease(_semaphore, _capacity);
        }
        catch
        {
            if (acquired > 0) _semaphore.Release(acquired);
            throw;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0) _semaphore.Dispose();
    }

    private sealed class Lease(SemaphoreSlim semaphore, int permits) : IDisposable
    {
        private int _released;
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) == 0) semaphore.Release(permits);
        }
    }
}
