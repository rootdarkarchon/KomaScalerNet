using System.Collections.ObjectModel;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using KomaScaler.Concurrency;
using KomaScaler.Configuration;
using KomaScaler.Models;
using KomaScaler.Tiling;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;

namespace KomaScaler.Inference;

public enum GpuServiceState { Uninitialized, Loading, Ready, Draining, Faulted }

public interface IGpuUpscaler : ITileInferenceBackend, IAsyncDisposable
{
    GpuServiceState State { get; }
    DateTimeOffset LastActivity { get; }
    int ActiveRuns { get; }
    int MaximumObservedActiveRuns { get; }
    Task WarmAsync(CancellationToken ct);
    Task DrainIfIdleAsync(TimeSpan idleFor, CancellationToken ct);
}

public sealed class GpuUpscaler(
    InventoryState inventoryState,
    IOptions<UpscalingOptions> options,
    ILogger<GpuUpscaler> logger,
    IHostApplicationLifetime lifetime) : IGpuUpscaler
{
    private readonly RunPermitSet _runPermits = new(1);
    private readonly SemaphoreSlim _lifecycleGate = new(1, 1);
    private readonly string _modelDirectory = options.Value.Models.Directory;
    private readonly int _deviceId = options.Value.Models.CudaDeviceId;
    private readonly ModelOptions _modelOptions = options.Value.Models;
    private IReadOnlyDictionary<string, SessionPool>? _sessions;
    private int _state;
    private int _activeRuns;
    private int _maximumObservedActiveRuns;
    private int _disposed;
    private int _tensorRtPrewarming;
    private int _tensorRtStartupWarmComplete;
    private long _lastActivityTicks = DateTimeOffset.UtcNow.UtcTicks;

    public GpuServiceState State => (GpuServiceState)Volatile.Read(ref _state);
    public DateTimeOffset LastActivity => new(Volatile.Read(ref _lastActivityTicks), TimeSpan.Zero);
    public int ActiveRuns => Volatile.Read(ref _activeRuns);
    public int MaximumObservedActiveRuns => Volatile.Read(ref _maximumObservedActiveRuns);

    public async Task WarmAsync(CancellationToken ct)
    {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var inventory = GetInventory();
        var input = new float[3 * 8 * 8];
        Volatile.Write(ref _tensorRtPrewarming, 1);
        try
        {
            foreach (var model in inventory.Models)
                _ = await RunAsync(model.Id, input, 8, 8, ct).ConfigureAwait(false);
            ValidateCompleteTensorRtEngineCache(inventory);
            Volatile.Write(ref _tensorRtStartupWarmComplete, 1);
            await DrainIfIdleAsync(TimeSpan.Zero, ct).ConfigureAwait(false);
        }
        finally { Volatile.Write(ref _tensorRtPrewarming, 0); }
        timer.Stop();
        logger.LogInformation("Pre-warmed {ModelCount} {ExecutionProvider} models sequentially in {ElapsedMs:F3} ms; residentAfterWarm={ResidentAfterWarm}",
            inventory.Models.Count, "TensorRt", timer.Elapsed.TotalMilliseconds, "none-active-model");
    }

    public async Task<float[]> RunAsync(string modelId, float[] nchwRgb, int height, int width, CancellationToken cancellationToken)
    {
        if (State == GpuServiceState.Faulted) throw new InvalidOperationException("GPU service is faulted.");
        IDisposable? permit = null;
        SessionPool? pool = null;
        while (permit is null)
        {
            await EnsureLoadedAsync(modelId, cancellationToken).ConfigureAwait(false);
            var candidate = await _runPermits.EnterAsync(cancellationToken).ConfigureAwait(false);
            if (State == GpuServiceState.Ready && _sessions is not null && _sessions.TryGetValue(modelId, out pool)) permit = candidate;
            else candidate.Dispose();
        }
        using (permit)
        {
            if (State != GpuServiceState.Ready || _sessions is null) throw new InvalidOperationException("GPU session set is not ready.");
            try
            {
                using var input = OrtValue.CreateTensorValueFromMemory(nchwRgb, [1, 3, height, width]);
                using var runOptions = new RunOptions();
                var inventory = GetInventory();
                var runTimer = System.Diagnostics.Stopwatch.StartNew();
                var slot = await pool!.RentAsync(cancellationToken).ConfigureAwait(false);
                try
                {
                    var active = Interlocked.Increment(ref _activeRuns);
                    UpdateMaximum(active);
                    Touch();
                    using var output = slot.Session.Run(runOptions, [inventory.GraphContract.InputName], [input], [inventory.GraphContract.OutputName]);
                    runTimer.Stop();
                    var copyTimer = System.Diagnostics.Stopwatch.StartNew();
                    var span = output[0].GetTensorDataAsSpan<float>();
                    var expected = checked(3 * height * 2 * width * 2);
                    if (span.Length != expected) throw new InvalidDataException("ONNX output shape was not exact NCHW 2x RGB.");
                    var result = span.ToArray();
                    copyTimer.Stop();
                    pool.ValidatePlacementAfterFirstRun(logger, modelId, slot.Profiles);
                    logger.LogDebug("{ExecutionProvider} tile timing model={ModelId} input={Width}x{Height} runMs={RunMs:F3} outputCopyMs={OutputCopyMs:F3}",
                        "TensorRt", modelId, width, height, runTimer.Elapsed.TotalMilliseconds, copyTimer.Elapsed.TotalMilliseconds);
                    return result;
                }
                finally { Interlocked.Decrement(ref _activeRuns); Touch(); pool.Return(slot); }
            }
            catch (Exception ex) when (ex is OnnxRuntimeException or InvalidOperationException)
            {
                Volatile.Write(ref _state, (int)GpuServiceState.Faulted);
                logger.LogError(ex, "TensorRT inference fault; GPU service entered {State}", GpuServiceState.Faulted);
                _ = Task.Run(DisposeFaultedSessionsAsync);
                throw;
            }
        }
    }

    public async Task DrainIfIdleAsync(TimeSpan idleFor, CancellationToken ct)
    {
        if (State != GpuServiceState.Ready || ActiveRuns != 0 || DateTimeOffset.UtcNow - LastActivity < idleFor) return;
        await _lifecycleGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (State != GpuServiceState.Ready || ActiveRuns != 0 || DateTimeOffset.UtcNow - LastActivity < idleFor) return;
            Volatile.Write(ref _state, (int)GpuServiceState.Draining);
            using (await _runPermits.DrainAsync(ct).ConfigureAwait(false))
            { DisposeSessions(); Volatile.Write(ref _state, (int)GpuServiceState.Uninitialized); Touch(); }
            logger.LogInformation("Disposed idle TensorRT session; ASP.NET remains live");
        }
        finally { _lifecycleGate.Release(); }
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        await _lifecycleGate.WaitAsync().ConfigureAwait(false);
        try
        {
            Volatile.Write(ref _state, (int)GpuServiceState.Draining);
            using (await _runPermits.DrainAsync(CancellationToken.None).ConfigureAwait(false)) DisposeSessions();
        }
        finally { _lifecycleGate.Release(); _lifecycleGate.Dispose(); _runPermits.Dispose(); }
    }

    private async Task EnsureLoadedAsync(string modelId, CancellationToken ct)
    {
        if (State == GpuServiceState.Ready && _sessions?.ContainsKey(modelId) == true) return;
        await _lifecycleGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (State == GpuServiceState.Ready && _sessions?.ContainsKey(modelId) == true) return;
            if (State == GpuServiceState.Faulted) throw new InvalidOperationException("GPU service is faulted.");
            Volatile.Write(ref _state, (int)GpuServiceState.Loading);
            var inventory = GetInventory();
            var available = OrtEnv.Instance().GetAvailableProviders();
            if (!available.Contains("TensorrtExecutionProvider", StringComparer.Ordinal))
            {
                Volatile.Write(ref _state, (int)GpuServiceState.Faulted);
                throw new InvalidOperationException("TensorrtExecutionProvider is unavailable; fallback inference is forbidden.");
            }
            var model = inventory.Models.Single(item => string.Equals(item.Id, modelId, StringComparison.Ordinal));
            EnsureTensorRtRequestCannotBuildEngine(model, inventory);
            try
            {
                using (await _runPermits.DrainAsync(ct).ConfigureAwait(false))
                {
                    var previousModel = _sessions?.Keys.SingleOrDefault();
                    var disposeTimer = System.Diagnostics.Stopwatch.StartNew();
                    DisposeSessions();
                    disposeTimer.Stop();
                    var loadTimer = System.Diagnostics.Stopwatch.StartNew();
                    try
                    {
                        var pool = CreateSessionPool(model, inventory);
                        _sessions = new ReadOnlyDictionary<string, SessionPool>(
                            new Dictionary<string, SessionPool>(StringComparer.Ordinal) { [model.Id] = pool });
                    }
                    catch
                    {
                        Volatile.Write(ref _state, (int)GpuServiceState.Faulted);
                        throw;
                    }
                    loadTimer.Stop();
                    Volatile.Write(ref _state, (int)GpuServiceState.Ready);
                    Touch();
                    logger.LogInformation("Switched TensorRT active model previous={PreviousModel} current={CurrentModel} disposeMs={DisposeMs:F3} cachedLoadMs={LoadMs:F3}",
                        previousModel ?? "none", model.Id, disposeTimer.Elapsed.TotalMilliseconds, loadTimer.Elapsed.TotalMilliseconds);
                }
            }
            catch (OperationCanceledException)
            {
                Volatile.Write(ref _state, (int)(_sessions is null ? GpuServiceState.Uninitialized : GpuServiceState.Ready));
                throw;
            }
        }
        finally { _lifecycleGate.Release(); }
    }

    private SessionPool CreateSessionPool(ModelDefinition model, ModelInventory inventory)
    {
        var sessions = new List<InferenceSession>(1);
        try
        {
            sessions.Add(CreateSession(model, inventory, profile: true));
            return new SessionPool(sessions);
        }
        catch { foreach (var session in sessions) session.Dispose(); throw; }
    }

    private InferenceSession CreateSession(ModelDefinition model, ModelInventory inventory, bool profile)
    {
        using var sessionOptions = CreateSessionOptions(model, inventory, profile);
        return new InferenceSession(Path.Combine(_modelDirectory, model.File), sessionOptions);
    }

    private Microsoft.ML.OnnxRuntime.SessionOptions CreateSessionOptions(ModelDefinition model, ModelInventory inventory, bool profile)
    {
        var result = new Microsoft.ML.OnnxRuntime.SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        if (profile)
        {
            var profileDirectory = Path.Combine(GetTensorRtIdentityDirectory(model, inventory), "profiles");
            Directory.CreateDirectory(profileDirectory);
            result.ProfileOutputPathPrefix = Path.Combine(profileDirectory, model.Id);
            result.EnableProfiling = true;
        }
        var cacheDirectory = GetTensorRtIdentityDirectory(model, inventory);
        Directory.CreateDirectory(cacheDirectory);
        using var tensorRt = new OrtTensorRTProviderOptions();
        tensorRt.UpdateOptions(new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["device_id"] = Invariant(_deviceId),
            ["trt_fp16_enable"] = "0",
            ["trt_int8_enable"] = "0",
            ["trt_engine_cache_enable"] = "1",
            ["trt_engine_cache_path"] = cacheDirectory,
            ["trt_engine_cache_prefix"] = model.Sha256[..16],
            ["trt_timing_cache_enable"] = _modelOptions.TensorRtTimingCache ? "1" : "0",
            ["trt_timing_cache_path"] = cacheDirectory,
            ["trt_force_timing_cache"] = "0",
            ["trt_force_sequential_engine_build"] = "1",
            ["trt_auxiliary_streams"] = Invariant(_modelOptions.TensorRtAuxiliaryStreams),
            ["trt_context_memory_sharing_enable"] = _modelOptions.TensorRtContextMemorySharing ? "1" : "0",
            ["trt_max_workspace_size"] = Invariant(_modelOptions.TensorRtMaxWorkspaceBytes),
            ["trt_profile_min_shapes"] = $"{inventory.GraphContract.InputName}:1x3x8x8",
            ["trt_profile_opt_shapes"] = $"{inventory.GraphContract.InputName}:1x3x{_modelOptions.TensorRtProfileOptimumExtent}x{_modelOptions.TensorRtProfileOptimumExtent}",
            ["trt_profile_max_shapes"] = $"{inventory.GraphContract.InputName}:1x3x{_modelOptions.TensorRtProfileMaximumExtent}x{_modelOptions.TensorRtProfileMaximumExtent}"
        });
        result.AppendExecutionProvider_Tensorrt(tensorRt);
        return result;
    }

    private string GetTensorRtIdentityDirectory(ModelDefinition model, ModelInventory inventory)
    {
        var material = $"{model.Sha256}|{TensorRtPolicy.EngineCacheIdentity(_modelOptions, inventory.Runtime.OnnxRuntimeVersion)}";
        var identity = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(material))).ToLowerInvariant();
        return Path.Combine(_modelOptions.TensorRtCacheDirectory, identity);
    }

    private bool HasTensorRtEngine(ModelDefinition model, ModelInventory inventory)
    {
        var directory = GetTensorRtIdentityDirectory(model, inventory);
        return Directory.Exists(directory) && Directory.EnumerateFiles(directory, "*.engine", SearchOption.TopDirectoryOnly).Any();
    }

    private void EnsureTensorRtRequestCannotBuildEngine(ModelDefinition model, ModelInventory inventory)
    {
        if (!_modelOptions.PreloadOnStartup || Volatile.Read(ref _tensorRtPrewarming) != 0) return;
        if (Volatile.Read(ref _tensorRtStartupWarmComplete) == 0)
            RejectRequestTimeEngineBuild("TensorRT startup engine preloading has not completed; request-time engine construction is forbidden.");
        if (!HasTensorRtEngine(model, inventory))
            RejectRequestTimeEngineBuild($"The production TensorRT engine for {model.Id} is missing or incompatible; request-time engine construction is forbidden. Restart to rebuild the complete cache before readiness.");
    }

    private void RejectRequestTimeEngineBuild(string message)
    {
        Volatile.Write(ref _state, (int)GpuServiceState.Faulted);
        Environment.ExitCode = 70;
        logger.LogCritical("{Message} Stopping with exit code {ExitCode}.", message, 70);
        lifetime.StopApplication();
        throw new InvalidOperationException(message);
    }

    private void ValidateCompleteTensorRtEngineCache(ModelInventory inventory)
    {
        var missing = inventory.Models.Where(model => !HasTensorRtEngine(model, inventory)).Select(model => model.Id).ToArray();
        if (missing.Length != 0)
            throw new InvalidOperationException("TensorRT startup did not produce compatible engines for: " + string.Join(", ", missing));
    }

    private static string Invariant<T>(T value) where T : IFormattable => value.ToString(null, System.Globalization.CultureInfo.InvariantCulture);

    private async Task DisposeFaultedSessionsAsync()
    {
        await _lifecycleGate.WaitAsync().ConfigureAwait(false);
        try { using (await _runPermits.DrainAsync(CancellationToken.None).ConfigureAwait(false)) DisposeSessions(); }
        finally { _lifecycleGate.Release(); }
        Environment.ExitCode = 70;
        logger.LogCritical("TensorRT state is unverified after a fault; stopping with exit code {ExitCode} for systemd restart", 70);
        lifetime.StopApplication();
    }

    private void DisposeSessions()
    {
        if (_sessions is not null) foreach (var pool in _sessions.Values) pool.Dispose();
        _sessions = null;
    }
    private void Touch() => Volatile.Write(ref _lastActivityTicks, DateTimeOffset.UtcNow.UtcTicks);
    private ModelInventory GetInventory() => inventoryState.Inventory ?? throw new InvalidOperationException("Model inventory is unavailable.");
    private void UpdateMaximum(int active)
    {
        int observed;
        while (active > (observed = Volatile.Read(ref _maximumObservedActiveRuns)) &&
            Interlocked.CompareExchange(ref _maximumObservedActiveRuns, active, observed) != observed) { }
    }

    private sealed class SessionPool : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly SemaphoreSlim _available = new(1, 1);
        private readonly Lock _placementLock = new();
        private bool _placementValidated;
        private Exception? _placementFailure;
        private int _disposed;

        public SessionPool(List<InferenceSession> sessions) => _session = sessions.Single();

        public async ValueTask<SessionSlot> RentAsync(CancellationToken ct)
        {
            await _available.WaitAsync(ct).ConfigureAwait(false);
            return new SessionSlot(_session, true);
        }
        public void Return(SessionSlot session) => _available.Release();

        public void ValidatePlacementAfterFirstRun(ILogger logger, string modelId, bool profileOwner)
        {
            if (!profileOwner) return;
            lock (_placementLock)
            {
                if (_placementFailure is not null) throw new InvalidOperationException("TensorRT placement validation previously failed.", _placementFailure);
                if (_placementValidated) return;
                try
                {
                    var profilePath = _session.EndProfiling();
                    using var profile = JsonDocument.Parse(File.ReadAllBytes(profilePath));
                    var counts = profile.RootElement.EnumerateArray()
                        .Where(item => item.TryGetProperty("cat", out var category) && string.Equals(category.GetString(), "Node", StringComparison.Ordinal) &&
                            item.TryGetProperty("args", out var arguments) && arguments.TryGetProperty("provider", out _))
                        .GroupBy(item => item.GetProperty("args").GetProperty("provider").GetString() ?? "unknown", StringComparer.Ordinal)
                        .ToDictionary(group => group.Key, group => group.Count(), StringComparer.Ordinal);
                    var cpuCompute = profile.RootElement.EnumerateArray().Count(item =>
                        item.TryGetProperty("cat", out var category) && string.Equals(category.GetString(), "Node", StringComparison.Ordinal) &&
                        item.TryGetProperty("args", out var arguments) &&
                        arguments.TryGetProperty("provider", out var provider) && string.Equals(provider.GetString(), "CPUExecutionProvider", StringComparison.Ordinal) &&
                        arguments.TryGetProperty("op_name", out var operation) && operation.GetString() is "Conv" or "Gemm" or "MatMul");
                    counts.TryGetValue("TensorrtExecutionProvider", out var tensorRtNodes);
                    counts.TryGetValue("CUDAExecutionProvider", out var cudaNodes);
                    counts.TryGetValue("CPUExecutionProvider", out var cpuNodes);
                    logger.LogInformation("Provider placement model={ModelId} TensorRtNodes={TensorRtNodes} CudaFallbackNodes={CudaNodes} CpuNodes={CpuNodes} CpuComputeNodes={CpuComputeNodes} Profile={Profile}",
                        modelId, tensorRtNodes, cudaNodes, cpuNodes, cpuCompute, profilePath);
                    if (tensorRtNodes == 0 || cudaNodes != 0 || cpuCompute != 0)
                        throw new InvalidOperationException($"TensorRT placement validation failed for {modelId}: TensorRT={tensorRtNodes}, CUDA={cudaNodes}, CPU={cpuNodes}, CPU compute={cpuCompute}.");
                    _placementValidated = true;
                }
                catch (Exception ex) { _placementFailure = ex; throw; }
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
            _session.Dispose();
            _available.Dispose();
        }

        public sealed record SessionSlot(InferenceSession Session, bool Profiles);
    }
}

public sealed class GpuIdleService(IGpuUpscaler gpu, IOptions<UpscalingOptions> options) : BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(30));
        while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
            await gpu.DrainIfIdleAsync(options.Value.Models.IdleSessionUnloadAfter, stoppingToken).ConfigureAwait(false);
    }
}
