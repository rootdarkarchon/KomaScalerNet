using System.Threading.Channels;
using KomaScaler.Cache;
using KomaScaler.Configuration;
using KomaScaler.Concurrency;
using KomaScaler.Images;
using KomaScaler.Models;
using KomaScaler.Tiling;
using Microsoft.Extensions.Options;

namespace KomaScaler.Inference;

public sealed record ConversionWork(
    string CacheKey, ModelDefinition Model, byte[] Luminance, int Width, int Height,
    TilingPolicy Tiling, TaskCompletionSource<SharedResult> Completion);

public interface IConversionQueue
{
    int Depth { get; }
    Task<SharedResult> EnqueueAsync(string key, ModelDefinition model, byte[] luminance, int width, int height, TilingPolicy tiling);
}

public sealed class ConversionQueue : BackgroundService, IConversionQueue
{
    private readonly Channel<ConversionWork> _channel;
    private readonly QueueOptions _queueOptions;
    private readonly OutputOptions _outputOptions;
    private readonly TiledUpscaler _tiler;
    private readonly IImageProcessor _images;
    private readonly IResultCache _cache;
    private readonly ILogger<ConversionQueue> _logger;
    private int _depth;

    public ConversionQueue(
        IOptions<UpscalingOptions> options, ITileInferenceBackend backend,
        IImageProcessor images, IResultCache cache, ILogger<ConversionQueue> logger)
    {
        _queueOptions = options.Value.Queue;
        _outputOptions = options.Value.Output;
        _tiler = new(backend);
        _images = images;
        _cache = cache;
        _logger = logger;
        _channel = Channel.CreateBounded<ConversionWork>(new BoundedChannelOptions(_queueOptions.Capacity)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = false,
            AllowSynchronousContinuations = false
        });
    }

    public int Depth => Volatile.Read(ref _depth);

    public async Task<SharedResult> EnqueueAsync(string key, ModelDefinition model, byte[] luminance, int width, int height, TilingPolicy tiling)
    {
        var completion = new TaskCompletionSource<SharedResult>(TaskCreationOptions.RunContinuationsAsynchronously);
        var work = new ConversionWork(key, model, luminance, width, height, tiling, completion);
        using var timeout = new CancellationTokenSource(_queueOptions.AdmissionTimeout);
        try
        {
            await _channel.Writer.WriteAsync(work, timeout.Token).ConfigureAwait(false);
            Interlocked.Increment(ref _depth);
        }
        catch (OperationCanceledException) when (timeout.IsCancellationRequested)
        {
            throw new QueueAdmissionException();
        }
        return await completion.Task.ConfigureAwait(false);
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await foreach (var work in _channel.Reader.ReadAllAsync(stoppingToken).ConfigureAwait(false))
        {
            Interlocked.Decrement(ref _depth);
            try
            {
                var upscale = await _tiler.UpscaleMeasuredAsync(work.Model.Id, work.Luminance, work.Width, work.Height, work.Tiling, stoppingToken).ConfigureAwait(false);
                var encodeTimer = System.Diagnostics.Stopwatch.StartNew();
                var encoded = _images.EncodeLossless(upscale.Bytes, checked(work.Width * 2), checked(work.Height * 2), _outputOptions);
                encodeTimer.Stop();
                var cacheTimer = System.Diagnostics.Stopwatch.StartNew();
                await _cache.WriteAsync(work.CacheKey, encoded.Bytes, stoppingToken).ConfigureAwait(false);
                cacheTimer.Stop();
                _logger.LogInformation(
                    "Conversion timing model={ModelId} size={Width}x{Height} tiles={TileCount} wallMs={WallMs:F3} prepareMs={PrepareMs:F3} inferenceAggregateMs={InferenceAggregateMs:F3} blendMs={BlendMs:F3} normalizeMs={NormalizeMs:F3} encodeMs={EncodeMs:F3} cacheWriteMs={CacheWriteMs:F3}",
                    work.Model.Id, work.Width, work.Height, upscale.Timings.TileCount,
                    upscale.Timings.Wall.TotalMilliseconds, upscale.Timings.Prepare.TotalMilliseconds,
                    upscale.Timings.InferenceAggregate.TotalMilliseconds, upscale.Timings.Blend.TotalMilliseconds,
                    upscale.Timings.Normalize.TotalMilliseconds, encodeTimer.Elapsed.TotalMilliseconds,
                    cacheTimer.Elapsed.TotalMilliseconds);
                work.Completion.TrySetResult(new(encoded.Bytes, encoded.ContentType));
            }
            catch (Exception ex) when (ex is not OperationCanceledException || !stoppingToken.IsCancellationRequested)
            {
                _logger.LogError(ex, "Conversion job failed for model {ModelId}", work.Model.Id);
                work.Completion.TrySetException(ex);
            }
        }
    }

    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        _channel.Writer.TryComplete();
        await base.StopAsync(cancellationToken).ConfigureAwait(false);
        while (_channel.Reader.TryRead(out var work)) work.Completion.TrySetCanceled(cancellationToken);
    }
}

public sealed class QueueAdmissionException : Exception
{
    public QueueAdmissionException() : base("The bounded conversion queue did not admit work before its deadline.") { }
}
