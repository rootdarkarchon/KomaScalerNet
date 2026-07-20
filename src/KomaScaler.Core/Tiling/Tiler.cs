using System.Runtime.InteropServices;
using KomaScaler.Configuration;

namespace KomaScaler.Tiling;

[StructLayout(LayoutKind.Sequential)]
public readonly record struct IntRect(int X, int Y, int Width, int Height)
{
    public int Right => checked(X + Width);
    public int Bottom => checked(Y + Height);
}

public sealed record TileRegion(IntRect Core, IntRect Input, int GridX, int GridY);

public static class BalancedTiler
{
    public static IReadOnlyList<TileRegion> Partition(int width, int height, TilingPolicy policy)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentNullException.ThrowIfNull(policy);
        var countX = DivideRoundUp(width, policy.MaximumCoreSize);
        var countY = DivideRoundUp(height, policy.MaximumCoreSize);
        var result = new List<TileRegion>(checked(countX * countY));

        for (var gy = 0; gy < countY; gy++)
        {
            var y0 = DivideRoundUp(gy * height, countY);
            var y1 = DivideRoundUp((gy + 1) * height, countY);
            for (var gx = 0; gx < countX; gx++)
            {
                var x0 = DivideRoundUp(gx * width, countX);
                var x1 = DivideRoundUp((gx + 1) * width, countX);
                var core = new IntRect(x0, y0, x1 - x0, y1 - y0);
                var inputX = Math.Max(0, x0 - policy.ContextPixelsPerSide);
                var inputY = Math.Max(0, y0 - policy.ContextPixelsPerSide);
                var inputRight = Math.Min(width, x1 + policy.ContextPixelsPerSide);
                var inputBottom = Math.Min(height, y1 + policy.ContextPixelsPerSide);
                result.Add(new(core, new(inputX, inputY, inputRight - inputX, inputBottom - inputY), gx, gy));
            }
        }

        return result;
    }

    private static int DivideRoundUp(int value, int divisor) => checked((value + divisor - 1) / divisor);
}

public static class HalfSineBlend
{
    public static float Weight(int coordinate, int inputLength, int coreStart, int coreLength)
    {
        if (coordinate < 0 || coordinate >= inputLength) throw new ArgumentOutOfRangeException(nameof(coordinate));
        var coreEnd = coreStart + coreLength;
        if (coordinate < coreStart && coreStart > 0)
        {
            var progress = (coordinate + 0.5f) / coreStart;
            return HalfSine(progress);
        }
        if (coordinate >= coreEnd && coreEnd < inputLength)
        {
            var progress = (inputLength - coordinate - 0.5f) / (inputLength - coreEnd);
            return HalfSine(progress);
        }
        return 1f;
    }

    public static float HalfSine(float value)
    {
        var clipped = Math.Clamp((2f * value) - 0.5f, 0f, 1f);
        return (MathF.Sin((MathF.PI * clipped) - (MathF.PI / 2f)) + 1f) / 2f;
    }
}

public interface ITileInferenceBackend
{
    Task<float[]> RunAsync(string modelId, float[] nchwRgb, int height, int width, CancellationToken cancellationToken);
}

public sealed record TiledUpscaleTimings(
    int TileCount, TimeSpan Wall, TimeSpan Prepare, TimeSpan InferenceAggregate,
    TimeSpan Blend, TimeSpan Normalize);

public sealed record TiledUpscaleResult(byte[] Bytes, TiledUpscaleTimings Timings);

public sealed class TiledUpscaler
{
    private readonly ITileInferenceBackend _backend;
    public TiledUpscaler(ITileInferenceBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _backend = backend;
    }

    public async Task<byte[]> UpscaleAsync(string modelId, ReadOnlyMemory<byte> luminance, int width, int height, TilingPolicy policy, CancellationToken ct)
        => (await UpscaleMeasuredAsync(modelId, luminance, width, height, policy, ct).ConfigureAwait(false)).Bytes;

    public async Task<TiledUpscaleResult> UpscaleMeasuredAsync(string modelId, ReadOnlyMemory<byte> luminance, int width, int height, TilingPolicy policy, CancellationToken ct)
    {
        if (luminance.Length != checked(width * height)) throw new ArgumentException("Luminance size does not match dimensions.", nameof(luminance));
        var wall = System.Diagnostics.Stopwatch.StartNew();
        var outputWidth = checked(width * 2);
        var outputHeight = checked(height * 2);
        var sums = new float[checked(outputWidth * outputHeight)];
        var weights = new float[sums.Length];
        var tiles = BalancedTiler.Partition(width, height, policy);
        long prepareTicks = 0, inferenceTicks = 0, blendTicks = 0;

        foreach (var tile in tiles)
        {
            ct.ThrowIfCancellationRequested();
            var preparation = Measure(() => CreateInput(luminance.Span, width, tile.Input));
            prepareTicks += preparation.Ticks;
            var item = await StartInferenceAsync(modelId, tile, preparation.Value, ct).ConfigureAwait(false);
            inferenceTicks += item.InferenceTicks;
            var tileOutputWidth = checked(item.Tile.Input.Width * 2);
            var tileOutputHeight = checked(item.Tile.Input.Height * 2);
            if (item.Output.Length != checked(3 * tileOutputWidth * tileOutputHeight))
                throw new InvalidDataException("Inference output shape is not exact 2x NCHW RGB.");
            var blend = Measure(() => BlendChannelZero(item.Output, item.Tile, sums, weights, outputWidth));
            blendTicks += blend.Ticks;
        }

        var normalizeTimer = System.Diagnostics.Stopwatch.StartNew();
        var bytes = new byte[sums.Length];
        for (var i = 0; i < bytes.Length; i++)
        {
            if (!(weights[i] > 0) || !float.IsFinite(sums[i])) throw new InvalidDataException("Tiled output is not finite or has zero weight.");
            var value = sums[i] / weights[i];
            if (!float.IsFinite(value)) throw new InvalidDataException("Normalized tiled output is not finite.");
            bytes[i] = (byte)Math.Clamp((int)MathF.Round(Math.Clamp(value, 0f, 1f) * 255f), 0, 255);
        }
        normalizeTimer.Stop();
        wall.Stop();
        return new(bytes, new(tiles.Count, wall.Elapsed, TimeSpan.FromTicks(prepareTicks),
            TimeSpan.FromTicks(inferenceTicks), TimeSpan.FromTicks(blendTicks), normalizeTimer.Elapsed));
    }

    private static float[] CreateInput(ReadOnlySpan<byte> source, int sourceWidth, IntRect rect)
    {
        var plane = checked(rect.Width * rect.Height);
        var result = new float[checked(plane * 3)];
        for (var y = 0; y < rect.Height; y++)
            for (var x = 0; x < rect.Width; x++)
            {
                var value = source[((rect.Y + y) * sourceWidth) + rect.X + x] / 255f;
                var index = (y * rect.Width) + x;
                result[index] = result[plane + index] = result[(2 * plane) + index] = value;
            }
        return result;
    }

    private static void BlendChannelZero(float[] result, TileRegion tile, float[] sums, float[] weights, int outputWidth)
    {
        var input = tile.Input;
        var core = tile.Core;
        var tileWidth = input.Width * 2;
        var tileHeight = input.Height * 2;
        var coreStartX = (core.X - input.X) * 2;
        var coreStartY = (core.Y - input.Y) * 2;
        var coreWidth = core.Width * 2;
        var coreHeight = core.Height * 2;
        var xWeights = new float[tileWidth];
        var yWeights = new float[tileHeight];
        for (var x = 0; x < tileWidth; x++) xWeights[x] = HalfSineBlend.Weight(x, tileWidth, coreStartX, coreWidth);
        for (var y = 0; y < tileHeight; y++) yWeights[y] = HalfSineBlend.Weight(y, tileHeight, coreStartY, coreHeight);
        for (var y = 0; y < tileHeight; y++)
            for (var x = 0; x < tileWidth; x++)
            {
                var weight = xWeights[x] * yWeights[y];
                var destination = ((input.Y * 2 + y) * outputWidth) + (input.X * 2 + x);
                sums[destination] += result[(y * tileWidth) + x] * weight;
                weights[destination] += weight;
            }
    }

    private static Measured<T> Measure<T>(Func<T> action)
    {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var value = action();
        timer.Stop();
        return new(value, timer.Elapsed.Ticks);
    }

    private static Measured<bool> Measure(Action action) => Measure(() => { action(); return true; });

    private Task<TileInference> StartInferenceAsync(string modelId, TileRegion tile, float[] input, CancellationToken ct) =>
        Task.Run(async () =>
        {
            var timer = System.Diagnostics.Stopwatch.StartNew();
            var output = await _backend.RunAsync(modelId, input, tile.Input.Height, tile.Input.Width, ct).ConfigureAwait(false);
            timer.Stop();
            return new TileInference(tile, output, timer.Elapsed.Ticks);
        }, ct);

    private sealed record TileInference(TileRegion Tile, float[] Output, long InferenceTicks);

    private static async Task<Measured<T>> MeasureAsync<T>(Func<Task<T>> action)
    {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        var value = await action().ConfigureAwait(false);
        timer.Stop();
        return new(value, timer.Elapsed.Ticks);
    }

    private sealed record Measured<T>(T Value, long Ticks);
}
