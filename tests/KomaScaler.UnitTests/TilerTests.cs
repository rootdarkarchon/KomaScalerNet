using KomaScaler.Configuration;
using KomaScaler.Tiling;

namespace KomaScaler.UnitTests;

public sealed class TilerTests
{
    [Theory]
    [InlineData(1, 1)]
    [InlineData(321, 641)]
    [InlineData(903, 1761)]
    [InlineData(2401, 1600)]
    public void Partition_CoversCoresExactlyOnce_AndStaysInBounds(int width, int height)
    {
        var policy = new TilingPolicy(320, 32);
        var tiles = BalancedTiler.Partition(width, height, policy);
        var covered = new bool[width * height];
        foreach (var tile in tiles)
        {
            Assert.InRange(tile.Core.Width, 1, 320);
            Assert.InRange(tile.Core.Height, 1, 320);
            Assert.True(tile.Input.X >= 0 && tile.Input.Y >= 0 && tile.Input.Right <= width && tile.Input.Bottom <= height);
            Assert.InRange(tile.Input.Width, 1, 384);
            Assert.InRange(tile.Input.Height, 1, 384);
            for (var y = tile.Core.Y; y < tile.Core.Bottom; y++)
                for (var x = tile.Core.X; x < tile.Core.Right; x++)
                {
                    var index = (y * width) + x;
                    Assert.False(covered[index]);
                    covered[index] = true;
                }
        }
        Assert.All(covered, Assert.True);
    }

    [Theory]
    [InlineData(3, 5)]
    [InlineData(320, 321)]
    [InlineData(641, 319)]
    public async Task Upscale_IsFiniteAndExact2x(int width, int height)
    {
        var source = Enumerable.Range(0, width * height).Select(i => (byte)(i % 256)).ToArray();
        var result = await new TiledUpscaler(new NearestBackend()).UpscaleAsync("fake", source, width, height, new(320, 32), CancellationToken.None);
        Assert.Equal(width * height * 4, result.Length);
        Assert.All(result, value => Assert.InRange(value, (byte)0, (byte)255));
        for (var y = 0; y < height * 2; y++)
            for (var x = 0; x < width * 2; x++)
                Assert.Equal(source[((y / 2) * width) + (x / 2)], result[(y * width * 2) + x]);
    }

    [Fact]
    public async Task UpscaleMeasured_ReportsEveryStageAndTile()
    {
        var result = await new TiledUpscaler(new NearestBackend()).UpscaleMeasuredAsync(
            "fake", new byte[641 * 321], 641, 321, new(320, 32), CancellationToken.None);
        Assert.Equal(6, result.Timings.TileCount);
        Assert.True(result.Timings.Wall > TimeSpan.Zero);
        Assert.True(result.Timings.Prepare >= TimeSpan.Zero);
        Assert.True(result.Timings.InferenceAggregate >= TimeSpan.Zero);
        Assert.True(result.Timings.Blend >= TimeSpan.Zero);
        Assert.True(result.Timings.Normalize >= TimeSpan.Zero);
    }

    [Fact]
    public async Task Upscale_SerializesTilesAndPreservesBytes()
    {
        var backend = new TrackingBackend();
        var source = Enumerable.Range(0, 641 * 321).Select(i => (byte)(i % 256)).ToArray();
        var result = await new TiledUpscaler(backend).UpscaleAsync("fake", source, 641, 321, new(320, 32), CancellationToken.None);
        Assert.Equal(1, backend.MaximumActive);
        for (var y = 0; y < 321 * 2; y++)
            for (var x = 0; x < 641 * 2; x++)
                Assert.Equal(source[((y / 2) * 641) + (x / 2)], result[(y * 641 * 2) + x]);
    }

    [Fact]
    public void HalfSine_IsFiniteBoundedAndMonotonic()
    {
        var values = Enumerable.Range(0, 100).Select(i => HalfSineBlend.HalfSine(i / 99f)).ToArray();
        Assert.All(values, x => Assert.True(float.IsFinite(x) && x is >= 0 and <= 1));
        for (var i = 1; i < values.Length; i++) Assert.True(values[i] >= values[i - 1]);
    }

    private sealed class NearestBackend : ITileInferenceBackend
    {
        public Task<float[]> RunAsync(string modelId, float[] input, int height, int width, CancellationToken cancellationToken)
        {
            var outputWidth = width * 2; var outputHeight = height * 2; var plane = outputWidth * outputHeight;
            var output = new float[plane * 3];
            for (var y = 0; y < outputHeight; y++)
                for (var x = 0; x < outputWidth; x++)
                {
                    var value = input[((y / 2) * width) + (x / 2)];
                    var index = (y * outputWidth) + x;
                    output[index] = output[plane + index] = output[(2 * plane) + index] = value;
                }
            return Task.FromResult(output);
        }
    }

    private sealed class TrackingBackend : ITileInferenceBackend
    {
        private readonly NearestBackend _inner = new();
        private int _active;
        private int _maximumActive;
        public int MaximumActive => Volatile.Read(ref _maximumActive);

        public async Task<float[]> RunAsync(string modelId, float[] input, int height, int width, CancellationToken cancellationToken)
        {
            var active = Interlocked.Increment(ref _active);
            lock (_inner) _maximumActive = Math.Max(_maximumActive, active);
            try
            {
                await Task.Delay(20, cancellationToken).ConfigureAwait(false);
                return await _inner.RunAsync(modelId, input, height, width, cancellationToken).ConfigureAwait(false);
            }
            finally { Interlocked.Decrement(ref _active); }
        }
    }
}
