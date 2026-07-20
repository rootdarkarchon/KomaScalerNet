using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using KomaScaler.Configuration;
using KomaScaler.Images;
using NetVips;

namespace KomaScaler.UnitTests;

public sealed class OutputEncodingBenchmarkTests
{
    [OutputBenchmarkFact]
    public async Task LosslessGrayscaleEncodingMatrix()
    {
        var sourcePath = Environment.GetEnvironmentVariable("KOMASCALER_ENCODING_SOURCE")!;
        var destination = Environment.GetEnvironmentVariable("KOMASCALER_ENCODING_OUTPUT")!;
        Directory.CreateDirectory(destination);
        using var source = Image.NewFromFile(sourcePath, access: Enums.Access.Sequential);
        Assert.Equal(3320, source.Width);
        Assert.Equal(2800, source.Height);
        using var firstBand = source.ExtractBand(0);
        var grayscale = firstBand.WriteToMemory<byte>();
        Assert.Equal(3320 * 2800, grayscale.Length);

        var configurations = new[]
        {
            new EncoderCase("webp-effort-0", new OutputOptions { Format = "webp", Effort = 0 }),
            new EncoderCase("webp-effort-1", new OutputOptions { Format = "webp", Effort = 1 }),
            new EncoderCase("webp-effort-2", new OutputOptions { Format = "webp", Effort = 2 }),
            new EncoderCase("png-level-0", new OutputOptions { Format = "png", PngCompression = 0 }),
            new EncoderCase("png-level-1", new OutputOptions { Format = "png", PngCompression = 1 }),
            new EncoderCase("png-level-3", new OutputOptions { Format = "png", PngCompression = 3 })
        };
        var processor = new VipsImageProcessor();
        var results = new List<EncoderResult>();
        foreach (var configuration in configurations)
        {
            _ = processor.EncodeLossless(grayscale, 3320, 2800, configuration.Options);
            var encodeSamples = new List<double>();
            EncodedImage? encoded = null;
            for (var iteration = 0; iteration < 5; iteration++)
            {
                var timer = Stopwatch.StartNew();
                encoded = processor.EncodeLossless(grayscale, 3320, 2800, configuration.Options);
                timer.Stop();
                encodeSamples.Add(timer.Elapsed.TotalMilliseconds);
            }
            Assert.NotNull(encoded);
            var extension = string.Equals(configuration.Options.Format, "png", StringComparison.Ordinal) ? ".png" : ".webp";
            var outputPath = Path.Combine(destination, configuration.Name + extension);
            await File.WriteAllBytesAsync(outputPath, encoded.Bytes);

            var decodeSamples = new List<double>();
            int decodedBands = 0;
            for (var iteration = 0; iteration < 5; iteration++)
            {
                var timer = Stopwatch.StartNew();
                using var decoded = Image.NewFromBuffer(encoded.Bytes, string.Empty, Enums.Access.Sequential, Enums.FailOn.Error);
                var decodedBytes = decoded.WriteToMemory<byte>();
                timer.Stop();
                decodedBands = decoded.Bands;
                AssertDecodedEquality(grayscale, decodedBytes, decoded.Bands);
                decodeSamples.Add(timer.Elapsed.TotalMilliseconds);
            }

            var httpSamples = await MeasureHttpAsync(encoded.Bytes, encoded.ContentType, 7);
            var encodeMedian = Median(encodeSamples);
            var decodeMedian = Median(decodeSamples);
            var httpMedian = Median(httpSamples);
            results.Add(new(configuration.Name, configuration.Options.Format, configuration.Options.Effort,
                configuration.Options.PngCompression, encoded.Bytes.Length, 1, decodedBands,
                encodeMedian, decodeMedian, httpMedian, httpMedian + decodeMedian, true, outputPath));
        }

        var report = new
        {
            generatedUtc = DateTimeOffset.UtcNow,
            sourcePath,
            width = 3320,
            height = 2800,
            sourceBands = 1,
            encodeWarmups = 1,
            encodeSamples = 5,
            decodeSamples = 5,
            localhostHttpSamples = 7,
            results
        };
        await File.WriteAllTextAsync(Path.Combine(destination, "results.json"),
            JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static void AssertDecodedEquality(byte[] expected, byte[] actual, int bands)
    {
        Assert.True(bands is 1 or 3, $"Expected grayscale or expanded RGB, found {bands} bands.");
        Assert.Equal(expected.Length * bands, actual.Length);
        for (var i = 0; i < expected.Length; i++)
            for (var band = 0; band < bands; band++)
                Assert.Equal(expected[i], actual[(i * bands) + band]);
    }

    private static async Task<List<double>> MeasureHttpAsync(byte[] bytes, string contentType, int samples)
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        var server = Task.Run(async () =>
        {
            for (var iteration = 0; iteration < samples; iteration++)
            {
                using var client = await listener.AcceptTcpClientAsync().ConfigureAwait(false);
                var stream = client.GetStream();
                await using (stream.ConfigureAwait(false))
                {
                    var request = new byte[4096];
                    _ = await stream.ReadAsync(request).ConfigureAwait(false);
                    var header = Encoding.ASCII.GetBytes($"HTTP/1.1 200 OK\r\nContent-Type: {contentType}\r\nContent-Length: {bytes.Length}\r\nConnection: close\r\n\r\n");
                    await stream.WriteAsync(header).ConfigureAwait(false);
                    await stream.WriteAsync(bytes).ConfigureAwait(false);
                }
            }
        });
        try
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
            var results = new List<double>();
            for (var iteration = 0; iteration < samples; iteration++)
            {
                var timer = Stopwatch.StartNew();
                var received = await http.GetByteArrayAsync($"http://127.0.0.1:{port}/image").ConfigureAwait(false);
                timer.Stop();
                Assert.Equal(bytes, received);
                results.Add(timer.Elapsed.TotalMilliseconds);
            }
            await server.ConfigureAwait(false);
            return results;
        }
        finally { listener.Stop(); }
    }

    private static double Median(List<double> values)
    {
        values.Sort();
        return values[values.Count / 2];
    }

    private sealed record EncoderCase(string Name, OutputOptions Options);
    private sealed record EncoderResult(string Name, string Format, int WebpEffort, int PngCompression,
        int Bytes, int EncoderInputBands, int DecodedBands, double EncodeMedianMs, double DecodeMedianMs,
        double LocalhostHttpMedianMs, double ReaderMedianMs, bool DecodedPixelsEqual, string OutputPath);
}

public sealed class OutputBenchmarkFactAttribute : FactAttribute
{
    public OutputBenchmarkFactAttribute()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("KOMASCALER_ENCODING_SOURCE")) ||
            string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("KOMASCALER_ENCODING_OUTPUT")))
            Skip = "Set KOMASCALER_ENCODING_SOURCE and KOMASCALER_ENCODING_OUTPUT to run the output benchmark.";
    }
}
