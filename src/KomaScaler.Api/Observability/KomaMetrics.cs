using System.Collections.Concurrent;
using System.Diagnostics.Metrics;
using System.Globalization;

namespace KomaScaler.Observability;

public sealed class KomaMetrics : IDisposable
{
    private readonly Meter _meter = new("KomaScaler", "1.0.0");
    private readonly ConcurrentDictionary<string, long> _results = new(StringComparer.Ordinal);
    private readonly Counter<long> _requestCounter;
    private readonly Histogram<double> _latency;

    public KomaMetrics()
    {
        _requestCounter = _meter.CreateCounter<long>("komascaler.requests", "requests");
        _latency = _meter.CreateHistogram<double>("komascaler.response.duration", "s");
    }

    public void Record(string result, TimeSpan elapsed)
    {
        _results.AddOrUpdate(result, 1, static (_, value) => value + 1);
        _requestCounter.Add(1, new KeyValuePair<string, object?>("result", result));
        _latency.Record(elapsed.TotalSeconds, new KeyValuePair<string, object?>("result", result));
    }

    public string Render(int queueDepth, string gpuState, int inFlight)
    {
        var lines = new List<string>
        {
            "# HELP komascaler_requests_total Conversion responses by result path.",
            "# TYPE komascaler_requests_total counter"
        };
        lines.AddRange(_results.OrderBy(x => x.Key, StringComparer.Ordinal).Select(x => $"komascaler_requests_total{{result=\"{x.Key}\"}} {x.Value.ToString(CultureInfo.InvariantCulture)}"));
        lines.Add("# TYPE komascaler_queue_depth gauge"); lines.Add($"komascaler_queue_depth {queueDepth.ToString(CultureInfo.InvariantCulture)}");
        lines.Add("# TYPE komascaler_inflight gauge"); lines.Add($"komascaler_inflight {inFlight.ToString(CultureInfo.InvariantCulture)}");
        lines.Add("# TYPE komascaler_gpu_state gauge"); lines.Add($"komascaler_gpu_state{{state=\"{gpuState}\"}} 1");
        return string.Join('\n', lines) + "\n";
    }

    public void Dispose() => _meter.Dispose();
}
