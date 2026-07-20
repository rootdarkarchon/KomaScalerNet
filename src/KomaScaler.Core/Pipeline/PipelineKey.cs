using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using KomaScaler.Configuration;
using KomaScaler.Models;

namespace KomaScaler.Pipeline;

public sealed record PipelineIdentity(
    byte[] SourceSha256, ModelDefinition Model, string SelectionPolicyVersion,
    TilingPolicy Tiling, string InferenceIdentity, string EncoderFormat, bool EncoderLossless, int EncoderQuality, int EncoderEffort,
    int EncoderPngCompression);

public static class PipelineKey
{
    public const string SchemaVersion = "komascaler-pipeline-v1";
    public const string PreprocessingVersion = "srgb-orient-whitealpha-neutral12-autolevel-v1";
    public const string OutputChannelPolicy = "scale2-output-channel0-gray8";

    public static string Create(PipelineIdentity identity)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Add(hash, SchemaVersion);
        Add(hash, identity.SourceSha256);
        Add(hash, identity.Model.Id);
        Add(hash, identity.Model.Sha256);
        Add(hash, PreprocessingVersion);
        Add(hash, identity.SelectionPolicyVersion);
        Add(hash, identity.Tiling.MaximumCoreSize.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add(hash, identity.Tiling.ContextPixelsPerSide.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add(hash, TilingPolicy.PartitionVersion);
        Add(hash, TilingPolicy.BlendVersion);
        Add(hash, OutputChannelPolicy);
        Add(hash, identity.InferenceIdentity);
        Add(hash, identity.EncoderFormat);
        Add(hash, identity.EncoderLossless ? "lossless" : "lossy");
        Add(hash, identity.EncoderQuality.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add(hash, identity.EncoderEffort.ToString(System.Globalization.CultureInfo.InvariantCulture));
        Add(hash, identity.EncoderPngCompression.ToString(System.Globalization.CultureInfo.InvariantCulture));
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    private static void Add(IncrementalHash hash, string value) => Add(hash, Encoding.UTF8.GetBytes(value));
    private static void Add(IncrementalHash hash, ReadOnlySpan<byte> value)
    {
        Span<byte> length = stackalloc byte[4];
        BinaryPrimitives.WriteInt32BigEndian(length, value.Length);
        hash.AppendData(length);
        hash.AppendData(value);
    }
}
