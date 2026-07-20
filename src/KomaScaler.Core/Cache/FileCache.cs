namespace KomaScaler.Cache;

public interface IResultCache
{
    Task<byte[]?> TryReadAsync(string key, CancellationToken ct);
    Task WriteAsync(string key, ReadOnlyMemory<byte> bytes, CancellationToken ct);
    string PathFor(string key);
}

public sealed class FileResultCache(string rootDirectory) : IResultCache
{
    public string PathFor(string key)
    {
        ValidateKey(key);
        return Path.Combine(rootDirectory, key[..2], key[2..4], key + ".image");
    }

    public async Task<byte[]?> TryReadAsync(string key, CancellationToken ct)
    {
        var path = PathFor(key);
        try
        {
            var bytes = await File.ReadAllBytesAsync(path, ct).ConfigureAwait(false);
            if (!ImageSignatures.IsSupportedOutput(bytes))
            {
                TryQuarantine(path);
                return null;
            }
            return bytes;
        }
        catch (FileNotFoundException) { return null; }
        catch (DirectoryNotFoundException) { return null; }
        catch (IOException) { return null; }
    }

    public async Task WriteAsync(string key, ReadOnlyMemory<byte> bytes, CancellationToken ct)
    {
        if (!ImageSignatures.IsSupportedOutput(bytes.Span)) throw new InvalidDataException("Cache accepts only validated WebP or PNG output.");
        var destination = PathFor(key);
        var directory = Path.GetDirectoryName(destination)!;
        Directory.CreateDirectory(directory);
        var temporary = Path.Combine(directory, $".{key}.{Guid.NewGuid():N}.tmp");
        try
        {
            var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None, 64 * 1024, FileOptions.Asynchronous | FileOptions.WriteThrough);
            await using (stream.ConfigureAwait(false))
            {
                await stream.WriteAsync(bytes, ct).ConfigureAwait(false);
                await stream.FlushAsync(ct).ConfigureAwait(false);
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporary, destination, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary)) File.Delete(temporary);
        }
    }

    private static void ValidateKey(string key)
    {
        if (key.Length != 64 || !key.All(Uri.IsHexDigit)) throw new ArgumentException("Cache key must be a lowercase SHA-256 digest.", nameof(key));
    }

    private static void TryQuarantine(string path)
    {
        try { File.Move(path, path + $".corrupt-{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}"); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}

public static class ImageSignatures
{
    public static bool IsPng(ReadOnlySpan<byte> bytes) => bytes.Length >= 8 &&
        bytes[..8].SequenceEqual(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });
    public static bool IsWebP(ReadOnlySpan<byte> bytes) => bytes.Length >= 12 &&
        bytes[..4].SequenceEqual("RIFF"u8) && bytes.Slice(8, 4).SequenceEqual("WEBP"u8);
    public static bool IsSupportedOutput(ReadOnlySpan<byte> bytes) => IsWebP(bytes) || IsPng(bytes);

    public static string? DetectMime(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length >= 3 && bytes[0] == 0xff && bytes[1] == 0xd8 && bytes[2] == 0xff) return "image/jpeg";
        if (IsPng(bytes)) return "image/png";
        if (IsWebP(bytes)) return "image/webp";
        if (bytes.Length >= 2 && bytes[..2].SequenceEqual("BM"u8)) return "image/bmp";
        if (bytes.Length >= 12 && bytes.Slice(4, 4).SequenceEqual("ftyp"u8) &&
            (bytes.Slice(8, 4).SequenceEqual("avif"u8) || bytes.Slice(8, 4).SequenceEqual("avis"u8))) return "image/avif";
        return null;
    }
}
