using KomaScaler.Cache;
using KomaScaler.Configuration;
using NetVips;

namespace KomaScaler.Images;

public sealed record ImageInspection(
    bool Supported, bool IsMonochrome, string ContentType, int Width, int Height,
    byte[]? PreprocessedLuminance, string? BypassReason);

public sealed record EncodedImage(byte[] Bytes, string ContentType);

public interface IImageProcessor
{
    ImageInspection Inspect(ReadOnlySpan<byte> encoded, long maxDecodedPixels, int monochromeThreshold);
    EncodedImage EncodeLossless(ReadOnlyMemory<byte> grayscale, int width, int height, OutputOptions options);
}

public sealed class VipsImageProcessor : IImageProcessor
{
    public ImageInspection Inspect(ReadOnlySpan<byte> encoded, long maxDecodedPixels, int monochromeThreshold)
    {
        var contentType = ImageSignatures.DetectMime(encoded);
        if (contentType is null) return new(false, false, "application/octet-stream", 0, 0, null, "unsupported-format");
        try
        {
            using var decoded = Image.NewFromBuffer(encoded, string.Empty, Enums.Access.Sequential, Enums.FailOn.Error);
            if (ReadIntMetadata(decoded, "n-pages", 1) > 1 || (decoded.PageHeight > 0 && decoded.PageHeight < decoded.Height))
                return new(false, false, contentType, decoded.Width, decoded.Height, null, "animated");

            using var oriented = decoded.Autorot();
            ValidateDimensions(oriented.Width, oriented.Height, maxDecodedPixels);
            using var colorManaged = HasField(oriented, "icc-profile-data")
                ? oriented.IccTransform("srgb", embedded: true)
                : oriented.Colourspace(Enums.Interpretation.Srgb);
            using var bytesImage = ToUchar(colorManaged);
            var pixels = bytesImage.WriteToMemory<byte>();
            var bands = bytesImage.Bands;
            if (bands is < 1 or > 4) return new(false, false, contentType, bytesImage.Width, bytesImage.Height, null, "unsupported-channels");
            var luminance = new byte[checked(bytesImage.Width * bytesImage.Height)];
            var monochrome = ExtractAndClassify(pixels, bands, luminance, monochromeThreshold);
            if (!monochrome) return new(true, false, contentType, bytesImage.Width, bytesImage.Height, null, "color");
            AutoLevel(luminance);
            return new(true, true, contentType, bytesImage.Width, bytesImage.Height, luminance, null);
        }
        catch (Exception ex) when (ex is VipsException or OverflowException or ArgumentException or InvalidDataException)
        {
            return new(false, false, contentType, 0, 0, null, "decode-failure");
        }
    }

    public EncodedImage EncodeLossless(ReadOnlyMemory<byte> grayscale, int width, int height, OutputOptions options)
    {
        if (grayscale.Length != checked(width * height)) throw new ArgumentException("Grayscale buffer does not match dimensions.", nameof(grayscale));
        using var image = Image.NewFromMemoryCopy(grayscale.Span, width, height, 1, Enums.BandFormat.Uchar);
        if (string.Equals(options.Format, "webp", StringComparison.OrdinalIgnoreCase))
        {
            var webp = image.WebpsaveBuffer(q: options.Quality, lossless: true, effort: options.Effort, keep: Enums.ForeignKeep.None);
            if (!ImageSignatures.IsWebP(webp)) throw new InvalidDataException("libvips produced invalid WebP bytes.");
            return new(webp, "image/webp");
        }
        if (string.Equals(options.Format, "png", StringComparison.OrdinalIgnoreCase))
        {
            var png = image.PngsaveBuffer(compression: options.PngCompression, palette: false, bitdepth: 8, keep: Enums.ForeignKeep.None);
            if (!ImageSignatures.IsPng(png)) throw new InvalidDataException("libvips produced invalid PNG bytes.");
            return new(png, "image/png");
        }
        throw new InvalidOperationException($"Unsupported output format: {options.Format}");
    }

    private static Image ToUchar(Image image)
    {
        return image.Format switch
        {
            Enums.BandFormat.Uchar => image.Copy(),
            Enums.BandFormat.Ushort => (image / 257d).Cast(Enums.BandFormat.Uchar),
            Enums.BandFormat.Float or Enums.BandFormat.Double => (image * 255d).Cast(Enums.BandFormat.Uchar),
            _ => image.Cast(Enums.BandFormat.Uchar)
        };
    }

    private static bool ExtractAndClassify(byte[] pixels, int bands, byte[] luminance, int threshold)
    {
        if (bands <= 2)
        {
            for (var i = 0; i < luminance.Length; i++)
            {
                var gray = pixels[i * bands];
                if (bands == 2) gray = CompositeOverWhite(gray, pixels[(i * bands) + 1]);
                luminance[i] = gray;
            }
            return true;
        }

        long residual = 0;
        long denominator = 0;
        for (var i = 0; i < luminance.Length; i++)
        {
            var offset = i * bands;
            var r = pixels[offset]; var g = pixels[offset + 1]; var b = pixels[offset + 2];
            if (bands == 4)
            {
                var alpha = pixels[offset + 3];
                r = CompositeOverWhite(r, alpha); g = CompositeOverWhite(g, alpha); b = CompositeOverWhite(b, alpha);
            }
            luminance[i] = (byte)Math.Clamp((int)Math.Round((0.2126 * r) + (0.7152 * g) + (0.0722 * b)), 0, 255);
            if ((r == 0 && g == 0 && b == 0) || (r == 255 && g == 255 && b == 255)) continue;
            residual += Math.Max(0, Math.Abs(r - g) - 12);
            residual += Math.Max(0, Math.Abs(r - b) - 12);
            residual += Math.Max(0, Math.Abs(g - b) - 12);
            denominator += 3;
        }
        return denominator == 0 || (double)residual / denominator <= threshold / 12d;
    }

    private static byte CompositeOverWhite(byte value, byte alpha) =>
        (byte)(((value * alpha) + (255 * (255 - alpha)) + 127) / 255);

    private static void AutoLevel(byte[] pixels)
    {
        Span<int> histogram = stackalloc int[256];
        foreach (var value in pixels) histogram[value]++;
        var black = FindPeak(histogram, 0, 30, 1);
        var white = FindPeak(histogram, 255, 225, -1);
        if (white <= black) return;
        for (var i = 0; i < pixels.Length; i++)
            pixels[i] = (byte)Math.Clamp((int)Math.Round(255d * (pixels[i] - black) / (white - black)), 0, 255);
    }

    private static int FindPeak(ReadOnlySpan<int> histogram, int start, int end, int direction)
    {
        var peak = start;
        for (var i = start; direction > 0 ? i <= end : i >= end; i += direction)
            if (histogram[i] > histogram[peak]) peak = i;
        var decreases = 0;
        for (var i = end + direction; i is >= 0 and < 256; i += direction)
        {
            if (histogram[i] > histogram[peak]) { peak = i; decreases = 0; }
            else if (++decreases >= 2) break;
        }
        return peak;
    }

    private static bool HasField(Image image, string field) => image.GetFields().Contains(field, StringComparer.Ordinal);
    private static int ReadIntMetadata(Image image, string name, int fallback)
    {
        if (!HasField(image, name)) return fallback;
        try { return Convert.ToInt32(image.Get(name), System.Globalization.CultureInfo.InvariantCulture); }
        catch (Exception ex) when (ex is FormatException or InvalidCastException or VipsException) { return fallback; }
    }

    private static void ValidateDimensions(int width, int height, long maximumPixels)
    {
        if (width <= 0 || height <= 0 || checked((long)width * height) > maximumPixels)
            throw new InvalidDataException("Decoded dimensions exceed the configured safety limit.");
    }
}
