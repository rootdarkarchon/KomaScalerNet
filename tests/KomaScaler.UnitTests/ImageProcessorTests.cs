using KomaScaler.Cache;
using KomaScaler.Configuration;
using KomaScaler.Images;
using NetVips;

namespace KomaScaler.UnitTests;

public sealed class ImageProcessorTests
{
    [NativeVipsFact]
    public void PureBlackAndWhiteAreMonochrome()
    {
        var pixels = new byte[] { 0, 0, 0, 255, 255, 255 };
        using var image = Image.NewFromMemoryCopy(pixels, 2, 1, 3, Enums.BandFormat.Uchar);
        var inspection = new VipsImageProcessor().Inspect(image.PngsaveBuffer(), 100, 12);
        Assert.True(inspection.Supported);
        Assert.True(inspection.IsMonochrome);
        Assert.Equal(2, inspection.Width);
        Assert.Equal(1, inspection.Height);
    }

    [NativeVipsFact]
    public void MeaningfulColorIsBypassedAndTransparentColorCompositesOverWhite()
    {
        using var color = Image.NewFromMemoryCopy(new byte[] { 255, 0, 0 }, 1, 1, 3, Enums.BandFormat.Uchar);
        Assert.False(new VipsImageProcessor().Inspect(color.PngsaveBuffer(), 100, 12).IsMonochrome);
        using var transparent = Image.NewFromMemoryCopy(new byte[] { 255, 0, 0, 0 }, 1, 1, 4, Enums.BandFormat.Uchar);
        Assert.True(new VipsImageProcessor().Inspect(transparent.PngsaveBuffer(), 100, 12).IsMonochrome);
    }

    [NativeVipsFact]
    public void LosslessWebPEncoderProducesMatchingSignature()
    {
        var encoded = new VipsImageProcessor().EncodeLossless(new byte[] { 0, 64, 128, 255 }, 2, 2, new OutputOptions { Format = "webp" });
        Assert.True(ImageSignatures.IsWebP(encoded.Bytes));
        Assert.Equal("image/webp", encoded.ContentType);
    }

    [NativeVipsFact]
    public void LosslessPngEncoderProducesSingleBandMatchingPixels()
    {
        var pixels = new byte[] { 0, 64, 128, 255 };
        var encoded = new VipsImageProcessor().EncodeLossless(pixels, 2, 2, new OutputOptions { Format = "png", PngCompression = 1 });
        Assert.True(ImageSignatures.IsPng(encoded.Bytes));
        Assert.Equal("image/png", encoded.ContentType);
        using var decoded = Image.NewFromBuffer(encoded.Bytes);
        Assert.Equal(1, decoded.Bands);
        Assert.Equal(pixels, decoded.WriteToMemory<byte>());
    }

    [NativeVipsFact]
    public void JpegPngWebPAndBmpAreDetectedFromBytes()
    {
        using var image = Image.NewFromMemoryCopy(new byte[] { 0, 64, 128, 255 }, 2, 2, 1, Enums.BandFormat.Uchar);
        var processor = new VipsImageProcessor();
        foreach (var suffix in new[] { ".jpg", ".png", ".webp", ".bmp" })
        {
            var encoded = image.WriteToBuffer(suffix);
            var inspection = processor.Inspect(encoded, 100, 12);
            Assert.True(inspection.Supported, suffix);
            Assert.True(inspection.IsMonochrome, suffix);
        }
    }

    [NativeVipsFact]
    public void DecodedPixelLimitStopsBeforeUpscaleAllocation()
    {
        using var image = Image.NewFromMemoryCopy(new byte[] { 0, 64, 128, 255 }, 2, 2, 1, Enums.BandFormat.Uchar);
        var inspection = new VipsImageProcessor().Inspect(image.PngsaveBuffer(), 3, 12);
        Assert.False(inspection.Supported);
        Assert.Null(inspection.PreprocessedLuminance);
    }

    [NativeVipsFact]
    public void AvifIsDetectedAndDecodedWhenLibvipsExposesAv1Heif()
    {
        using var image = Image.NewFromMemoryCopy(new byte[] { 0, 64, 128, 255 }, 2, 2, 1, Enums.BandFormat.Uchar);
        var encoded = image.HeifsaveBuffer(lossless: true, compression: Enums.ForeignHeifCompression.Av1);
        Assert.Equal("image/avif", ImageSignatures.DetectMime(encoded));
        var inspection = new VipsImageProcessor().Inspect(encoded, 100, 12);
        Assert.True(inspection.Supported);
        Assert.True(inspection.IsMonochrome);
        Assert.Equal(2, inspection.Width);
        Assert.Equal(2, inspection.Height);
    }

    [NativeVipsFact]
    public void ExifOrientationIsPhysicallyAppliedBeforeDimensionsAreReported()
    {
        using var image = Image.NewFromMemoryCopy(Enumerable.Range(0, 2 * 3).Select(x => (byte)(x * 32)).ToArray(), 2, 3, 1, Enums.BandFormat.Uchar);
        using var orientedMetadata = image.Mutate(mutable => mutable.Set(GValue.GIntType, "orientation", 6));
        var encoded = orientedMetadata.JpegsaveBuffer(keep: Enums.ForeignKeep.All);
        var inspection = new VipsImageProcessor().Inspect(encoded, 100, 12);
        Assert.True(inspection.Supported);
        Assert.Equal(3, inspection.Width);
        Assert.Equal(2, inspection.Height);
    }
}

public sealed class NativeVipsFactAttribute : FactAttribute
{
    public NativeVipsFactAttribute()
    {
        try { _ = NetVips.NetVips.Version(0); }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException)
        {
            Skip = "Install Debian libvips with PNG/WebP loaders to run native image-pipeline tests.";
        }
    }
}
