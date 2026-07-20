using KomaScaler.Models;

namespace KomaScaler.UnitTests;

public sealed class ModelInventoryTests
{
    public static IEnumerable<object[]> RoutingCases()
    {
        foreach (var pair in new[]
        {
            (1, 1200), (1250, 1200), (1251, 1300), (1350, 1300), (1351, 1400), (1450, 1400),
            (1451, 1500), (1550, 1500), (1551, 1600), (1760, 1600), (1761, 1920), (1984, 1920),
            (1985, 2048), (10000, 2048)
        }) yield return [pair.Item1, pair.Item2];
    }

    [Theory]
    [MemberData(nameof(RoutingCases))]
    public void Select_UsesExactOrientedHeightBands(int height, int nominalHeight)
    {
        Assert.Equal(nominalHeight, TestInventory.Create().Select(height).NominalHeight);
    }

    [Fact]
    public void Select_RejectsNonPositiveHeight() => Assert.Throws<ArgumentOutOfRangeException>(() => TestInventory.Create().Select(0));

    [Fact]
    public async Task ProductionManifest_SchemaAndBandsValidate()
    {
        var path = Path.Combine(AppContext.BaseDirectory, "models", "models.production.json");
        var result = await ModelInventoryLoader.LoadAsync(path, Path.GetDirectoryName(path)!, verifyFiles: false);
        Assert.True(result.IsValid, string.Join(" | ", result.Errors));
    }
}

internal static class TestInventory
{
    public static ModelInventory Create() => new(
        1, "selection-v1",
        new(17, 1, "input", "float32", "NCHW", [0, 1], "output", "float32", "NCHW", 2, 4, 4, true, "graph-owned"),
        new("1.26.0", 12, 9, "fp32", false, "TensorrtExecutionProvider"), null,
        Bands());

    public static IReadOnlyList<ModelDefinition> Bands() =>
    [
        Make(1200, 1, 1250), Make(1300, 1251, 1350), Make(1400, 1351, 1450),
        Make(1500, 1451, 1550), Make(1600, 1551, 1760), Make(1920, 1761, 1984), Make(2048, 1985, null)
    ];

    private static ModelDefinition Make(int nominal, int minimum, int? maximum) =>
        new($"model-{nominal}", nominal, minimum, maximum, $"{nominal}.onnx", new string('a', 64), $"{nominal}.pth", new string('b', 64));
}
