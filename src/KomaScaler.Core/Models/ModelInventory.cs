using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace KomaScaler.Models;

public sealed record GraphContract(
    int Opset, int Batch, string InputName, string InputType, string InputLayout,
    double[] InputRange, string OutputName, string OutputType, string OutputLayout,
    int Scale, int MinimumHeight, int MinimumWidth, bool DynamicSpatialDimensions, string Padding);

public sealed record RuntimeContract(
    string OnnxRuntimeVersion, int CudaMajor, int CudnnMajor, string Precision,
    bool Tf32, string RequiredExecutionProvider);

public sealed record LicenseContract(string SPDXLikeIdentifier, string Source, string Notice);

public sealed record ModelDefinition(
    string Id, int NominalHeight, int MinimumHeight, int? MaximumHeight,
    string File, string Sha256, string SourceFile, string SourceSha256);

public sealed record ModelInventory(
    int SchemaVersion, string SelectionPolicyVersion, GraphContract GraphContract,
    RuntimeContract Runtime, LicenseContract? License, IReadOnlyList<ModelDefinition> Models)
{
    public ModelDefinition Select(int orientedHeight)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(orientedHeight);
        return Models.FirstOrDefault(x => orientedHeight >= x.MinimumHeight &&
            (x.MaximumHeight is null || orientedHeight <= x.MaximumHeight))
            ?? throw new InvalidOperationException("Validated model bands do not cover the image height.");
    }
}

public sealed record InventoryValidationResult(ModelInventory? Inventory, IReadOnlyList<string> Errors)
{
    public bool IsValid => Inventory is not null && Errors.Count == 0;
}

public static class ModelInventoryLoader
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow
    };

    public static async Task<InventoryValidationResult> LoadAsync(string inventoryPath, string modelDirectory, bool verifyFiles, CancellationToken ct = default)
    {
        var errors = new List<string>();
        ModelInventory? inventory;
        try
        {
            var stream = File.OpenRead(inventoryPath);
            await using (stream.ConfigureAwait(false))
            {
                inventory = await JsonSerializer.DeserializeAsync<ModelInventory>(stream, JsonOptions, ct).ConfigureAwait(false);
            }
        }
        catch (Exception ex) when (ex is IOException or JsonException or UnauthorizedAccessException)
        {
            return new(null, [$"Inventory cannot be read: {ex.Message}"]);
        }

        if (inventory is null) return new(null, ["Inventory is empty."]);
        ValidateSchema(inventory, errors);
        if (verifyFiles) await ValidateFilesAsync(inventory, modelDirectory, errors, ct).ConfigureAwait(false);
        return new(inventory, errors);
    }

    private static void ValidateSchema(ModelInventory value, List<string> errors)
    {
        if (value.SchemaVersion != 1) errors.Add("Inventory schemaVersion must be 1.");
        if (value.Models.Count != 7) errors.Add("Inventory must contain exactly seven models.");
        if (value.GraphContract is not { Opset: 17, Batch: 1, InputName: "input", InputType: "float32", InputLayout: "NCHW", OutputName: "output", OutputType: "float32", OutputLayout: "NCHW", Scale: 2, DynamicSpatialDimensions: true })
            errors.Add("Graph contract is not the required dynamic FP32 NCHW 2x opset-17 contract.");
        if (value.Runtime is not { OnnxRuntimeVersion: "1.26.0", Precision: "fp32", Tf32: false, RequiredExecutionProvider: "TensorrtExecutionProvider" })
            errors.Add("Runtime contract must require ORT 1.26.0 TensorRT FP32 with TF32 disabled.");

        for (var i = 0; i < value.Models.Count; i++)
        {
            var model = value.Models[i];
            if (i == 0 && model.MinimumHeight != 1) errors.Add("First model band must start at height 1.");
            if (i > 0)
            {
                var previous = value.Models[i - 1];
                if (model.NominalHeight <= previous.NominalHeight) errors.Add("Nominal heights must strictly increase.");
                if (previous.MaximumHeight is null || model.MinimumHeight != previous.MaximumHeight + 1) errors.Add("Model bands must be continuous and non-overlapping.");
                var midpoint = (previous.NominalHeight + model.NominalHeight) / 2;
                if (previous.MaximumHeight != midpoint) errors.Add("Each finite band must end at the lower-model midpoint tie.");
            }
            if (i < value.Models.Count - 1 && model.MaximumHeight is null) errors.Add("Only the last model band may be open-ended.");
            if (i == value.Models.Count - 1 && model.MaximumHeight is not null) errors.Add("Last model band must be open-ended.");
            if (model.Sha256.Length != 64 || !model.Sha256.All(Uri.IsHexDigit)) errors.Add($"Model {model.Id} has an invalid SHA-256.");
        }
    }

    private static async Task ValidateFilesAsync(ModelInventory value, string directory, List<string> errors, CancellationToken ct)
    {
        foreach (var model in value.Models)
        {
            var path = Path.Combine(directory, model.File);
            if (!File.Exists(path)) { errors.Add($"Model file is missing: {model.File}"); continue; }
            var stream = File.OpenRead(path);
            await using (stream.ConfigureAwait(false))
            {
                var digest = Convert.ToHexString(await SHA256.HashDataAsync(stream, ct).ConfigureAwait(false)).ToLowerInvariant();
                if (!CryptographicOperations.FixedTimeEquals(Convert.FromHexString(digest), Convert.FromHexString(model.Sha256)))
                    errors.Add($"Model hash mismatch: {model.File}");
            }
        }
    }
}
