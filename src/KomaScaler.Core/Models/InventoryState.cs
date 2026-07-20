using KomaScaler.Configuration;

namespace KomaScaler.Models;

public sealed class InventoryState(UpscalingOptions options)
{
    private InventoryValidationResult _result = new(null, ["Model inventory has not been validated."]);
    public InventoryValidationResult Result => Volatile.Read(ref _result);
    public ModelInventory? Inventory => Result.Inventory;
    public bool IsReady => Result.IsValid;

    public async Task RefreshAsync(CancellationToken ct = default)
    {
        var configured = Path.Combine(options.Models.Directory, options.Models.InventoryFile);
        var bundled = Path.Combine(AppContext.BaseDirectory, "models", options.Models.InventoryFile);
        var inventoryPath = File.Exists(configured) ? configured : bundled;
        var result = await ModelInventoryLoader.LoadAsync(inventoryPath, options.Models.Directory, verifyFiles: true, ct).ConfigureAwait(false);
        Volatile.Write(ref _result, result);
    }
}

public sealed class InventoryRefreshService(InventoryState state) : Microsoft.Extensions.Hosting.BackgroundService
{
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(30));
        while (await timer.WaitForNextTickAsync(stoppingToken).ConfigureAwait(false))
            await state.RefreshAsync(stoppingToken).ConfigureAwait(false);
    }
}
