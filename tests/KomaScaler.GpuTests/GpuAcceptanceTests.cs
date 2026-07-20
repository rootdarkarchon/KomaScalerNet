using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using KomaScaler.Configuration;
using KomaScaler.Inference;
using KomaScaler.Models;
using KomaScaler.Tiling;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Xunit.Abstractions;

namespace KomaScaler.GpuTests;

public sealed class GpuAcceptanceTests(ITestOutputHelper output)
{
    [TensorRtFact]
    public async Task TensorRtProviderPreflightAndPlacement()
    {
        var target = await LoadTargetAsync();
        var providers = OrtEnv.Instance().GetAvailableProviders();
        output.WriteLine("availableProviders={0}", string.Join(',', providers));
        Assert.Contains("TensorrtExecutionProvider", providers);
        Assert.Contains("CUDAExecutionProvider", providers);
        var modelIndex = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_TENSORRT_MODEL_INDEX"), System.Globalization.CultureInfo.InvariantCulture, out var configuredModelIndex) ? configuredModelIndex : 0;
        var extent = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_TENSORRT_PREFLIGHT_EXTENT"), System.Globalization.CultureInfo.InvariantCulture, out var configuredExtent) ? configuredExtent : 320;
        var model = target.Inventory.Models[modelIndex];
        var cache = target.Options.Models.TensorRtCacheDirectory;
        Directory.CreateDirectory(cache);
        using var sessionOptions = CreateTensorRtOptions(target.Inventory, target.Options.Models, cache);
        using var session = new InferenceSession(Path.Combine(target.Directory, model.File), sessionOptions);
        var input = CreateInput(extent, extent);
        var timer = Stopwatch.StartNew();
        var result = DirectCudaRun(session, input, extent, extent);
        timer.Stop();
        Assert.Equal(3 * extent * 2 * extent * 2, result.Length);
        var profilePath = session.EndProfiling();
        using var profile = JsonDocument.Parse(await File.ReadAllTextAsync(profilePath));
        var tensorRtNodes = CountProviderNodes(profile.RootElement, "TensorrtExecutionProvider");
        var cudaNodes = CountProviderNodes(profile.RootElement, "CUDAExecutionProvider");
        var cpuNodes = CountProviderNodes(profile.RootElement, "CPUExecutionProvider");
        var cpuCompute = CountCpuComputeNodes(profile.RootElement);
        output.WriteLine("providerProfile={0} tensorRtNodes={1} cudaFallbackNodes={2} cpuNodes={3} cpuComputeNodes={4} elapsedMs={5:F3} vramMiB={6}",
            profilePath, tensorRtNodes, cudaNodes, cpuNodes, cpuCompute, timer.Elapsed.TotalMilliseconds, ReadVram());
        Assert.True(tensorRtNodes > 0, "TensorRT registered but no graph compute was assigned to it.");
        Assert.Equal(0, cudaNodes);
        Assert.Equal(0, cpuCompute);
    }

    [TensorRtFact]
    public async Task TensorRtSelectedModelBuildAndRun()
    {
        var target = await LoadTargetAsync();
        var modelIndex = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_TENSORRT_MODEL_INDEX"), System.Globalization.CultureInfo.InvariantCulture, out var configuredModelIndex) ? configuredModelIndex : 0;
        var extent = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_TENSORRT_PREFLIGHT_EXTENT"), System.Globalization.CultureInfo.InvariantCulture, out var configuredExtent) ? configuredExtent : 320;
        var model = target.Inventory.Models[modelIndex];
        await using var gpu = target.CreateGpu();
        var timer = Stopwatch.StartNew();
        var result = await gpu.RunAsync(model.Id, CreateInput(extent, extent), extent, extent, CancellationToken.None);
        timer.Stop();
        Assert.Equal(3 * extent * 2 * extent * 2, result.Length);
        output.WriteLine("selectedModelBuild model={0} extent={1} elapsedMs={2:F3} vramMiB={3}",
            model.Id, extent, timer.Elapsed.TotalMilliseconds, ReadVram());
    }

    [TensorRtFact]
    public async Task TensorRtMissingPostStartupEngineIsRejectedWithoutBuild()
    {
        var target = await LoadTargetAsync();
        Assert.True(target.Options.Models.PreloadOnStartup, "Set KOMASCALER_PRELOAD_ON_STARTUP=true for this production-policy test.");
        var model = target.Inventory.Models[0];
        await using var gpu = target.CreateGpu();
        await gpu.WarmAsync(CancellationToken.None);
        var material = $"{model.Sha256}|{TensorRtPolicy.EngineCacheIdentity(target.Options.Models, target.Inventory.Runtime.OnnxRuntimeVersion)}";
        var identity = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(material))).ToLowerInvariant();
        var directory = Path.Combine(target.Options.Models.TensorRtCacheDirectory, identity);
        var engine = Directory.EnumerateFiles(directory, "*.engine", SearchOption.TopDirectoryOnly).Single();
        var disabled = engine + ".disabled-for-acceptance";
        File.Move(engine, disabled);
        try
        {
            var timer = Stopwatch.StartNew();
            var failure = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                gpu.RunAsync(model.Id, CreateInput(8, 8), 8, 8, CancellationToken.None));
            timer.Stop();
            Assert.Contains("request-time engine construction is forbidden", failure.Message, StringComparison.Ordinal);
            Assert.True(timer.Elapsed < TimeSpan.FromSeconds(5), $"Missing-engine rejection took {timer.Elapsed}.");
            Assert.Equal(GpuServiceState.Faulted, gpu.State);
            output.WriteLine("missingProductionEngineRejectedMs={0:F3} state={1}", timer.Elapsed.TotalMilliseconds, gpu.State);
        }
        finally
        {
            File.Move(disabled, engine);
            Environment.ExitCode = 0;
        }
    }

    [TensorRtFact]
    public async Task TensorRtProviderNumericalIsolation()
    {
        var target = await LoadTargetAsync();
        var model = target.Inventory.Models[0];
        const int width = 320, height = 320;
        var nchw = CreateInput(height, width);
        float[] cudaSingle;
        using (var options = CreateCudaOptions(profile: false))
        using (var session = new InferenceSession(Path.Combine(target.Directory, model.File), options))
            cudaSingle = DirectCudaRun(session, nchw, height, width);
        await Task.Delay(500);
        float[] tensorRtSingle;
        using (var options = CreateTensorRtOptions(target.Inventory, target.Options.Models, target.Options.Models.TensorRtCacheDirectory))
        using (var session = new InferenceSession(Path.Combine(target.Directory, model.File), options))
            tensorRtSingle = DirectCudaRun(session, nchw, height, width);
        var singleFloat = CalculateFloatError(cudaSingle, tensorRtSingle);
        var singleByte = CalculateError(QuantizeChannelZero(cudaSingle, width * 2, height * 2),
            QuantizeChannelZero(tensorRtSingle, width * 2, height * 2), null, width * 2, height * 2);
        output.WriteLine("singleTileCudaVsTensorRt floatMax={0:E8} floatMae={1:E8} floatPsnr={2:F4} byteSsim={3:F9}",
            singleFloat.Maximum, singleFloat.Mae, singleFloat.Psnr, singleByte.Ssim);

        Assert.InRange(singleFloat.Maximum, 0, 5e-6);
        Assert.InRange(singleFloat.Mae, 0, 1e-6);
        Assert.True(singleByte.Ssim > 0.999999);
    }

    [TensorRtFact]
    public async Task TensorRtActiveModelSwitchLifecycle()
    {
        var target = await LoadTargetAsync();
        await using var gpu = target.CreateGpu();
        var switchExtent = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_SWITCH_EXTENT"), System.Globalization.CultureInfo.InvariantCulture, out var configuredExtent) ? configuredExtent : 320;
        var input = CreateInput(switchExtent, switchExtent);
        var baseline = ReadVram();
        var sequence = new[] { 0, 0, 1, 0, 1 };
        foreach (var index in sequence)
        {
            var timer = Stopwatch.StartNew();
            var result = await gpu.RunAsync(target.Inventory.Models[index].Id, input, switchExtent, switchExtent, CancellationToken.None);
            timer.Stop();
            Assert.Equal(3 * switchExtent * 2 * switchExtent * 2, result.Length);
            output.WriteLine("activeModel={0} elapsedMs={1:F3} vramMiB={2}", target.Inventory.Models[index].Id, timer.Elapsed.TotalMilliseconds, ReadVram());
        }
        await gpu.DrainIfIdleAsync(TimeSpan.Zero, CancellationToken.None);
        await Task.Delay(500);
        output.WriteLine("activeModelIdleDrain baselineVramMiB={0} drainedVramMiB={1}", baseline, ReadVram());
        Assert.Equal(GpuServiceState.Uninitialized, gpu.State);
    }
    [GpuQualityFact]
    public async Task UntiledDirectOrtQualityReferenceMatrix()
    {
        var target = await LoadTargetAsync();
        var model = target.Inventory.Models[0];
        var configuredWidth = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_QUALITY_WIDTH"), System.Globalization.CultureInfo.InvariantCulture, out var width) ? width : 0;
        var configuredHeight = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_QUALITY_HEIGHT"), System.Globalization.CultureInfo.InvariantCulture, out var height) ? height : 0;
        QualityInput[] cases = configuredWidth > 0 && configuredHeight > 0
            ? [new QualityInput("configured", configuredWidth, configuredHeight, CreateQualityInput(configuredWidth, configuredHeight))]
            :
            [
                new QualityInput("landscape", 640, 384, CreateQualityInput(640, 384)),
                new QualityInput("portrait", 384, 640, CreateQualityInput(384, 640)),
                new QualityInput("square", 640, 640, CreateQualityInput(640, 640))
            ];
        var references = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        using (var options = CreateCudaOptions(profile: false))
        using (var session = new InferenceSession(Path.Combine(target.Directory, model.File), options))
        {
            foreach (var item in cases)
            {
                var direct = DirectCudaRun(session, CreateNchwInput(item.Bytes), item.Height, item.Width);
                references[item.Name] = QuantizeChannelZero(direct, item.Width * 2, item.Height * 2);
            }
        }

        await Task.Delay(500);
        var results = new List<QualityResult>();
        await using (var gpu = target.CreateGpu())
        {
            var configuredCore = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_QUALITY_CORE"), System.Globalization.CultureInfo.InvariantCulture, out var coreValue) ? coreValue : 0;
            var configuredContext = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_QUALITY_CONTEXT"), System.Globalization.CultureInfo.InvariantCulture, out var contextValue) ? contextValue : 0;
            IEnumerable<TilingPolicy> policies = configuredCore > 0 && configuredContext > 0
                ? [new TilingPolicy(configuredCore, configuredContext)]
                : new[] { new TilingPolicy(320, 32) }.Concat(
                    from core in new[] { 512, 576 }
                    from context in new[] { 32, 48, 64 }
                    select new TilingPolicy(core, context));
            foreach (var item in cases)
                foreach (var policy in policies)
                {
                    var core = policy.MaximumCoreSize;
                    var context = policy.ContextPixelsPerSide;
                    var timer = Stopwatch.StartNew();
                    var candidate = await new TiledUpscaler(gpu).UpscaleMeasuredAsync(
                        model.Id, item.Bytes, item.Width, item.Height, policy, CancellationToken.None);
                    timer.Stop();
                    var whole = CalculateError(references[item.Name], candidate.Bytes, null, item.Width * 2, item.Height * 2);
                    var seams = CreateSeamMask(item.Width, item.Height, policy, band: 8);
                    var seam = CalculateError(references[item.Name], candidate.Bytes, seams, item.Width * 2, item.Height * 2);
                    results.Add(new(item.Name, item.Width, item.Height, core, context,
                        candidate.Timings.TileCount, timer.Elapsed.TotalMilliseconds, whole, seam));
                    output.WriteLine("quality input={0} size={1}x{2} core={3} context={4} tiles={5} elapsedMs={6:F3} wholePsnr={7:F4} wholeSsim={8:F6} seamPsnr={9:F4} seamMae={10:F6}",
                        item.Name, item.Width, item.Height, core, context, candidate.Timings.TileCount,
                        timer.Elapsed.TotalMilliseconds, whole.Psnr, whole.Ssim, seam.Psnr, seam.Mae);
                }
        }

        var destination = Environment.GetEnvironmentVariable("KOMASCALER_QUALITY_OUTPUT")!;
        Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
        await File.WriteAllTextAsync(destination, JsonSerializer.Serialize(new
        {
            generatedUtc = DateTimeOffset.UtcNow,
            provider = Environment.GetEnvironmentVariable("KOMASCALER_GPU_PROVIDER") ?? "Cuda",
            reference = "untiled direct ORT FP32, channel zero quantized to byte",
            seamBandOutputPixels = 8,
            model = model.Id,
            results
        }, new JsonSerializerOptions { WriteIndented = true }));
        Assert.All(results, item => Assert.True(double.IsFinite(item.Whole.Psnr) && double.IsFinite(item.Seam.Psnr)));
    }

    [GpuFact]
    public async Task ProviderProfileSingleModel()
    {
        var target = await LoadTargetAsync();
        var providers = OrtEnv.Instance().GetAvailableProviders();
        output.WriteLine("availableProviders={0}", string.Join(',', providers));
        Assert.Contains("CUDAExecutionProvider", providers);
        var input = CreateInput(8, 8);
        var before = ReadVram();
        var result = DirectCudaRun(target.Directory, target.Inventory.Models[0], input, profile: true, out var profilePath);
        var during = ReadVram();
        Assert.Equal(3 * 16 * 16, result.Length);
        Assert.All(result, value => Assert.True(float.IsFinite(value) && value is >= 0 and <= 1));
        var actualProfilePath = profilePath ?? throw new InvalidOperationException("ORT did not return a profiling path.");
        var profile = await File.ReadAllTextAsync(actualProfilePath);
        using var document = JsonDocument.Parse(profile);
        var cudaNodes = CountProviderNodes(document.RootElement, "CUDAExecutionProvider");
        var cpuNodes = CountProviderNodes(document.RootElement, "CPUExecutionProvider");
        output.WriteLine("providerProfile={0} cudaNodes={1} cpuNodes={2} vramBeforeMiB={3} vramDuringMiB={4}", actualProfilePath, cudaNodes, cpuNodes, before, during);
        Assert.True(cudaNodes > 0, "ORT profile did not record CUDA-assigned graph nodes.");
    }

    [GpuFact]
    public async Task AllSevenRoutedSwitchesUseSingleActiveTensorRtSession()
    {
        var target = await LoadTargetAsync();
        var input = CreateInput(8, 8);
        var direct = DirectCudaRun(target.Directory, target.Inventory.Models[0], input, profile: false, out _);
        var baseline = ReadVram();
        await using var gpu = target.CreateGpu();
        if (target.Options.Models.PreloadOnStartup)
            await gpu.WarmAsync(CancellationToken.None);
        var cold = Stopwatch.StartNew();
        var first = await gpu.RunAsync(target.Inventory.Models[0].Id, input, 8, 8, CancellationToken.None);
        cold.Stop();
        var maximumDifference = direct.Zip(first, static (left, right) => Math.Abs(left - right)).Max();
        output.WriteLine("coldSevenSessionLoadAndFirstRunMs={0:F3} wrapperDirectMaxAbs={1:E8}", cold.Elapsed.TotalMilliseconds, maximumDifference);
        Assert.InRange(maximumDifference, 0f, 5e-6f);

        var routeHeights = new[] { 1200, 1251, 1351, 1451, 1551, 1761, 1985 };
        foreach (var (model, routeHeight) in target.Inventory.Models.Zip(routeHeights))
        {
            var selected = target.Inventory.Select(routeHeight);
            Assert.Equal(model.Id, selected.Id);
            var timer = Stopwatch.StartNew();
            var modelOutput = await gpu.RunAsync(selected.Id, input, 8, 8, CancellationToken.None);
            timer.Stop();
            Assert.Equal(3 * 16 * 16, modelOutput.Length);
            Assert.All(modelOutput, value => Assert.True(float.IsFinite(value) && value is >= 0 and <= 1));
            output.WriteLine("routeHeight={0} selectedModel={1} nominal={2} cachedSwitchAndRun8x8Ms={3:F3} vramMiB={4} min={5:F8} max={6:F8}",
                routeHeight, selected.Id, model.NominalHeight, timer.Elapsed.TotalMilliseconds, ReadVram(), modelOutput.Min(), modelOutput.Max());
        }
        output.WriteLine("vramBaselineMiB={0} vramCurrentResidentSetMiB={1}", baseline, ReadVram());
        Assert.Equal(GpuServiceState.Ready, gpu.State);
    }

    [GpuFact]
    public async Task ProductionTiling832By64IsSequential()
    {
        var target = await LoadTargetAsync();
        await using var gpu = target.CreateGpu();
        var source = Enumerable.Range(0, 900 * 1291).Select(x => (byte)(x % 256)).ToArray();
        var timer = Stopwatch.StartNew();
        var result = await new TiledUpscaler(gpu).UpscaleAsync(target.Inventory.Models[0].Id, source, 900, 1291, new(832, 64), CancellationToken.None);
        timer.Stop();
        Assert.Equal(1800 * 2582, result.Length);
        output.WriteLine("profile=832/64 runs=1 input=900x1291 output=1800x2582 tiles=4 elapsedMs={0:F3} vramMiB={1}", timer.Elapsed.TotalMilliseconds, ReadVram());
        Assert.Equal(1, gpu.MaximumObservedActiveRuns);
    }

    [GpuFact]
    public async Task ConcurrentCallersStillPermitOnlyOneGpuRun()
    {
        var target = await LoadTargetAsync();
        await using var gpu = target.CreateGpu();
        await gpu.WarmAsync(CancellationToken.None);
        output.WriteLine("postSequentialEngineValidationVramMiB={0}", ReadVram());
        const int extent = 320;
        var input = CreateInput(extent, extent);
        var timer = Stopwatch.StartNew();
        const int requestCount = 8;
        await Task.WhenAll(Enumerable.Range(0, requestCount).Select(_ => Task.Factory.StartNew(
            () => gpu.RunAsync(target.Inventory.Models[0].Id, input, extent, extent, CancellationToken.None),
            CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default).Unwrap()));
        timer.Stop();
        Assert.Equal(1, gpu.MaximumObservedActiveRuns);
        output.WriteLine("concurrentRequests={0} maximumObservedActiveRuns={1} elapsedMs={2:F3} vramMiB={3}", requestCount, gpu.MaximumObservedActiveRuns, timer.Elapsed.TotalMilliseconds, ReadVram());
    }

    [GpuFact]
    public async Task IdleDrainAndReload()
    {
        var target = await LoadTargetAsync();
        var input = CreateInput(8, 8);
        var baseline = ReadVram();
        await using var gpu = target.CreateGpu();
        await gpu.RunAsync(target.Inventory.Models[0].Id, input, 8, 8, CancellationToken.None);
        var loaded = ReadVram();
        await gpu.DrainIfIdleAsync(TimeSpan.Zero, CancellationToken.None);
        Assert.Equal(GpuServiceState.Uninitialized, gpu.State);
        await Task.Delay(500);
        var drained = ReadVram();
        var reload = Stopwatch.StartNew();
        var result = await gpu.RunAsync(target.Inventory.Models[^1].Id, input, 8, 8, CancellationToken.None);
        reload.Stop();
        Assert.Equal(3 * 16 * 16, result.Length);
        Assert.Equal(GpuServiceState.Ready, gpu.State);
        output.WriteLine("vramBaselineMiB={0} vramLoadedMiB={1} vramDrainedMiB={2} reloadMs={3:F3} vramReloadedMiB={4}", baseline, loaded, drained, reload.Elapsed.TotalMilliseconds, ReadVram());
    }

    [GpuFact]
    public async Task DisposalIsIdempotentAfterIdleDrain()
    {
        var target = await LoadTargetAsync();
        var gpu = target.CreateGpu();
        var input = CreateInput(8, 8);
        await gpu.RunAsync(target.Inventory.Models[0].Id, input, 8, 8, CancellationToken.None);
        await gpu.DrainIfIdleAsync(TimeSpan.Zero, CancellationToken.None);
        await gpu.DisposeAsync();
        await gpu.DisposeAsync();
        output.WriteLine("doubleDisposeAfterIdleDrain=passed vramMiB={0}", ReadVram());
    }

    private static async Task<TargetContext> LoadTargetAsync()
    {
        var directory = Environment.GetEnvironmentVariable("KOMASCALER_GPU_MODEL_DIR")!;
        var configuredManifest = Path.Combine(directory, "models.production.json");
        var manifest = File.Exists(configuredManifest) ? configuredManifest : Path.Combine(AppContext.BaseDirectory, "models", "models.production.json");
        var validation = await ModelInventoryLoader.LoadAsync(manifest, directory, verifyFiles: true).ConfigureAwait(false);
        Assert.True(validation.IsValid, string.Join(" | ", validation.Errors));
        var tensorRtCache = Environment.GetEnvironmentVariable("KOMASCALER_TENSORRT_CACHE") ??
            Path.Combine(Environment.GetEnvironmentVariable("KOMASCALER_ACCEPTANCE_ROOT") ?? "/devtmp/komascaler-acceptance", "tensorrt-engines");
        var profileOptimum = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_TENSORRT_PROFILE_OPTIMUM"), System.Globalization.CultureInfo.InvariantCulture, out var configuredOptimum) ? configuredOptimum : 384;
        var profileMaximum = int.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_TENSORRT_PROFILE_MAXIMUM"), System.Globalization.CultureInfo.InvariantCulture, out var configuredMaximum) ? configuredMaximum : 704;
        var preloadOnStartup = bool.TryParse(Environment.GetEnvironmentVariable("KOMASCALER_PRELOAD_ON_STARTUP"), out var configuredPreload) && configuredPreload;
        var options = new UpscalingOptions
        {
            Models = new ModelOptions
            {
                Directory = directory,
                InventoryFile = manifest,
                TensorRtCacheDirectory = tensorRtCache,
                TensorRtProfileOptimumExtent = profileOptimum,
                TensorRtProfileMaximumExtent = profileMaximum,
                PreloadOnStartup = preloadOnStartup
            }
        };
        var state = new InventoryState(options);
        await state.RefreshAsync().ConfigureAwait(false);
        Assert.True(state.IsReady, string.Join(" | ", state.Result.Errors));
        return new(directory, validation.Inventory!, options, state);
    }

    private static float[] DirectCudaRun(string directory, ModelDefinition model, float[] input, bool profile, out string? profilePath)
    {
        using var options = CreateCudaOptions(profile);
        using var session = new InferenceSession(Path.Combine(directory, model.File), options);
        using var value = OrtValue.CreateTensorValueFromMemory(input, [1, 3, 8, 8]);
        using var runOptions = new RunOptions();
        using var result = session.Run(runOptions, ["input"], [value], ["output"]);
        var output = result[0].GetTensorDataAsSpan<float>().ToArray();
        profilePath = profile ? session.EndProfiling() : null;
        return output;
    }

    private static float[] DirectCudaRun(InferenceSession session, float[] input, int height, int width)
    {
        using var value = OrtValue.CreateTensorValueFromMemory(input, [1, 3, height, width]);
        using var runOptions = new RunOptions();
        using var result = session.Run(runOptions, ["input"], [value], ["output"]);
        return result[0].GetTensorDataAsSpan<float>().ToArray();
    }

    private static Microsoft.ML.OnnxRuntime.SessionOptions CreateCudaOptions(bool profile)
    {
        var result = new Microsoft.ML.OnnxRuntime.SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        if (profile)
        {
            var acceptanceRoot = Environment.GetEnvironmentVariable("KOMASCALER_ACCEPTANCE_ROOT") ?? Path.GetTempPath();
            result.ProfileOutputPathPrefix = Path.Combine(acceptanceRoot, "komascaler-ort-profile-" + Guid.NewGuid().ToString("N"));
            result.EnableProfiling = true;
        }
        using var cuda = new OrtCUDAProviderOptions();
        cuda.UpdateOptions(new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["device_id"] = "0",
            ["cudnn_conv_algo_search"] = "HEURISTIC",
            ["use_tf32"] = "0"
        });
        result.AppendExecutionProvider_CUDA(cuda);
        return result;
    }

    private static Microsoft.ML.OnnxRuntime.SessionOptions CreateTensorRtOptions(ModelInventory inventory, ModelOptions modelOptions, string cache)
    {
        var result = new Microsoft.ML.OnnxRuntime.SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ProfileOutputPathPrefix = Path.Combine(cache, "preflight-profile"),
            EnableProfiling = true
        };
        using var tensorRt = new OrtTensorRTProviderOptions();
        tensorRt.UpdateOptions(new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["device_id"] = modelOptions.CudaDeviceId.ToString(CultureInfo.InvariantCulture),
            ["trt_fp16_enable"] = "0",
            ["trt_int8_enable"] = "0",
            ["trt_engine_cache_enable"] = "1",
            ["trt_engine_cache_path"] = cache,
            ["trt_timing_cache_enable"] = "1",
            ["trt_timing_cache_path"] = cache,
            ["trt_force_timing_cache"] = "0",
            ["trt_force_sequential_engine_build"] = "1",
            ["trt_auxiliary_streams"] = "0",
            ["trt_context_memory_sharing_enable"] = "1",
            ["trt_max_workspace_size"] = modelOptions.TensorRtMaxWorkspaceBytes.ToString(CultureInfo.InvariantCulture),
            ["trt_profile_min_shapes"] = $"{inventory.GraphContract.InputName}:1x3x8x8",
            ["trt_profile_opt_shapes"] = $"{inventory.GraphContract.InputName}:1x3x{modelOptions.TensorRtProfileOptimumExtent}x{modelOptions.TensorRtProfileOptimumExtent}",
            ["trt_profile_max_shapes"] = $"{inventory.GraphContract.InputName}:1x3x{modelOptions.TensorRtProfileMaximumExtent}x{modelOptions.TensorRtProfileMaximumExtent}"
        });
        result.AppendExecutionProvider_Tensorrt(tensorRt);
        return result;
    }

    private static float[] CreateInput(int height, int width) => Enumerable.Range(0, checked(3 * height * width)).Select(x => (x % 256) / 255f).ToArray();

    private static byte[] CreateQualityInput(int width, int height)
    {
        var result = new byte[checked(width * height)];
        for (var y = 0; y < height; y++)
            for (var x = 0; x < width; x++)
            {
                var checker = ((x / 8) + (y / 8)) % 2 == 0 ? 28 : 224;
                var line = x % 47 < 2 || y % 61 < 2 || Math.Abs((x % 113) - (y % 113)) < 2;
                var gradient = (x * 97 / Math.Max(1, width - 1)) + (y * 63 / Math.Max(1, height - 1));
                result[(y * width) + x] = (byte)(line ? 8 : Math.Clamp((checker * 3 + gradient) / 4, 0, 255));
            }
        return result;
    }

    private static float[] CreateNchwInput(byte[] source)
    {
        var result = new float[checked(source.Length * 3)];
        for (var i = 0; i < source.Length; i++)
            result[i] = result[source.Length + i] = result[(2 * source.Length) + i] = source[i] / 255f;
        return result;
    }

    private static byte[] QuantizeChannelZero(float[] source, int width, int height)
    {
        var result = new byte[checked(width * height)];
        for (var i = 0; i < result.Length; i++)
            result[i] = (byte)Math.Clamp((int)MathF.Round(Math.Clamp(source[i], 0f, 1f) * 255f), 0, 255);
        return result;
    }

    private static bool[] CreateSeamMask(int width, int height, TilingPolicy policy, int band)
    {
        var outputWidth = width * 2;
        var outputHeight = height * 2;
        var result = new bool[checked(outputWidth * outputHeight)];
        var tiles = BalancedTiler.Partition(width, height, policy);
        var vertical = tiles.Select(tile => tile.Core.Right * 2).Where(value => value < outputWidth).Distinct().ToArray();
        var horizontal = tiles.Select(tile => tile.Core.Bottom * 2).Where(value => value < outputHeight).Distinct().ToArray();
        for (var y = 0; y < outputHeight; y++)
            for (var x = 0; x < outputWidth; x++)
                result[(y * outputWidth) + x] = vertical.Any(value => Math.Abs(x - value) < band) || horizontal.Any(value => Math.Abs(y - value) < band);
        return result;
    }

    private static ErrorMetrics CalculateError(byte[] reference, byte[] candidate, bool[]? mask, int width, int height)
    {
        double absolute = 0, squared = 0, referenceMean = 0, candidateMean = 0;
        var maximum = 0;
        long count = 0;
        for (var i = 0; i < reference.Length; i++)
        {
            if (mask is not null && !mask[i]) continue;
            var difference = Math.Abs(reference[i] - candidate[i]);
            absolute += difference;
            squared += difference * difference;
            maximum = Math.Max(maximum, difference);
            referenceMean += reference[i];
            candidateMean += candidate[i];
            count++;
        }
        referenceMean /= count;
        candidateMean /= count;
        double referenceVariance = 0, candidateVariance = 0, covariance = 0;
        for (var i = 0; i < reference.Length; i++)
        {
            if (mask is not null && !mask[i]) continue;
            var left = reference[i] - referenceMean;
            var right = candidate[i] - candidateMean;
            referenceVariance += left * left;
            candidateVariance += right * right;
            covariance += left * right;
        }
        referenceVariance /= count;
        candidateVariance /= count;
        covariance /= count;
        var mse = squared / count;
        var psnr = mse == 0 ? 999 : 10 * Math.Log10((255d * 255d) / mse);
        const double c1 = 6.5025, c2 = 58.5225;
        var ssim = ((2 * referenceMean * candidateMean + c1) * (2 * covariance + c2)) /
                   ((referenceMean * referenceMean + candidateMean * candidateMean + c1) * (referenceVariance + candidateVariance + c2));
        return new(count, absolute / count, Math.Sqrt(mse), maximum, psnr, ssim);
    }

    private static FloatErrorMetrics CalculateFloatError(float[] reference, float[] candidate)
    {
        double absolute = 0, squared = 0, maximum = 0;
        for (var index = 0; index < reference.Length; index++)
        {
            var difference = Math.Abs(reference[index] - candidate[index]);
            absolute += difference;
            squared += difference * difference;
            maximum = Math.Max(maximum, difference);
        }
        var mse = squared / reference.Length;
        return new(absolute / reference.Length, maximum, mse == 0 ? 999 : 10 * Math.Log10(1 / mse));
    }

    private static int CountProviderNodes(JsonElement root, string provider)
    {
        var count = 0;
        foreach (var item in root.EnumerateArray())
            if (item.TryGetProperty("args", out var args) && args.TryGetProperty("provider", out var value) && string.Equals(value.GetString(), provider, StringComparison.Ordinal)) count++;
        return count;
    }

    private static int CountCpuComputeNodes(JsonElement root) => root.EnumerateArray().Count(item =>
        item.TryGetProperty("args", out var args) && args.TryGetProperty("provider", out var provider) &&
        string.Equals(provider.GetString(), "CPUExecutionProvider", StringComparison.Ordinal) && args.TryGetProperty("op_name", out var operation) &&
        operation.GetString() is "Conv" or "Gemm" or "MatMul");

    private static int ReadVram()
    {
        var start = new ProcessStartInfo("nvidia-smi", "--query-gpu=memory.used --format=csv,noheader,nounits") { RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
        using var process = Process.Start(start) ?? throw new InvalidOperationException("Could not start nvidia-smi.");
        var text = process.StandardOutput.ReadToEnd(); process.WaitForExit();
        if (process.ExitCode != 0) throw new InvalidOperationException(process.StandardError.ReadToEnd());
        return int.Parse(text.Trim().Split('\n')[0], CultureInfo.InvariantCulture);
    }

    private sealed record TargetContext(string Directory, ModelInventory Inventory, UpscalingOptions Options, InventoryState State)
    {
        public GpuUpscaler CreateGpu() => new(State, Microsoft.Extensions.Options.Options.Create(Options), NullLogger<GpuUpscaler>.Instance, new TestLifetime());
    }

    private sealed record QualityInput(string Name, int Width, int Height, byte[] Bytes);
    private sealed record ErrorMetrics(long Pixels, double Mae, double Rmse, int Maximum, double Psnr, double Ssim);
    private sealed record FloatErrorMetrics(double Mae, double Maximum, double Psnr);
    private sealed record QualityResult(string Input, int Width, int Height, int Core, int Context, int Tiles, double ElapsedMs, ErrorMetrics Whole, ErrorMetrics Seam);
}

public sealed class GpuFactAttribute : FactAttribute
{
    public GpuFactAttribute()
    {
        var directory = Environment.GetEnvironmentVariable("KOMASCALER_GPU_MODEL_DIR");
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory) || Directory.EnumerateFiles(directory, "*.onnx").Count() != 7)
            Skip = "Set KOMASCALER_GPU_MODEL_DIR to a directory containing all seven production ONNX files.";
    }
}

public sealed class GpuQualityFactAttribute : FactAttribute
{
    public GpuQualityFactAttribute()
    {
        var directory = Environment.GetEnvironmentVariable("KOMASCALER_GPU_MODEL_DIR");
        var output = Environment.GetEnvironmentVariable("KOMASCALER_QUALITY_OUTPUT");
        if (string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory) || Directory.EnumerateFiles(directory, "*.onnx").Count() != 7 || string.IsNullOrWhiteSpace(output))
            Skip = "Set KOMASCALER_GPU_MODEL_DIR and KOMASCALER_QUALITY_OUTPUT to run the untiled quality matrix.";
    }
}

public sealed class TensorRtFactAttribute : FactAttribute
{
    public TensorRtFactAttribute()
    {
        var directory = Environment.GetEnvironmentVariable("KOMASCALER_GPU_MODEL_DIR");
        if (!string.Equals(Environment.GetEnvironmentVariable("KOMASCALER_TENSORRT_ACCEPTANCE"), "1", StringComparison.Ordinal) ||
            !string.Equals(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"), "0", StringComparison.Ordinal) ||
            string.IsNullOrWhiteSpace(directory) || !Directory.Exists(directory) || Directory.EnumerateFiles(directory, "*.onnx").Count() != 7)
            Skip = "Set KOMASCALER_TENSORRT_ACCEPTANCE=1, NVIDIA_TF32_OVERRIDE=0, and KOMASCALER_GPU_MODEL_DIR to opt in.";
    }
}

internal sealed class TestLifetime : IHostApplicationLifetime
{
    private readonly CancellationTokenSource _started = new();
    private readonly CancellationTokenSource _stopping = new();
    private readonly CancellationTokenSource _stopped = new();
    public CancellationToken ApplicationStarted => _started.Token;
    public CancellationToken ApplicationStopping => _stopping.Token;
    public CancellationToken ApplicationStopped => _stopped.Token;
    public void StopApplication() => _stopping.Cancel();
}
