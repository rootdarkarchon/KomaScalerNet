# Third-party notices

KomaScaler.Net does not claim ownership of third-party projects and does not redistribute the model checkpoints, generated ONNX models, TensorRT engines, or NVIDIA runtime packages.

## MangaJaNai models

The seven source checkpoints are published by the independent [MangaJaNai project](https://github.com/the-database/MangaJaNai) in its [1.0.0 release](https://github.com/the-database/MangaJaNai/releases/tag/1.0.0). This repository supplies conversion/integration tooling only and is not affiliated with, endorsed by, or supported by MangaJaNai's authors. Operators must review the upstream repository/release terms themselves before downloading, converting, using, or redistributing the checkpoints or derived ONNX files. This notice makes no licensing conclusion.

## Runtime and build components

- [.NET and ASP.NET Core](https://github.com/dotnet/runtime) are supplied under their upstream licenses; they are not vendored here.
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) supplies the `Microsoft.ML.OnnxRuntime.Gpu` package under its upstream MIT license. The package is used for the TensorRT Execution Provider; the CPU-only package is not referenced.
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt), CUDA, and cuDNN are separately installed vendor components governed by NVIDIA's applicable terms. They are not redistributed by this repository.
- [libvips](https://github.com/libvips/libvips) is a separately installed image-processing library governed by its upstream LGPL-2.1-or-later license.
- [NetVips](https://github.com/kleisauke/net-vips) is the managed libvips binding and is consumed from NuGet under its upstream MIT license.
- [Meziantou.Analyzer](https://github.com/meziantou/Meziantou.Analyzer) is a build-time analyzer consumed from NuGet under its upstream MIT license and is excluded from application publish output.

Transitive package notices remain available through their package metadata and upstream repositories. The repository's MIT license covers only KomaScaler.Net's original source and does not replace these third-party terms.
