# Offline model tools

These scripts are evidence and maintenance tools, not production dependencies.
They inspect official `.pth` checkpoints, export the exact RRDB graph to FP32
ONNX, and compare PyTorch/CUDA against ONNX Runtime/CUDA.

The accepted production ONNX files were generated on the target host and have
the hashes in `../../models/models.production.json`. If the exporter,
PyTorch/Spandrel/ONNX versions, or source model changes, regenerate all hashes
and rerun the Step 5 and Step 7 validation gates before deployment.
