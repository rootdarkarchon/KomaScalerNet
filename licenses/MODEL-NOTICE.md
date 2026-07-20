# MangaJaNai model notice

The seven bundled ONNX files are format-converted derivatives of the official
MangaJaNai V1 2× black-and-white models by `the-database`.

- Source project: https://github.com/the-database/MangaJaNai
- Source release: https://github.com/the-database/MangaJaNai/releases/tag/1.0.0
- License: Creative Commons Attribution-NonCommercial 4.0 International
- Modification: official PyTorch state dictionaries converted to dynamic-shape
  ONNX, opset 17, with FP16 weights/input/output and an explicit
  reflect-pad/pixel-unshuffle/clamp/crop boundary.
- Intended use of this package: local, noncommercial compatibility validation.

The complete license text is included as `MODEL-LICENSE.txt`. Source and ONNX
SHA-256 values are recorded in `models/models-export-proof.json` and are
verified again during the validation run.
