# Research archive

These are the original investigation reports and work brief. They preserve
source citations, measurements, and reasoning.

Architecture evolved during the investigation. When wording conflicts, use
this precedence:

1. `../../design/FINAL-ARCHITECTURE.md`
2. `../CODEX-PROMPT.md`
3. Step 7
4. Steps 1–6
5. the original brief

In particular, early references to a separate worker process, one model loaded
at a time, FP16, fixed 16-pixel context, or process restart for tiling changes
are superseded. The final design is one ASP.NET process, seven resident FP32
sessions, serialized CUDA, and live 320/32 defaults.
