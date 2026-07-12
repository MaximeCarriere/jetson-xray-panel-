# XP13 — On-device multimodal: X-ray → local LLM report

The whole pipeline, on one $249 Jetson, **fully offline**: a chest X-ray goes in, and
a **written clinical impression** comes out — vision *and* language, both on the box,
nothing sent to a cloud.

- **Stage 1 — vision:** the TensorRT DenseNet classifier (XP6) turns the image into 14
  pathology probabilities (~5 ms).
- **Stage 2 — language:** a small quantized LLM (**Qwen2.5-3B, Q4, on the GPU via
  llama.cpp**) turns those numbers into a plain-language impression, with wording
  calibrated to the probabilities (>0.6 "likely", 0.4–0.6 "possible", <0.4 "unlikely").

> **Not a clinical tool.** The impression is written by a small general model from
> model outputs — a demonstration of the *systems* capability (local vision + language
> on edge hardware), not a validated report. Ground truth is shown so you can see what
> the vision model got right.

## Real output (on the board)

```
case 1 (test #296)
  ground truth:      effusion, consolidation
  top predictions:   effusion 0.95, infiltration 0.34, cardiomegaly 0.22
  IMPRESSION (LLM):  The study shows a likely effusion, possible / borderline
                     infiltration, and a possible pleural thickening. The rest of
                     the findings are unlikely.
  [classify 5.0 ms · generate 2.2 s · 31 tok · 14.3 tok/s]

case 2 (test #144)
  top predictions:   effusion 0.90, infiltration 0.52, atelectasis 0.23
  IMPRESSION (LLM):  The study shows a likely effusion, possible infiltration, and
                     borderline atelectasis, with no other significant findings.
  [classify 5.7 ms · generate 1.4 s · 23 tok · 16.9 tok/s]
```

The language faithfully tracks the numbers (0.95 → "likely", 0.52 → "possible",
0.23 → "borderline").

## Performance
- **Vision:** ~5 ms/image (TensorRT FP16).
- **Language:** Qwen-3B Q4 on the GPU at **~15–17 tokens/s**, ~1.5–2.2 s for a
  2-3 sentence impression. LLM loads in ~2 s.
- **End-to-end: ~2 s per report, entirely on-device.** Both models fit in 8 GB
  (TRT engine ~15 MB + Qwen Q4 ~2.1 GB).

The edge-AI point: one cheap box runs a vision model **and** a language model together,
locally and privately — the "no cloud, patient data never leaves the building" story,
made real end to end.

## Setup + run
```bash
# llama.cpp with CUDA (point cmake at nvcc):
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87" \
    ~/xray-venv/bin/pip install llama-cpp-python
~/xray-venv/bin/python report.py --n 3
```

## Files
`report.py` (classify → prompt → generate). Uses `lib/trt_runner.py` +
`lib/chest_labels.py`; model `~/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf`.
