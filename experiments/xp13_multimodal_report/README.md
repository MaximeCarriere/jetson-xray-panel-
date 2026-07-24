# XP13 — On-device multimodal: X-ray → local LLM report

The whole pipeline, on one $249 Jetson, **fully offline**: a chest X-ray goes in, and
a **written clinical impression** comes out — vision *and* language, both on the box,
nothing sent to a cloud.

**"Multimodal"** just means two different *kinds* of data (modalities) in one pipeline:
here an **image** (the X-ray) and **text** (the written impression). Two different neural
networks — one that *sees*, one that *writes* — run back-to-back on the same GPU:

- **Stage 1 — vision:** the TensorRT DenseNet classifier (XP6) turns the image into 14
  pathology probabilities (~5 ms).
- **Stage 2 — language:** a small quantized *medical* LLM (**MedGemma 4B, Q4, text-only,
  on the GPU via llama.cpp**) turns those numbers into a plain-language impression, with
  wording calibrated to the probabilities (>0.6 "likely", 0.4–0.6 "possible", <0.4
  "unlikely").

The LLM never sees the raw image — it sees the *numbers* the vision model produced, plus a
prompt that tells it how to phrase each probability band. That's what keeps the language
faithful to the classifier instead of hallucinating findings: the two models are chained,
vision → numbers → language.

> **Not a clinical tool.** The impression is written by a small general model from
> model outputs — a demonstration of the *systems* capability (local vision + language
> on edge hardware), not a validated report. Ground truth is shown so you can see what
> the vision model got right.

## Real output (on the board)

```
case 1 (test #296)
  ground truth:      effusion, consolidation
  top predictions:   effusion 0.95, infiltration 0.34, cardiomegaly 0.22
  IMPRESSION (LLM):  The study shows a likely effusion.
  [classify 5.9 ms · generate 1.1 s · 8 tok · 7.1 tok/s]

case 3 (test #469)
  ground truth:      pleural thickening
  top predictions:   mass 0.64, effusion 0.63, infiltration 0.41
  IMPRESSION (LLM):  The study shows a likely mass and effusion, with possible /
                     borderline infiltration.
  [classify 6.1 ms · generate 1.1 s · 17 tok · 15.4 tok/s]
```

The bands (> 0.6 = likely, 0.4–0.6 = possible) are decided **in code**; the LLM only
phrases them, so the language tracks the numbers exactly (0.64/0.63 → "likely", 0.41 →
"possible"). This matters: left to judge the raw scores *itself*, MedGemma — a medical
model — second-guesses them and quietly drops findings (e.g. omitting a 0.88 effusion, or
calling a study "unremarkable" over a listed finding). Deciding the bands in code and
leaving only the wording to the model is what keeps it faithful. A "smarter" model is not
automatically better for a faithfulness-critical pipeline.

## The interactive reading station

The pipeline is packaged as a self-contained, offline HTML **reading station**
([`demos/report_station.html`](../../demos/report_station.html)) — a clinical-style viewer
you can open in any browser with no server. Three panels, left to right: the **X-ray**
(with ground-truth tags), the vision model's **14 pathology probabilities** as bars, and
the local LLM's **written impression** typing out to evoke on-device generation. A footer
shows live telemetry (classify ms · generate s · tokens/s), and you can flip through **6
real cases** captured from the board. The pipeline strip along the top spells out
`chest X-ray → TensorRT DenseNet → 14 probabilities → MedGemma 4B (llama.cpp, GPU) → impression`.

![on-device reading station](../../demos/report_station.png)

*One $249 box, fully offline: image in, calibrated written impression out. The bar colours
encode the wording bands (orange = likely >0.6, amber = possible 0.4–0.6, grey = unlikely).
Ground truth is shown so the vision model's hits and misses are visible — it's a **systems
demonstration, not a clinical tool**.*

It's built from `results/report_cases.json`; `report.py --html-out demos/report_station.html`
regenerates this page directly from a fresh run, injecting the cases **and** the model name
(the labels are data-driven), so every number, sentence, and label on the page is a **real**
on-board result, not a mockup.

## Performance
- **Vision:** ~6 ms/image (TensorRT FP16).
- **Language:** MedGemma 4B Q4 (text-only) on the GPU at **~15 tokens/s**, ~0.6–1.1 s for a
  1-2 sentence impression. LLM loads in ~4 s.
- **End-to-end: ~1–2 s per report, entirely on-device.** Both models fit in 8 GB
  (TRT engine ~15 MB + MedGemma 4B Q4 ~2.5 GB).

The edge-AI point: one cheap box runs a vision model **and** a language model together,
locally and privately — the "no cloud, patient data never leaves the building" story,
made real end to end.

## Setup + run
```bash
# llama.cpp with CUDA (point cmake at nvcc):
export CUDACXX=/usr/local/cuda-12.6/bin/nvcc
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=87" \
    ~/xray-venv/bin/pip install llama-cpp-python

# get a MedGemma 4B Q4 GGUF into ~/models/ (e.g. unsloth/medgemma-4b-it-GGUF), then:
~/xray-venv/bin/python report.py --n 6 \
    --export ../../results/report_cases.json \
    --html-out ../../demos/report_station.html
```

## Files
`report.py` (classify → band in code → LLM phrases the bands; `--export` writes
`results/report_cases.json`, `--html-out` rebuilds the reading station). Uses
`lib/trt_runner.py` + `lib/chest_labels.py`; model `~/models/medgemma-4b-it-Q4_K_M.gguf`
(override with `--llm-path`). Interactive demo:
[`demos/report_station.html`](../../demos/report_station.html) (screenshot above).
