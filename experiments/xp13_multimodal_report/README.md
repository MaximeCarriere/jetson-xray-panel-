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
  on the GPU via llama.cpp**) drafts the **next steps** and the **clinical considerations**.

The report has three parts. The **diagnostic** (findings line) is composed **in code** from
the probabilities (>0.6 "likely", 0.4–0.6 "possible", <0.4 "unlikely") — the LLM never writes
it, and never sees the raw image. The LLM is handed only that finished sentence and, in one
call, returns two labelled lines: a **specific next step** to confirm the findings (CT,
ultrasound, a lateral decubitus view…) and a **clinical consideration**. That split is what
keeps the report faithful to the classifier instead of hallucinating findings.

> **Not a clinical tool.** The findings come from a pretrained classifier and the wording
> from a small general model — a demonstration of the *systems* capability (local vision +
> language on edge hardware), not a validated report.

## Real output (on the board)

```
case 1 (test #296)
  top predictions:     effusion 0.95, infiltration 0.34, cardiomegaly 0.22
  DIAGNOSTIC (code):   The study shows a likely effusion.
  NEXT STEPS (LLM):    A lateral decubitus view would help confirm the effusion.
  CONSIDERATIONS (LLM):If the effusion is large, a thoracentesis may be needed.
  [classify 5.5 ms · generate 2.9 s · 36 tok · 12.5 tok/s]

case 2 (test #517)
  top predictions:     cardiomegaly 0.88, effusion 0.29, infiltration 0.17
  DIAGNOSTIC (code):   The study shows a likely cardiomegaly.
  NEXT STEPS (LLM):    Echocardiogram to assess cardiac function and chamber sizes.
  CONSIDERATIONS (LLM):Assess for pulmonary edema if the patient is symptomatic.
  [classify 6.0 ms · generate 1.9 s · 31 tok · 16.2 tok/s]
```

**Diagnostic in code, the rest by the model.** The diagnostic (findings) sentence is
composed entirely in code from the bands (> 0.6 = likely, 0.4–0.6 = possible), so it tracks
the numbers exactly. MedGemma is handed only that sentence and, in **one call**, returns two
labelled lines — a specific next step and a clinical consideration. The split is deliberate:
asked to write the findings *itself*, MedGemma — a medical model — second-guesses the
numbers. It invents findings that are not there (a phantom "pleural effusion"), re-bands a
0.63 effusion as "possible", and calls a study "unremarkable" over a listed finding. Keeping
the diagnostic in code, and letting the model add only the next step and the consideration,
is what keeps it faithful. (A JSON grammar gave the same split but ran ~3× slower, so the two
sections are parsed from two labelled lines.)

## The interactive reading station

The pipeline is packaged as a self-contained, offline HTML **reading station**
([`demos/report_station.html`](../../demos/report_station.html)) — a clinical-style viewer
you can open in any browser with no server. Three panels, left to right: the **X-ray**,
the vision model's **top pathology probabilities** as bars (top 6, expandable to all 14),
and the **report** in three labelled boxes — **current diagnostic** (from the vision model),
**next steps to confirm**, and **clinical considerations** (both from MedGemma), the last two
typing out to evoke on-device generation. Each box shows its source. A footer shows live
telemetry (classify ms · generate s · tokens/s), and you can flip through **6 real cases**
captured from the board. The pipeline strip along the top spells out
`chest X-ray → TensorRT DenseNet → 14 probabilities → MedGemma 4B (llama.cpp, GPU) → report`.

![on-device reading station](../../demos/report_station.png)

*One $249 box, fully offline: image in, written report out. The bar colours encode the
wording bands (orange = likely >0.6, amber = possible 0.4–0.6, grey = unlikely). A **systems
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
