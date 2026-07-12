"""XP13 — on-device multimodal: chest X-ray -> pathology probabilities -> local LLM
report, all on one $249 Jetson, fully offline.

Stage 1 (vision): the TensorRT DenseNet classifier turns an image into 14 pathology
probabilities (XP6). Stage 2 (language): a small quantized LLM (Qwen2.5-3B, Q4, on
the GPU via llama.cpp) turns those numbers into a plain-language clinical impression.
Nothing leaves the box.

*** NOT a clinical tool. *** The impression is written by a small general model from
model outputs — a demonstration of the *systems* capability (local vision + language
on edge hardware), not a validated radiology report. Ground-truth labels are shown
so the reader can see what the vision model got right/wrong.

    ~/xray-venv/bin/python report.py --n 4
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "lib"))

import numpy as np
import torch

import chest_labels as cl
from trt_runner import TRTModel

ENGINE = "/home/a/densenet_nih_fp16.engine"
LLM_PATH = "/home/a/models/qwen2.5-coder-3b-instruct-q4_k_m.gguf"

# Human-friendly names for the prompt/output.
NICE = {"pleural": "pleural thickening"}


def classify(engine: TRTModel, cmap, img255: np.ndarray):
    """One image (H,W uint8) -> dict{pathology: probability}."""
    x = torch.from_numpy((2.0 * (img255.astype(np.float32) / 255.0) - 1.0) * 1024.0)
    x = x[None, None].cuda()
    logits = engine.infer(x)
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    out = {}
    for model_col, med_col in cmap:
        out[cl.MEDMNIST_LABELS[med_col]] = float(probs[model_col])
    return out


def build_messages(probs: dict):
    findings = "\n".join(
        f"- {NICE.get(k, k).capitalize()}: {v:.2f}"
        for k, v in sorted(probs.items(), key=lambda kv: -kv[1])
    )
    system = ("You are a radiology assistant. You are given a chest X-ray model's "
              "predicted probability (0-1) that each pathology is present. Calibrate "
              "your wording to the numbers: probability > 0.6 = 'likely'; 0.4-0.6 = "
              "'possible / borderline'; < 0.4 = 'unlikely' (do not call it a finding). "
              "Write a brief 2-3 sentence clinical impression in plain English. If "
              "nothing exceeds 0.4, say the study appears largely unremarkable. Do NOT "
              "invent findings not in the list. Be concise.")
    user = f"Chest X-ray model outputs:\n{findings}\n\nImpression:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _img_datauri(img255: np.ndarray) -> str:
    """Grayscale uint8 -> contrast-stretched PNG data URI for display."""
    from PIL import Image
    a = img255.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / max(1e-3, hi - lo) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="sample images to report on")
    ap.add_argument("--max-tokens", type=int, default=120)
    ap.add_argument("--export", default=None, help="write cases (with images) to JSON")
    args = ap.parse_args()

    from llama_cpp import Llama
    from medmnist import ChestMNIST

    print("Loading vision engine + LLM…", flush=True)
    engine = TRTModel(ENGINE)
    import models
    cmap = cl.col_map(models.load_model("densenet121-res224-nih", "cpu"))  # only need .pathologies
    t_llm = time.perf_counter()
    llm = Llama(model_path=LLM_PATH, n_gpu_layers=-1, n_ctx=1024, verbose=False)
    print(f"  LLM loaded in {time.perf_counter()-t_llm:.1f}s\n", flush=True)

    ds = ChestMNIST(split="test", size=224, download=True)
    # Pick images the vision model is most confident about (most definitive reports).
    scan = 300
    maxprob = np.array([max(classify(engine, cmap, ds.imgs[i]).values()) for i in range(scan)])
    idxs = list(np.argsort(maxprob)[::-1][:args.n])

    cases = []
    for k, i in enumerate(idxs):
        img = ds.imgs[i]
        truth = [cl.MEDMNIST_LABELS[j] for j in range(14) if ds.labels[i][j] == 1]

        t0 = time.perf_counter()
        probs = classify(engine, cmap, img)
        t_classify = (time.perf_counter() - t0) * 1000

        msgs = build_messages(probs)
        t1 = time.perf_counter()
        resp = llm.create_chat_completion(msgs, max_tokens=args.max_tokens, temperature=0.3)
        t_gen = time.perf_counter() - t1
        text = resp["choices"][0]["message"]["content"].strip()
        n_tok = resp["usage"]["completion_tokens"]

        top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
        print(f"===== case {k+1} (test #{i}) =====")
        print(f"  ground truth: {', '.join(truth) if truth else '(none labelled)'}")
        print(f"  top predictions: " + ", ".join(f"{n} {p:.2f}" for n, p in top))
        print(f"  IMPRESSION (local LLM): {text}")
        print(f"  [classify {t_classify:.1f} ms · generate {t_gen:.1f} s "
              f"· {n_tok} tok · {n_tok/t_gen:.1f} tok/s]\n", flush=True)

        if args.export:
            cases.append({
                "id": int(i),
                "image": _img_datauri(img),
                "ground_truth": [NICE.get(t, t) for t in truth],
                "probs": {NICE.get(k2, k2): round(v, 3) for k2, v in probs.items()},
                "impression": text,
                "classify_ms": round(t_classify, 1),
                "generate_s": round(t_gen, 2),
                "tokens": int(n_tok),
                "tok_s": round(n_tok / t_gen, 1),
            })

    if args.export:
        with open(args.export, "w") as f:
            json.dump({"model_vision": "densenet121-res224-nih (TensorRT FP16)",
                       "model_language": "Qwen2.5-3B Q4 (llama.cpp, GPU)",
                       "cases": cases}, f)
        print(f"exported {len(cases)} cases -> {args.export}")


if __name__ == "__main__":
    main()
