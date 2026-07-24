"""XP13 — on-device multimodal: chest X-ray -> pathology probabilities -> local LLM
report, all on one $249 Jetson, fully offline.

Stage 1 (vision): the TensorRT DenseNet classifier turns an image into 14 pathology
probabilities (XP6). Stage 2 (language): a small quantized medical LLM (MedGemma 4B,
Q4, text-only, on the GPU via llama.cpp) turns those numbers into a plain-language
clinical impression. Nothing leaves the box.

The language model is *text-only* here: it never sees the image, only the vision
model's numbers plus a rule for how to phrase each probability band. That is what
keeps the wording faithful to the classifier instead of inventing findings.

*** NOT a clinical tool. *** The impression is written by a small general medical model
from model outputs — a demonstration of the *systems* capability (local vision +
language on edge hardware), not a validated radiology report. Ground-truth labels are
shown so the reader can see what the vision model got right/wrong.

    # regenerate the six cases and rebuild the offline reading station in one shot:
    ~/xray-venv/bin/python report.py --n 6 \
        --export ../../results/report_cases.json \
        --html-out ../../demos/report_station.html
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

# Text-only medical writer. MedGemma 4B (instruction-tuned, Google's medical Gemma),
# quantized to Q4 and run on the GPU via llama.cpp. It is fed ONLY the vision model's
# numbers, never the image. Point --llm-path / --llm-name at whatever GGUF is on the box.
LLM_PATH = "/home/a/models/medgemma-4b-it-Q4_K_M.gguf"
LLM_NAME = "MedGemma 4B (Q4, llama.cpp, GPU)"

_STATION = os.path.join(_HERE, "..", "..", "demos", "report_station.html")

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


def band_findings(probs: dict):
    """Split pathologies into the two reportable bands (>0.6 likely, 0.4-0.6 possible),
    most-confident first."""
    likely = sorted([(k, v) for k, v in probs.items() if v > 0.6], key=lambda kv: -kv[1])
    possible = sorted([(k, v) for k, v in probs.items() if 0.4 <= v <= 0.6], key=lambda kv: -kv[1])
    return likely, possible


def _phrase(items, adj):
    names = [NICE.get(k, k) for k, _ in items]
    if not names:
        return ""
    if len(names) == 1:
        return f"a {adj} {names[0]}"
    if len(names) == 2:
        return f"{adj} {names[0]} and {names[1]}"
    return f"{adj} " + ", ".join(names[:-1]) + f", and {names[-1]}"


def findings_sentence(likely, possible):
    """Compose the findings **in code** so they track the numbers exactly. Left to write
    findings freely, MedGemma invents extras (a phantom "pleural effusion"), re-bands
    (a 0.63 effusion becomes "possible"), and prepends "unremarkable" over real findings.
    Deterministic phrasing removes all of that: the model never authors a finding."""
    likely_txt = _phrase(likely, "likely")
    possible_txt = _phrase(possible, "possible / borderline")
    if likely_txt and possible_txt:
        return f"The study shows {likely_txt}, with {possible_txt}."
    if likely_txt:
        return f"The study shows {likely_txt}."
    if possible_txt:
        return f"The study shows {possible_txt}."
    return "The study appears largely unremarkable."


def report_messages(findings: str, has_findings: bool):
    """One LLM call for the two *generated* sections of the report: the specific next
    steps, and the clinical considerations. The model is handed only the code-composed
    findings sentence (never the raw numbers, never the image), so it tailors advice to
    real findings without authoring or re-banding them. Two labelled lines out (a JSON
    grammar constraint worked but ran ~3x slower), parsed into their own boxes."""
    if has_findings:
        instr = ("You are a radiology assistant. Given the chest X-ray impression below, "
                 "reply with exactly these two lines and nothing else:\n"
                 "NEXT STEPS: one sentence naming the most appropriate specific next "
                 "investigation to confirm or characterise the findings (e.g. CT, "
                 "ultrasound, a lateral decubitus view, or comparison with prior imaging).\n"
                 "CONSIDERATIONS: one sentence of brief clinical considerations or cautions "
                 "relevant to the findings, phrased as considerations, not directives.\n"
                 "Base both only on the findings in the impression. Do not invent findings.")
    else:
        instr = ("You are a radiology assistant. The chest X-ray has no significant "
                 "findings. Reply with exactly these two lines and nothing else:\n"
                 "NEXT STEPS: one sentence recommending routine follow-up as clinically "
                 "indicated.\n"
                 "CONSIDERATIONS: one short sentence, e.g. correlate with the clinical "
                 "presentation.")
    return [{"role": "user", "content": f"{instr}\n\nImpression: {findings}\n\n"}]


def _img_datauri(img255: np.ndarray) -> str:
    """Grayscale uint8 -> contrast-stretched PNG data URI for display."""
    from PIL import Image
    a = img255.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / max(1e-3, hi - lo) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def write_station(template_path: str, payload: dict, out_path: str) -> None:
    """Rebuild the self-contained offline reading station: read the HTML template and
    replace its single `const DATA = …;` line with these cases. The page renders the
    model names from that data, so a model swap never leaves stale labels."""
    with open(template_path) as f:
        lines = f.read().splitlines()
    blob = "const DATA = " + json.dumps(payload) + ";"
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("const DATA ="):
            lines[i] = blob
            break
    else:
        raise SystemExit(f"{template_path}: no 'const DATA =' line to replace")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"built reading station -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="sample images to report on")
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--llm-path", default=LLM_PATH, help="GGUF path for the text writer")
    ap.add_argument("--llm-name", default=LLM_NAME, help="label recorded in the output")
    ap.add_argument("--export", default=None, help="write cases (with images) to JSON")
    ap.add_argument("--html-out", default=None,
                    help="write a self-contained report_station.html with these cases")
    ap.add_argument("--html-template", default=_STATION,
                    help="reading-station HTML template to inject the cases into")
    args = ap.parse_args()

    want_cases = bool(args.export or args.html_out)

    from llama_cpp import Llama
    from medmnist import ChestMNIST

    print("Loading vision engine + LLM…", flush=True)
    engine = TRTModel(ENGINE)
    import models
    cmap = cl.col_map(models.load_model("densenet121-res224-nih", "cpu"))  # only need .pathologies
    t_llm = time.perf_counter()
    llm = Llama(model_path=args.llm_path, n_gpu_layers=-1, n_ctx=1024, verbose=False)
    print(f"  {args.llm_name} loaded in {time.perf_counter()-t_llm:.1f}s\n", flush=True)

    ds = ChestMNIST(split="test", size=224, download=True)
    # Diversify the panel: take the single most-confident example of each *distinct* top
    # pathology, so the cases show different findings rather than six near-identical
    # effusions (the model's most-confident class dominates a naive top-N).
    scan = 600
    tops = []  # (image_index, top_pathology, top_prob)
    for i in range(scan):
        p = classify(engine, cmap, ds.imgs[i])
        lab, pr = max(p.items(), key=lambda kv: kv[1])
        tops.append((i, lab, pr))
    best = {}  # top_pathology -> (image_index, prob), keeping the most confident per class
    for i, lab, pr in tops:
        if lab not in best or pr > best[lab][1]:
            best[lab] = (i, pr)
    idxs = [i for i, _ in sorted(best.values(), key=lambda t: -t[1])[:args.n]]
    if len(idxs) < args.n:  # not enough distinct classes: top up with next most-confident
        idxs += [i for i, _, _ in sorted(tops, key=lambda t: -t[2]) if i not in idxs][:args.n - len(idxs)]

    cases = []
    for k, i in enumerate(idxs):
        img = ds.imgs[i]
        truth = [cl.MEDMNIST_LABELS[j] for j in range(14) if ds.labels[i][j] == 1]

        t0 = time.perf_counter()
        probs = classify(engine, cmap, img)
        t_classify = (time.perf_counter() - t0) * 1000

        likely, possible = band_findings(probs)
        findings = findings_sentence(likely, possible)
        msgs = report_messages(findings, bool(likely or possible))
        t1 = time.perf_counter()
        resp = llm.create_chat_completion(msgs, max_tokens=args.max_tokens, temperature=0.2)
        t_gen = time.perf_counter() - t1
        raw = resp["choices"][0]["message"]["content"].strip()
        next_steps, considerations = "", ""
        for line in raw.splitlines():
            s = line.strip().lstrip("-*• ").strip()
            up = s.upper()
            if up.startswith("NEXT STEP"):
                next_steps = s.split(":", 1)[1].strip() if ":" in s else s
            elif up.startswith("CONSIDERATION"):
                considerations = s.split(":", 1)[1].strip() if ":" in s else s
        if not next_steps and not considerations:
            next_steps = raw
        n_tok = resp["usage"]["completion_tokens"]

        top = sorted(probs.items(), key=lambda kv: -kv[1])[:4]
        print(f"===== case {k+1} (test #{i}) =====")
        print(f"  ground truth: {', '.join(truth) if truth else '(none labelled)'}")
        print(f"  top predictions: " + ", ".join(f"{n} {p:.2f}" for n, p in top))
        print(f"  DIAGNOSTIC (code): {findings}")
        print(f"  NEXT STEPS (LLM):  {next_steps}")
        print(f"  CONSIDERATIONS (LLM): {considerations}")
        print(f"  [classify {t_classify:.1f} ms · generate {t_gen:.1f} s "
              f"· {n_tok} tok · {n_tok/t_gen:.1f} tok/s]\n", flush=True)

        if want_cases:
            cases.append({
                "id": int(i),
                "image": _img_datauri(img),
                "probs": {NICE.get(k2, k2): round(v, 3) for k2, v in probs.items()},
                "findings": findings,
                "next_steps": next_steps,
                "considerations": considerations,
                "classify_ms": round(t_classify, 1),
                "generate_s": round(t_gen, 2),
                "tokens": int(n_tok),
                "tok_s": round(n_tok / t_gen, 1),
            })

    if want_cases:
        payload = {"model_vision": "densenet121-res224-nih (TensorRT FP16)",
                   "model_language": args.llm_name, "cases": cases}
        if args.export:
            with open(args.export, "w") as f:
                json.dump(payload, f)
            print(f"exported {len(cases)} cases -> {args.export}")
        if args.html_out:
            write_station(args.html_template, payload, args.html_out)


if __name__ == "__main__":
    main()
