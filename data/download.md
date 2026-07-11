# Dataset

**Important:** the core throughput / power / latency metrics are
**content-agnostic** — only the input tensor *shape* affects GPU work. So the
benchmark runs on a fixed pool of 224×224 images and, if no real images are
present, `utils.build_input_pool()` synthesizes noise of the correct shape (with
a note in the code). Real labelled images are only needed for the **stretch**
accuracy / TTA experiment.

## Option A — Kaggle Chest X-Ray (Pneumonia)

~5,856 images, ~1.2 GB, binary normal/pneumonia (Kermany et al.). Needs a Kaggle
account + `~/.kaggle/kaggle.json` API token.

```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p data/
unzip -q data/chest-xray-pneumonia.zip -d data/
# then point the benchmark at the images:
#   python src/benchmark.py --image-dir data/chest_xray/train
```

## Option B — torchxrayvision bundled samples / NIH subset

For multi-pathology exercising, use a small NIH ChestX-ray14 sample or the images
bundled with torchxrayvision. Sufficient for the systems benchmark.

`data/` is gitignored (see `.gitignore`) — do **not** commit images.
