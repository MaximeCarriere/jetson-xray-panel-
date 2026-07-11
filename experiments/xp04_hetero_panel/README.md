# XP4 — Realistic mixed-architecture panel

A genuinely heterogeneous panel — different architectures *and* sizes, all detecting
chest disease: **ResNet-50 (23.5 M, 512 px) + 3 DenseNet-121 (6.97 M, 224 px)**,
concurrent, batch 1.

## Result
| Model | Params / input | Throughput | Latency |
|---|---|---:|---:|
| ResNet-50 | 23.5 M / 512 px | 30 img/s | 33 ms |
| DenseNet-121 ×3 | 6.97 M / 224 px | 15 img/s each | 66 ms |
| **Aggregate** | | **75.5 img/s** | GPU 97% |

**Counter-intuitive:** the bigger 512 px ResNet-50 is the *fastest* per image
(33 ms vs 66 ms). DenseNet-121's many small dense-block kernels are launch- and
bandwidth-bound and contend more under concurrency; ResNet-50's fewer, larger
kernels run efficiently even at 4× the pixels. A "heavier" model is not the bottleneck.

(Aside: nearly all published chest-disease models are DenseNet-121 — CheXNet made it
the standard — so ResNet-50 is the only distinct architecture with real chest weights.)

## Run (uses XP2's orchestrator with --panel)
```bash
setsid bash ../xp02_concurrency/run_concurrent.sh --repeats 3 --duration 8 \
    --panel "resnet50-res512-all,densenet121-res224-nih,densenet121-res224-chex,densenet121-res224-pc"
```
