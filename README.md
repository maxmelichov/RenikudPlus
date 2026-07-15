# ReNikud Plus — Hebrew Grapheme-to-Phoneme Inference

Convert unvocalized Hebrew text into IPA for TTS, speech technology, and
spoken-language research.

## Benchmark

![G2P benchmark comparison](assets/bar_plot_comparison.jpeg)

## Install

This repository contains only the code required for ONNX inference. It has no
training pipeline, checkpoint files, or PyTorch dependency.

```console
uv sync
hf download notmax123/RenikudPlus model.onnx --local-dir .
```

## Usage

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃlˈom ʔolˈam
```

For a gender-conditioned ONNX model, pass `speaker` and `target_speaker` as
`0` (unknown), `1` (male), or `2` (female):

```python
g2p.phonemize("היא רצה", speaker=2, target_speaker=2)
```

### Niqqud output

`vocalize` renders the same predictions as pointed Hebrew (niqqud) instead of
IPA — for TTS engines that read niqqud natively but ignore phoneme markup. It
accepts the same `speaker` / `target_speaker` arguments.

```python
print(g2p.vocalize("שלום עולם"))
# → שׁלוֹם עוֹלַם
```

Niqqud has no stress mark, so predicted stress is not represented in this output
(it is in `phonemize`). Diacritization is phonetically faithful but not
publication-grade — e.g. shva in clusters is omitted.

## Performance in containers

onnxruntime sizes its thread pool from the host core count, so in a CPU-limited
container it oversubscribes and slows down. Pass a `SessionOptions` with
`intra_op_num_threads` set to the CPU quota:

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2  # match the container CPU limit
opts.inter_op_num_threads = 1
g2p = G2P("model.onnx", session_options=opts)
```

On a 2-CPU pod this cut median latency ~30% versus the default thread pool.

## Citation

```bibtex
@misc{melichov2026renikud,
  title={ReNikud: Audio-Supervised Hebrew Grapheme-to-Phoneme Conversion},
  author={Maxim Melichov and Yakov Kolani and Morris Alper},
  year={2026},
  url={https://arxiv.org/pdf/2606.20179},
}
```
