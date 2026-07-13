# ReNikud Plus — Hebrew Grapheme-to-Phoneme Inference

Convert unvocalized Hebrew text into IPA for TTS, speech technology, and
spoken-language research.

## Benchmark

![G2P benchmark comparison](assets/bar_plot_comparison.jpeg)

## Install

This repository contains only the code required for ONNX inference. It has no
training pipeline, checkpoint files, or PyTorch dependency.

```console
pip install .
hf download notmax123/RenikudPlus model.onnx --local-dir .
```

## Usage

```python
from renikud_onnx import G2P

g2p = G2P("model.onnx")
print(g2p.phonemize("שלום עולם"))
# → ʃlˈom ʔolˈam
```

## Citation

```bibtex
@misc{melichov2026renikud,
  title={ReNikud: Audio-Supervised Hebrew Grapheme-to-Phoneme Conversion},
  author={Maxim Melichov and Yakov Kolani and Morris Alper},
  year={2026},
  url={https://arxiv.org/pdf/2606.20179},
}
```
