"""Run with: uv run examples/basic.py model.onnx."""

from __future__ import annotations

import sys

from renikud_onnx import G2P


if len(sys.argv) != 2:
    raise SystemExit("Usage: uv run examples/basic.py path/to/model.onnx")

g2p = G2P(sys.argv[1])
sentence = "הוא רצה את זה גם, אבל היא רצה מהר והקדימה אותו"
print(g2p.phonemize(sentence))  # IPA
print(g2p.vocalize(sentence))  # niqqud
