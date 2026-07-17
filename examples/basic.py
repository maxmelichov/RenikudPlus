"""Run with: python examples/basic.py [optional/path/to/model.onnx]."""

from __future__ import annotations

import sys

from renikud_onnx import G2P


model_path = sys.argv[1] if len(sys.argv) > 1 else None
g2p = G2P(model_path)
sentence = "הוא רצה את זה גם, אבל היא רצה מהר והקדימה אותו"
print(g2p.phonemize(sentence))  # IPA
print(g2p.vocalize(sentence))  # niqqud
