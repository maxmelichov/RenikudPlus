# Changelog

## 0.6.0

- **Synced with the Hugging Face model repo**, which re-uploaded the weights and
  dropped `datastore.json`. New `model.onnx` downloads automatically on first
  use; the published benchmark moves to 93.3% word accuracy on 126k words of
  unseen podcast speech, with per-category numbers updated in the README.
- **The attested-reading rescorer is gone**, along with the datastore it read:
  the candidate inventory, the proposal energies and the trust gate. The new
  weights fold in what it was doing — on 24 hand-checked pointed words the
  default decode goes from 13/24 to 17/24 segments right without it. The
  aligner stays (it validates lexicon entries) and so does the force lexicon.
  `G2P(datastore=..., tau=..., mask=..., inventory_sources=...)` and
  `rescoring_enabled` are removed.
- `niqqud="use"` is unaffected and still constrains the exact-MAP decode: 24/24
  segments and 18/24 exact on that set, and 7/7 against the reference readings
  on the pointed possessive-suffix words.

## 0.5.0

- **Vowelized (pointed) input is supported.** Niqqud and cantillation are not in
  the model's character vocabulary: pointed input used to tokenize to `[UNK]`
  and leak the raw combining marks straight into the IPA string.

  ```python
  # before — the qamats, shin dot and holam survive into the output
  >>> g2p.phonemize("\u05e9\u05b8\u05c1\u05dc\u05d5\u05b9\u05dd")   # pointed shalom
  'ʃe\u05b8\u05c1lˈo\u05b9m'
  # after
  'ʃalˈom'                            # identical to the bare form
  ```

  `strip_niqqud()` now runs on the NFD text in both `phonemize` and `vocalize`,
  so pointed text decodes byte-identically to the same text unpointed. Maqaf,
  paseq and sof pasuq stay as punctuation.
- **Niqqud can also be read as evidence** — `G2P(niqqud="use")`, or per call
  `phonemize(text, niqqud="use")`. `parse_niqqud()` turns each vowel sign,
  dagesh qal and shin/sin dot into a hard constraint pinned into the exact-MAP
  energy, so the model predicts only what the pointing leaves open. Qamats
  (qamats qatan is the same sign) and shva (na vs nah) stay ambiguous by design;
  matres are resolved to the model's label convention. A constraint that would
  leave a letter with no legal reading is dropped, so sloppy pointing degrades
  to the default decode. Default stays `"strip"`: existing callers see no
  change. On 24 hand-checked pointed words: 24/24 vowels and consonants right
  (13/24 stripping), 19/24 including stress (13/24); on the seven pointed
  possessive-suffix words from the field report, 7/7. Niqqud does not mark
  stress, so stress remains the model's own call.
- **Exact-MAP cascade decode**, on by default. Consonant, vowel and stress are
  decoded jointly under `E(c,v,s) = log P(c) + log P(v|c) + log P(s|c,v)`, with
  per-letter legality and "stress needs a vowel" as hard constraints. This
  removes the stranded-stress-mark class (`ʃˈ`, `benatˈv`): a mark can no longer
  land on a letter with no vowel. `exact_map=False` restores the greedy argmax.
- **Attested-reading rescorer**, on whenever `datastore.json` sits beside the
  weights — which it now does, since the automatic download fetches it too.
  Trust gate at `tau=0.2` over corpus-attested readings; `datastore=None` off.
- **Force lexicon** — `G2P(..., lexicon={...})` or a TSV path, wins where it
  matches. Off unless passed.
- **Long inputs** past the encoder's 2,046-character window are split on
  sentence, then comma, then word boundaries and decoded window by window, with
  a raw-Hebrew-in-output backstop (`on_hebrew_leak`).
- **The Hebrew number front end moved to
  [`hebrew-num2words`](https://pypi.org/project/hebrew-num2words/)** and is now
  a dependency rather than ~560 bundled lines. Output is unchanged; the API is
  re-exported, so `from renikud_onnx import normalize_numbers` still works.
- `G2P()` now also accepts `session_options`, `repo_id`, `filename` and
  `cache_dir`.

## 0.4.0

- ONNX Runtime is no longer a hard dependency: `onnxruntime` and
  `onnxruntime-gpu` unpack into the same directory, so pinning the CPU build
  silently disabled CUDA for anyone who also asked for the GPU one. Install
  `renikud-plus[cpu]` or `renikud-plus[gpu]`, or bring your own build.

## 0.3.0

- `vocalize()` renders predictions as pointed Hebrew (niqqud) instead of IPA.
- ONNX weights download from Hugging Face on first `G2P()` use and are cached.
