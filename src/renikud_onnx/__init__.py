"""ReNikud Plus: Hebrew grapheme-to-phoneme inference via ONNX.

    from renikud_onnx import G2P
    g2p = G2P()                      # weights download from Hugging Face on first use
    g2p.phonemize("שלום לכולם")      # IPA
    g2p.vocalize("שלום לכולם")       # niqqud

Points in the INPUT are ignored by default and can instead be read as evidence:
`G2P(niqqud="use")` pins the vowel signs, dagesh qal and shin/sin dot into the
decode, so pointed text disambiguates what the skeleton cannot.

The Hebrew number front end comes from the `hebrew-num2words` package. Two
decode layers sit beyond the neural graph:

  1. the attested-reading rescorer. ON when `datastore.json` is found next to
     the model file, at its tuned settings (tau = 0.2, corpus mask). It ships in
     the same Hugging Face repo as the weights, so it is downloaded and enabled
     by default. Pass `datastore=None` to switch it off, or a path to point it
     elsewhere.
  2. a force-only runtime lexicon: `lexicon=` — a surface -> IPA map that always
     wins where it matches. Off unless you pass one.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:  # pragma: no cover - depends on the install
    raise ModuleNotFoundError(
        "renikud-plus needs an ONNX Runtime build, and deliberately does not pin "
        "one: `onnxruntime` and `onnxruntime-gpu` install into the same package "
        "directory, so depending on the CPU build silently overwrites the GPU one "
        "for anyone who wanted CUDA. Install whichever matches your machine:\n"
        "    pip install 'renikud-plus[cpu]'\n"
        "    pip install 'renikud-plus[gpu]'"
    ) from exc

#: rescoring datastore looked for beside the model file when `datastore="auto"`.
DEFAULT_DATASTORE_NAME = "datastore.json"

DEFAULT_HF_REPO = "notmax123/RenikudPlus"
DEFAULT_MODEL_FILENAME = "model.onnx"


def download_model(
    repo_id: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_MODEL_FILENAME,
    cache_dir: str | Path | None = None,
) -> str:
    """Download one file from the Hugging Face model repo (cached after the first call)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )




# --------------------------------------------------------------------------
# Number words
# --------------------------------------------------------------------------
# The Hebrew number front end lives in its own package, `hebrew-num2words`
# (https://pypi.org/project/hebrew-num2words/) — same rules, mined from the same
# 473-row Hebrew<->IPA gold set, now usable without pulling in onnxruntime.
# Re-exported here so `from renikud_onnx import normalize_numbers` keeps working.
from hebrew_num2words import (  # noqa: E402,F401
    FEM,
    MASC,
    HebrewNumberNormalizer,
    cardinal,
    contains_digit,
    expand_body,
    expand_date,
    expand_integer,
    expand_time,
    expand_token,
    expand_year,
    normalize_numbers,
    noun_gender,
    ordinal,
    spell_digits,
)




import sys
import warnings

#: max_position_embeddings (2048) minus CLS + SEP. Measured: 2046 OK, 2047 raises.
SAFE_CHUNK_CHARS = 2046

#: Hebrew letters alef..tav. Hebrew punctuation (geresh, gershayim, maqaf) is
#: outside this range and is legitimately passed through by the decoder.
HEBREW_LETTER_RE = re.compile(r"[א-ת]")

_SENTENCE_ENDS = ".!?;\n\r…׃"
_CLAUSE_ENDS = ",:–—"


def _windows_with_constraints(windows: list[str], constraints: dict | None):
    """Pair each decode window with the constraints falling inside it.

    `split_for_decode` returns contiguous substrings, so a running offset is
    enough to re-base the whole-text indices onto each window.
    """
    offset = 0
    for window in windows:
        if constraints:
            yield window, {i - offset: c for i, c in constraints.items()
                           if offset <= i < offset + len(window)}
        else:
            yield window, None
        offset += len(window)


class RawHebrewLeakError(ValueError):
    """Raw Hebrew letters survived into what should be an IPA string."""


def _split_after(text: str, terminators: str) -> list[str]:
    pieces: list[str] = []
    start = i = 0
    n = len(text)
    while i < n:
        if text[i] in terminators:
            j = i + 1
            while j < n and text[j] in terminators:
                j += 1
            while j < n and text[j].isspace():
                j += 1
            pieces.append(text[start:j])
            start = i = j
        else:
            i += 1
    if start < n:
        pieces.append(text[start:])
    return pieces


def _split_words(text: str) -> list[str]:
    pieces: list[str] = []
    start = i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            j = i
            while j < n and text[j].isspace():
                j += 1
            pieces.append(text[start:j])
            start = i = j
        else:
            i += 1
    if start < n:
        pieces.append(text[start:])
    return pieces


def _pack(pieces: list[str], limit: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if cur and len(cur) + len(piece) > limit:
            out.append(cur)
            cur = piece
        else:
            cur += piece
    if cur:
        out.append(cur)
    return out


def split_for_decode(text: str, limit: int = SAFE_CHUNK_CHARS,
                     warn_on_hard_cut: bool = True) -> list[str]:
    """Split into decode windows of at most `limit` characters, losslessly.

    ``"".join(split_for_decode(t)) == t``; ``len(text) <= limit`` returns
    ``[text]`` unchanged.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if len(text) <= limit:
        return [text]

    out: list[str] = []
    for sent_chunk in _pack(_split_after(text, _SENTENCE_ENDS), limit):
        if len(sent_chunk) <= limit:
            out.append(sent_chunk)
            continue
        for clause_chunk in _pack(_split_after(sent_chunk, _CLAUSE_ENDS), limit):
            if len(clause_chunk) <= limit:
                out.append(clause_chunk)
                continue
            for word_chunk in _pack(_split_words(clause_chunk), limit):
                if len(word_chunk) <= limit:
                    out.append(word_chunk)
                    continue
                if warn_on_hard_cut:
                    warnings.warn(
                        f"[long-input] a {len(word_chunk)}-character run with no "
                        f"whitespace exceeds the {limit}-character decode window; "
                        f"cutting it mid-token.",
                        RuntimeWarning, stacklevel=2,
                    )
                out.extend(word_chunk[i:i + limit]
                           for i in range(0, len(word_chunk), limit))

    assert "".join(out) == text, "split_for_decode is not lossless"
    return out


def check_no_raw_hebrew(output: str, text: str, mode: str = "warn",
                        where: str = "phonemize") -> str:
    """Fail loudly when raw Hebrew letters survive into an IPA string."""
    if mode == "ignore":
        return output
    leaked = HEBREW_LETTER_RE.findall(output)
    if not leaked:
        return output
    sample = "".join(dict.fromkeys(leaked))[:20]
    msg = (
        f"[{where}] RAW HEBREW IN OUTPUT: {len(leaked)} Hebrew letter(s) "
        f"({sample}) survived into what should be an IPA string — the tail of the "
        f"input was never transcribed. input={len(text)} chars, "
        f"output={len(output)} chars."
    )
    if mode == "raise":
        raise RawHebrewLeakError(msg)
    print(f"WARNING: {msg}", file=sys.stderr)
    return output








ALEF_ORD = ord("א")
TAF_ORD = ord("ת")
STRESS_MARK = "ˈ"
ORTHOGRAPHIC_MARKERS = ("'", '"')
NEG = -1e30


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], n), dtype=np.float32)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def _is_hebrew(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


def normalize_graphemes(text: str) -> str:
    text = re.sub(r"[׳'`´]", "'", text)
    text = re.sub(r'[״""]', '"', text)
    return text


#: Hebrew points and cantillation marks: niqqud, dagesh, shin/sin dots, te'amim.
#: The maqaf (U+05BE), paseq (U+05C0), sof pasuq (U+05C3) and nun hafukha
#: (U+05C6) are punctuation, not marks on a letter, so they are left in place --
#: the decoder passes them through and the chunker splits on the sentence ones.
_NIQQUD_RE = re.compile(
    "[\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7]"
)


def strip_niqqud(text: str) -> str:
    """Drop niqqud and cantillation, leaving the bare consonantal skeleton.

    The model is trained on unpointed Hebrew: a combining point is not in its
    character vocabulary, so pointed input both tokenizes to [UNK] and leaks the
    raw mark into the output -- pointed shalom decodes as `ʃe` + qamats + shin
    dot + `lˈo` + holam + `m`, not `ʃalˈom`. Stripping first makes
    pointed input decode exactly like the same text unpointed. Input must be NFD
    (precomposed letters such as U+FB2A decompose to letter + point there).

    Note the points are discarded, not read: they cannot disambiguate a
    homograph for the model. Pointed text in, prediction from the skeleton out.
    """
    return _NIQQUD_RE.sub("", text)


#: vowel signs -> the vowel classes they admit. Two signs stay ambiguous on
#: purpose and are left to the model: qamats, because qamats qatan (/o/, as in
#: כָּל) is written with the same sign in most pointed text, and shva, because
#: shva na is /e/ while shva nah is nothing. Every other sign is determined.
_POINT_VOWELS = {
    "\N{HEBREW POINT PATAH}": ("a",),
    "\N{HEBREW POINT HATAF PATAH}": ("a",),
    "\N{HEBREW POINT QAMATS}": ("a", "o"),
    "\N{HEBREW POINT QAMATS QATAN}": ("o",),
    "\N{HEBREW POINT HATAF QAMATS}": ("o",),
    "\N{HEBREW POINT SEGOL}": ("e",),
    "\N{HEBREW POINT TSERE}": ("e",),
    "\N{HEBREW POINT HATAF SEGOL}": ("e",),
    "\N{HEBREW POINT HIRIQ}": ("i",),
    "\N{HEBREW POINT HOLAM}": ("o",),
    "\N{HEBREW POINT HOLAM HASER FOR VAV}": ("o",),
    "\N{HEBREW POINT QUBUTS}": ("u",),
    "\N{HEBREW POINT SHEVA}": ("∅", "e"),
}

#: dagesh qal: present -> stop, absent on an otherwise pointed letter -> fricative.
_DAGESH_HARD = {"ב": "b", "כ": "k", "ך": "k", "פ": "p", "ף": "p"}
_DAGESH_SOFT = {"ב": "v", "כ": "χ", "ך": "χ", "פ": "f", "ף": "f"}


def parse_niqqud(text: str) -> tuple[str, dict[int, tuple[tuple[str, ...] | None,
                                                          tuple[str, ...] | None]]]:
    """Split pointed text into its skeleton and the readings its points force.

    Returns ``(bare_text, constraints)``, where `constraints` maps an index into
    `bare_text` to ``(consonants, vowels)`` — the label sets that index is
    allowed, or None where the pointing says nothing. Input must be NFD.

    Matres lectionis are resolved to the model's own label convention: it hangs
    the vowel on the consonant and leaves the vowel letter silent, so a holam
    male or shuruk vav (וֹ / וּ) pushes its /o/ or /u/ back onto the preceding
    letter and takes ∅/∅ itself — the inverse of the fixup `vocalize` applies.
    """
    bare: list[str] = []
    # One record per pointed letter: [bare_index, letter, marks].
    letters: list[list] = []
    for char in text:
        if _NIQQUD_RE.match(char):
            if letters and letters[-1][0] == len(bare) - 1:
                letters[-1][2].append(char)
            continue
        if _is_hebrew(char):
            letters.append([len(bare), char, []])
        bare.append(char)

    # A word counts as pointed once any of its letters carries a vowel sign; only
    # there is an UNMARKED letter informative (in pointed text a vowel vav always
    # carries holam or dagesh, so a bare vav is the consonant).
    text_bare = "".join(bare)
    word_of: dict[int, int] = {}
    word_pointed: dict[int, bool] = {}
    word = 0
    for i, ch in enumerate(text_bare):
        if ch.isspace():
            word += 1
        word_of[i] = word
    for idx, _letter, marks in letters:
        if any(m in _POINT_VOWELS for m in marks):
            word_pointed[word_of[idx]] = True

    cons: dict[int, tuple[str, ...]] = {}
    vows: dict[int, tuple[str, ...]] = {}
    mater: list[tuple[int, str]] = []  # (record index, "o" | "u")
    for rec_i, (idx, letter, marks) in enumerate(letters):
        signs = [m for m in marks if m in _POINT_VOWELS]
        dagesh = _DAGESH in marks

        if letter == "ש":
            if _SHIN_DOT in marks:
                cons[idx] = ("ʃ",)
            elif _SIN_DOT in marks:
                cons[idx] = ("s",)
        elif letter == "ה" and dagesh:
            cons[idx] = ("h",)  # mapiq: consonantal he
        elif letter in _DAGESH_HARD:
            if dagesh:
                cons[idx] = (_DAGESH_HARD[letter],)
            elif signs:
                # Pointed, yet no dagesh: in pointed text that is the fricative.
                cons[idx] = (_DAGESH_SOFT[letter],)

        if letter == "ו" and not signs and dagesh:
            mater.append((rec_i, "u"))  # shuruk
            continue
        if letter == "ו" and len(signs) == 1 and _POINT_VOWELS[signs[0]] == ("o",):
            mater.append((rec_i, "o"))  # holam male
            continue
        if signs:
            allowed: set[str] = set(_POINT_VOWELS[signs[0]])
            for sign in signs[1:]:
                allowed &= set(_POINT_VOWELS[sign])
            if allowed:
                vows[idx] = tuple(sorted(allowed))

    # Unmarked letters inside a pointed word.
    for rec_i, (idx, letter, marks) in enumerate(letters):
        if marks or not word_pointed.get(word_of[idx]):
            continue
        prev = letters[rec_i - 1] if rec_i else None
        nxt = letters[rec_i + 1] if rec_i + 1 < len(letters) else None
        same_word = nxt is not None and word_of[nxt[0]] == word_of[idx]
        if letter == "ו":
            # A vowel vav carries a point; this one does not, so it is a consonant.
            cons[idx] = ("v", "w")
        elif letter == "י" and prev is not None and prev[0] == idx - 1 \
                and vows.get(prev[0]) in (("i",), ("e",)):
            cons[idx] = ("∅",)  # hiriq / tsere male
            vows[idx] = ("∅",)
        elif letter in ("א", "ה") and not same_word:
            cons[idx] = ("∅",)  # silent word-final alef / he (no mapiq)

    for rec_i, quality in mater:
        idx = letters[rec_i][0]
        prev = letters[rec_i - 1] if rec_i else None
        # The vowel lands on the letter before, when that letter is adjacent and
        # has no vowel of its own; otherwise the vav keeps it.
        if prev is not None and prev[0] == idx - 1 and prev[0] not in vows:
            vows[prev[0]] = (quality,)
            vows[idx] = ("∅",)
        else:
            vows[idx] = (quality,)
        cons[idx] = ("∅",)

    constraints = {
        idx: (cons.get(idx), vows.get(idx))
        for idx in set(cons) | set(vows)
    }
    return "".join(bare), constraints


# --------------------------------------------------------------------------
# Niqqud rendering (`G2P.vocalize`)
# --------------------------------------------------------------------------

# Niqqud points named by their Unicode names -- the bare combining glyphs are
# invisible in source. `vocalize` renders each of the model's five predicted vowel
# qualities (a/e/i/o/u) as one representative sign; signs that share a sound
# (patah/qamats, tsere/segol) collapse to one, which is lossless for pronunciation.
_NIQQUD_VOWEL = {
    "a": "\N{HEBREW POINT PATAH}",
    "e": "\N{HEBREW POINT SEGOL}",
    "i": "\N{HEBREW POINT HIRIQ}",
    "o": "\N{HEBREW POINT HOLAM}",
    "u": "\N{HEBREW POINT QUBUTS}",
}
_DAGESH = "\N{HEBREW POINT DAGESH OR MAPIQ}"  # hard b/k/p: בּ כּ פּ
_SHIN_DOT = "\N{HEBREW POINT SHIN DOT}"  # שׁ
_SIN_DOT = "\N{HEBREW POINT SIN DOT}"  # שׂ

# A letter takes a consonant-conditioned point when the model's predicted consonant
# selects one: בכפ take a dagesh in their hard (stop) realization, ש takes the
# shin- or sin-dot. One table instead of branches scattered through `vocalize`.
_DAGESH_PAIRS = frozenset({("ב", "b"), ("כ", "k"), ("ך", "k"), ("פ", "p"), ("ף", "p")})


def _consonant_point(letter: str, consonant: str) -> str:
    """Dagesh or shin/sin dot implied by the consonant the model chose for `letter`."""
    if letter == "ש":
        return _SHIN_DOT if consonant == "ʃ" else _SIN_DOT
    if (letter, consonant) in _DAGESH_PAIRS:
        return _DAGESH
    return ""


# ==========================================================================
# ATTESTED-READING RESCORING, AND THE FORCE-ONLY LEXICON
# --------------------------------------------------------------------------
# The rescorer is ON when a datastore is found beside the model file and OFF
# with `datastore=None`; the force lexicon is off unless `lexicon=` is passed.
# The candidate inventory, the proposal energies, the trust gate and the aligner
# all live in this file, with their constants and tie-breaks intact.
# ==========================================================================

# --- source bits of the candidate inventory --------------------------------
SRC_LEX, SRC_HOMO, SRC_CORPUS, SRC_OLD = 1, 2, 4, 8

#: every source except SRC_OLD.
MASK_ALL = SRC_LEX | SRC_HOMO | SRC_CORPUS
#: the default: corpus attestation only.
MASK_CORPUS = SRC_CORPUS

#: punctuation stripped off an IPA token before comparing it to a proposal.
PUNCT_STRIP = ".,?!:;\"'()-—…«»"

_IPA_ASCII_FIXUPS = {"g": "ɡ", "x": "χ", "r": "ʁ"}


def normalize_ipa(text: str) -> str:
    """ASCII stand-ins to canonical IPA."""
    for old, new in _IPA_ASCII_FIXUPS.items():
        text = text.replace(old, new)
    return text


# --- label vocabularies ----------------------------------------------------
VOWEL_SYMBOLS = ("∅", "a", "e", "i", "o", "u")
VOWEL_TO_ID = {v: i for i, v in enumerate(VOWEL_SYMBOLS)}
NUM_VOWEL_CLASSES = len(VOWEL_SYMBOLS)

CONSONANT_SYMBOLS = (
    "∅",
    "b", "v", "d", "h", "z", "χ", "t", "j", "k", "l", "m", "n", "s", "f", "p",
    "ts", "tʃ", "w", "ʔ", "ɡ", "ʁ", "ʃ", "ʒ", "dʒ",
)
CONSONANT_TO_ID = {c: i for i, c in enumerate(CONSONANT_SYMBOLS)}

_VOWELS_SET = frozenset("aeiou")


def chunk_to_labels(chunk: str) -> tuple[str, str, int]:
    """Aligned IPA chunk -> (consonant, vowel, stress)."""
    if not chunk or chunk == " ":
        return ("∅", "∅", 0)
    pos = 0
    stress = 0
    if STRESS_MARK in chunk:
        stress = 1
        chunk = chunk.replace(STRESS_MARK, "")
    consonant = "∅"
    for multi in ("tʃ", "dʒ", "ts"):
        if chunk.startswith(multi):
            consonant = multi
            pos = len(multi)
            break
    else:
        if pos < len(chunk) and chunk[pos] not in _VOWELS_SET:
            consonant = chunk[pos]
            pos += 1
    vowel = chunk[pos:] if pos < len(chunk) else "∅"
    if not vowel:
        vowel = "∅"
    if vowel.endswith("χ"):
        consonant = "χ"
        vowel = vowel[:-1] or "∅"
    return (
        consonant if consonant in CONSONANT_TO_ID else "∅",
        vowel if vowel in VOWEL_TO_ID else "∅",
        stress,
    )


# --- the aligner -----------------------------------------------------------
ALIGN_LETTER_CONSONANTS: dict[str, tuple[str, ...]] = {
    "א": ("ʔ", ""), "ב": ("b", "v"), "ג": ("ɡ", "dʒ"), "ד": ("d",),
    "ה": ("h", ""), "ו": ("v", "w", ""), "ז": ("z", "ʒ"), "ח": ("χ",),
    "ט": ("t",), "י": ("j", ""), "כ": ("k", "χ"), "ך": ("k", "χ"),
    "ל": ("l", ""), "מ": ("m",), "ם": ("m",), "נ": ("n",), "ן": ("n",),
    "ס": ("s",), "ע": ("ʔ", ""), "פ": ("p", "f"), "ף": ("p", "f"),
    "צ": ("ts", "tʃ"), "ץ": ("ts", "tʃ"), "ק": ("k",), "ר": ("ʁ",),
    "ש": ("ʃ", "s", ""), "ת": ("t",),
}
ALIGN_VOWELS = ("a", "e", "i", "o", "u")
ALIGN_SPECIAL_CHARS = ("'", '"')
FURTIVE_PATAH_LETTER = "ח"
FINAL_HE_LETTER = "ה"
_GERESH = "'"


def _ctx_free(w: str, i: int) -> bool:
    return True


def _ctx_geresh_after(w: str, i: int) -> bool:
    return i + 1 < len(w) and w[i + 1] == _GERESH


def _ctx_dalet_gimel_geresh(w: str, i: int) -> bool:
    return i + 2 < len(w) and w[i + 1] == "ג" and w[i + 2] == _GERESH


def _ctx_final_vav_after_alef(w: str, i: int) -> bool:
    if i == 0 or w[i - 1] != "א":
        return False
    return all(c in ALIGN_SPECIAL_CHARS for c in w[i + 1:])


#: optional extra aligner cells (letter, extra consonant, predicate, name).
#: OFF by default; `align_cells=()` is the standard aligner.
EXTRA_CELLS: tuple[tuple[str, str, object, str], ...] = (
    ("ס", "z", _ctx_free, "samz"),
    ("ת", "s", _ctx_geresh_after, "tafs"),
    ("ד", "", _ctx_dalet_gimel_geresh, "daled0"),
    ("ו", "j", _ctx_final_vav_after_alef, "vavj"),
)
EXTRA_BY_NAME = {c[3]: c for c in EXTRA_CELLS}


def resolve_align_cells(spec) -> tuple:
    """`()`/None -> standard; "extended" -> all four cells; or an iterable of names."""
    if not spec:
        return ()
    if isinstance(spec, str):
        spec = EXTRA_BY_NAME.keys() if spec.lower() in ("extended", "w4") else [
            s for s in re.split(r"[,\s]+", spec) if s
        ]
    out = []
    for name in spec:
        if isinstance(name, tuple):
            out.append(name)
        elif name in EXTRA_BY_NAME:
            out.append(EXTRA_BY_NAME[name])
        else:
            raise ValueError(f"unknown aligner cell {name!r}; known: {sorted(EXTRA_BY_NAME)}")
    return tuple(out)


def align_word(heb_word: str, ipa_word: str, cells: tuple = ()) -> list[tuple[str, str]] | None:
    """DP alignment of a Hebrew surface to an IPA string, or None if illegal.

    `cells=()` is the standard aligner; a non-empty `cells` widens it.
    """
    if not ipa_word or ipa_word.count(STRESS_MARK) > 1:
        return None

    extra: dict[str, list] = {}
    for letter, cons, pred, *_ in cells:
        extra.setdefault(letter, []).append((cons, pred))

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def search(h_idx: int, i_idx: int):
        if h_idx == len(heb_word) and i_idx == len(ipa_word):
            return ()
        if h_idx == len(heb_word):
            return None
        char = heb_word[h_idx]
        if char in ALIGN_SPECIAL_CHARS:
            res = search(h_idx + 1, i_idx)
            if res is not None:
                return ((char, ""),) + res
            return None
        if char not in ALIGN_LETTER_CONSONANTS:
            return None
        allowed_cons = list(ALIGN_LETTER_CONSONANTS[char])
        for cons, pred in extra.get(char, ()):
            if cons not in allowed_cons and pred(heb_word, h_idx):
                allowed_cons.append(cons)
        rest_ipa = ipa_word[i_idx:]
        is_final_he = (
            char == FINAL_HE_LETTER
            and h_idx > 0
            and all(c in ALIGN_SPECIAL_CHARS for c in heb_word[h_idx + 1:])
        )
        for cons in allowed_cons:
            if cons and not rest_ipa.startswith(cons):
                continue
            c_len = len(cons)
            for has_stress in (True, False):
                if has_stress and is_final_he:
                    continue
                s_len = 0
                if has_stress:
                    if (c_len < len(rest_ipa)) and rest_ipa[c_len] == STRESS_MARK:
                        s_len = 1
                    else:
                        continue
                for v in (*ALIGN_VOWELS, ""):
                    if has_stress and not v:
                        continue
                    if v and is_final_he and not cons:
                        continue
                    v_start = c_len + s_len
                    if v and rest_ipa[v_start:].startswith(v):
                        v_len = len(v)
                    elif not v:
                        v_len = 0
                    else:
                        continue
                    total_step = v_start + v_len
                    res = search(h_idx + 1, i_idx + total_step)
                    if res is not None:
                        return ((char, rest_ipa[:total_step]),) + res
        if char == FURTIVE_PATAH_LETTER:
            for cons in allowed_cons:
                if not cons:
                    continue
                for has_stress in (True, False):
                    prefix = STRESS_MARK + "a" if has_stress else "a"
                    if rest_ipa.startswith(prefix + cons):
                        step = len(prefix) + len(cons)
                        res = search(h_idx + 1, i_idx + step)
                        if res is not None:
                            return ((char, rest_ipa[:step]),) + res
        return None

    result = search(0, 0)
    return list(result) if result is not None else None


def candidate_labels(heb_word: str, ipa_word: str,
                     cells: tuple = ()) -> list[tuple[int, int, int]] | None:
    """eval_candidate_rescore.candidate_labels — per-letter (cid, vid, stress)."""
    aligned = align_word(heb_word, ipa_word, cells)
    if aligned is None:
        return None
    out = []
    for _, chunk in aligned:
        c, v, s = chunk_to_labels(chunk)
        cid, vid = CONSONANT_TO_ID.get(c), VOWEL_TO_ID.get(v)
        if cid is None or vid is None:
            return None
        out.append((cid, vid, s))
    return out


# --- the candidate inventory -----------------------------------------------
_VE_RE = re.compile(r"^(ve|va|vi|v)")
_LEGAL_IPA_RE = re.compile(r"^[abdefghijklmnoprstuvwzɡʁʃʒʔˈχ]+$")


def _inv_clean(ipa: str) -> str | None:
    ipa = normalize_ipa(ipa.strip())
    return ipa if ipa and _LEGAL_IPA_RE.match(ipa) else None


def _inv_heb_ok(w: str) -> bool:
    """A separator-bearing surface (במד״א, ג'אז) can never be a candidate at any
    tau. The force lexicon exists for exactly this class."""
    return bool(w) and all(_is_hebrew(c) for c in w)


def _inv_veva_surface(readings: list[str]) -> bool:
    seen: dict[str, str] = {}
    for r in readings:
        m = _VE_RE.match(r)
        if not m:
            continue
        stem = r[m.end():]
        if stem in seen and seen[stem] != r:
            return True
        seen[stem] = r
    return False


def build_inventory(datastore: str,
                    lexnikud: str | None = None,
                    homograph_inventory: str | None = None,
                    old_lexicon: str | None = None,
                    corpus_min_total: int = 5,
                    corpus_min_count: int = 3,
                    corpus_min_share: float = 0.05,
                    homo_min_share: float = 0.02) -> tuple[dict, dict]:
    """Build the candidate inventory. Every source path is supplied by the caller.

    Only `datastore` is required, and it is the only source shipped beside the
    model; the other three are optional extras.
    """
    surf: dict[str, dict[str, int]] = {}

    def add(w: str, ipa: str, bit: int) -> None:
        if not _inv_heb_ok(w):
            return
        c = _inv_clean(ipa)
        if c is None:
            return
        d = surf.setdefault(w, {})
        d[c] = d.get(c, 0) | bit

    stats: dict = {}

    n = 0
    if lexnikud:
        with open(lexnikud, encoding="utf-8") as fh:
            fh.readline()
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) == 3:
                    add(p[1], p[2], SRC_LEX)
                    n += 1
    stats["lexnikud_rows"] = n

    n = 0
    if homograph_inventory:
        with open(homograph_inventory, encoding="utf-8") as fh:
            inv = json.load(fh)
        for w, d in inv.items():
            tot = max(d.get("total", 0), 1)
            for ipa, c in d.get("readings", {}).items():
                if c / tot >= homo_min_share:
                    add(w, ipa, SRC_HOMO)
                    n += 1
    stats["homograph_inventory_readings"] = n

    with open(datastore, encoding="utf-8") as fh:
        ds = json.load(fh)
    n = 0
    for w, d in ds.items():
        tot = d.get("total", 0)
        if tot < corpus_min_total:
            continue
        for ipa, c in d.get("readings", {}).items():
            if c >= corpus_min_count and c / tot >= corpus_min_share:
                add(w, ipa, SRC_CORPUS)
                n += 1
    stats["corpus_datastore_readings"] = n

    n = 0
    if old_lexicon:
        with open(old_lexicon, encoding="utf-8") as fh:
            head = fh.readline()
            if "word" not in head:
                fh.seek(0)
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) < 2 or not p[0] or not p[1]:
                    continue
                for ipa in p[1].split(";"):
                    add(p[0].strip(), ipa, SRC_OLD)
                    n += 1
    stats["old_lexicon_readings"] = n

    dropped = [w for w, r in surf.items() if _inv_veva_surface(list(r))]
    for w in dropped:
        del surf[w]
    stats["veva_surfaces_dropped"] = len(dropped)
    stats["total_surfaces"] = len(surf)
    stats["total_readings"] = sum(len(r) for r in surf.values())
    stats["multi_reading_surfaces"] = sum(1 for r in surf.values() if len(r) > 1)
    return surf, stats


# --- the trust gate --------------------------------------------------------
def apply_gate(rec: dict, tau: float, mask: int) -> tuple[str, list[dict]]:
    """Take an attested reading wherever it costs less than `tau` nats/letter.

    `rec` is `{"base": <ipa string>, "props": [...]}`.
    """
    words = rec["base"].split()
    touched = []
    for p in rec["props"]:
        best = None
        for ipa, delta, src in p["cands"]:
            if not (src & mask):
                continue
            if delta < -tau:
                continue
            if best is None or delta > best[1]:
                best = (ipa, delta)
        if best is None or best[0] == p["base_ipa"]:
            continue
        wi = p["wi"]
        if wi >= len(words):
            continue
        tok = words[wi]
        lead = len(tok) - len(tok.lstrip(PUNCT_STRIP))
        trail = len(tok) - len(tok.rstrip(PUNCT_STRIP))
        core = tok[lead: len(tok) - trail] if trail else tok[lead:]
        if core != p["base_ipa"]:
            continue
        words[wi] = tok[:lead] + best[0] + (tok[len(tok) - trail:] if trail else "")
        touched.append({"wi": wi, "surface": p["surface"], "from": p["base_ipa"],
                        "to": best[0], "delta": best[1]})
    return " ".join(words), touched


# ==========================================================================
# THE FORCE-ONLY RUNTIME LEXICON
# ==========================================================================

class LexiconEntryError(ValueError):
    """One or more force-lexicon entries failed alignment validation."""


#: characters stripped from the ENDS of a Hebrew token when looking the token up.
#: Deliberately excludes `'` `"` `-`: those are part of the surface, and are
#: exactly the keys the candidate store cannot hold.
LEX_EDGE_STRIP = ".,?!:;()[]{}…«»־–—"


class ForceLexicon:
    """A user-supplied `surface -> IPA` map that ALWAYS wins where it matches.

    It is NOT a candidate: it never enters `apply_gate`, is never scored against
    the model, and is applied after the rescoring pass. It is keyed on the RAW
    surface **including separators** (gershayim, geresh, hyphen), normalized with
    this module's `normalize_graphemes` — precisely the key space the candidate
    store cannot hold.

    Every entry is validated with `align_word` at load time. Entries that fail
    are collected and reported; by default the constructor raises with the full
    list. `on_invalid="report"` keeps them out of the map and exposes them on
    `.rejected` instead — nothing is ever silently dropped.
    """

    def __init__(self, source, align_cells: tuple = (), on_invalid: str = "raise") -> None:
        if on_invalid not in ("raise", "report"):
            raise ValueError("on_invalid must be 'raise' or 'report'")
        pairs = self._read(source)
        self.entries: dict[str, str] = {}
        self.rejected: list[tuple[str, str, str]] = []   # (surface, ipa, reason)
        for raw_surface, raw_ipa in pairs:
            surface = normalize_graphemes(unicodedata.normalize("NFD", raw_surface)).strip()
            ipa = normalize_ipa(raw_ipa.strip())
            if not surface or not ipa:
                self.rejected.append((raw_surface, raw_ipa, "empty surface or IPA"))
                continue
            reason = self._validate(surface, ipa, align_cells)
            if reason is not None:
                self.rejected.append((raw_surface, raw_ipa, reason))
                continue
            self.entries[surface] = ipa
        if self.rejected and on_invalid == "raise":
            lines = "\n".join(f"  {s!r} -> {i!r}: {why}" for s, i, why in self.rejected)
            raise LexiconEntryError(
                f"{len(self.rejected)} force-lexicon entr"
                f"{'y' if len(self.rejected) == 1 else 'ies'} failed validation "
                f"(pass on_invalid='report' to collect instead of raising):\n{lines}"
            )

    # -- loading ------------------------------------------------------------
    @staticmethod
    def _read(source) -> list[tuple[str, str]]:
        if isinstance(source, dict):
            return list(source.items())
        pairs = []
        with open(source, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.rstrip("\n").rstrip("\r")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    raise LexiconEntryError(
                        f"{source}:{lineno}: expected 'surface<TAB>IPA', got {line!r}")
                pairs.append((parts[0], parts[1]))
        return pairs

    # -- validation ---------------------------------------------------------
    @staticmethod
    def _validate(surface: str, ipa: str, align_cells: tuple) -> str | None:
        """None when the entry aligns; otherwise the reason it does not."""
        if align_word(surface, ipa, align_cells) is not None:
            return None
        if "-" in surface:
            # A hyphen is a real key character but the aligner has no
            # rule for it: validate the compound segment by segment, which
            # requires the IPA to carry the same number of separators.
            hs = surface.split("-")
            ps = re.split(r"[-\s]", ipa)
            if len(hs) != len(ps):
                return (f"hyphenated surface has {len(hs)} segments but the IPA has "
                        f"{len(ps)} — split the IPA on '-' or a space to validate it")
            for h, p in zip(hs, ps):
                if align_word(h, p, align_cells) is None:
                    return f"segment {h!r} does not align to {p!r}"
            return None
        if any(ch not in ALIGN_LETTER_CONSONANTS and ch not in ALIGN_SPECIAL_CHARS
               for ch in surface):
            bad = "".join(sorted({ch for ch in surface
                                  if ch not in ALIGN_LETTER_CONSONANTS
                                  and ch not in ALIGN_SPECIAL_CHARS}))
            return f"surface contains character(s) the aligner has no rule for: {bad!r}"
        return "IPA does not align to the surface under align_word"

    # -- lookup -------------------------------------------------------------
    def lookup(self, token: str) -> str | None:
        """Match the raw whitespace-delimited token, then the token with edge
        punctuation stripped. Separators inside the surface are preserved."""
        if token in self.entries:
            return self.entries[token]
        stripped = token.strip(LEX_EDGE_STRIP)
        if stripped and stripped in self.entries:
            return self.entries[stripped]
        return None

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, token: str) -> bool:
        return self.lookup(token) is not None


def apply_force_lexicon(text: str, ipa: str, lex: "ForceLexicon") -> tuple[str, list[dict]]:
    """Overwrite every output word whose Hebrew token is in `lex`.

    Word alignment is positional (`re.finditer(r"\\S+", ...)` on the Hebrew vs
    `.split()` on the IPA), the same correspondence `apply_gate` uses; if the two
    counts disagree the override is skipped for that string rather than guessed.
    Leading/trailing IPA punctuation on the replaced token is preserved.
    """
    heb_tokens = [m.group() for m in re.finditer(r"\S+", text)]
    words = ipa.split()
    forced: list[dict] = []
    if len(heb_tokens) != len(words):
        return ipa, forced
    for wi, tok_heb in enumerate(heb_tokens):
        repl = lex.lookup(tok_heb)
        if repl is None:
            continue
        tok = words[wi]
        lead = len(tok) - len(tok.lstrip(PUNCT_STRIP))
        trail = len(tok) - len(tok.rstrip(PUNCT_STRIP))
        core = tok[lead: len(tok) - trail] if trail else tok[lead:]
        if core == repl:
            continue
        words[wi] = tok[:lead] + repl + (tok[len(tok) - trail:] if trail else "")
        forced.append({"wi": wi, "surface": tok_heb, "from": core, "to": repl})
    return " ".join(words), forced



class G2P:
    def __init__(self, model_path: str | Path | None = None, exact_map: bool = True,
                 chunk_chars: int | None = SAFE_CHUNK_CHARS,
                 datastore: str | None = "auto",
                 tau: float = 0.2,
                 mask: str | int = "corpus",
                 inventory_sources: dict | None = None,
                 align_cells=(),
                 lexicon=None,
                 on_invalid_lexicon: str = "raise",
                 niqqud: str = "strip",
                 session_options: ort.SessionOptions | None = None,
                 repo_id: str = DEFAULT_HF_REPO,
                 filename: str = DEFAULT_MODEL_FILENAME,
                 cache_dir: str | Path | None = None) -> None:
        # `model_path=None` downloads the weights from Hugging Face
        # (`notmax123/RenikudPlus`) on first use and caches them; the rescoring
        # datastore ships in the same repo and is fetched beside them, so the
        # default constructor gets the full decode stack with no manual steps.
        #
        # `niqqud` says what to do with points in the INPUT. "strip" (the
        # default) drops them and predicts from the skeleton, as before.
        # "use" reads them: the vowel signs, dagesh qal and shin/sin dot become
        # hard constraints on the decode, so pointed input can disambiguate a
        # homograph the skeleton cannot. Either mode can be overridden per call.
        #
        # `session_options` is passed straight to onnxruntime. In a CPU-limited
        # container, set `intra_op_num_threads` to the CPU quota -- onnxruntime
        # otherwise sizes its thread pool from the host core count and
        # oversubscribes, which measurably slows inference on a small pod.
        # `chunk_chars` is the long-input decode window. The default 2,046 is the
        # encoder's hard wall (2,048 positions - CLS/SEP); None disables windowing
        # and lets onnxruntime raise past the wall.
        #
        # THE RESCORING LAYER. `datastore="auto"` (the default) looks for
        # `datastore.json` beside the model file and switches the layer on
        # if it is there; `datastore=None` switches it off; a path points it
        # somewhere else. `lexicon=None` -> the force map is never built.
        #   datastore          : corpus datastore JSON (the SRC_CORPUS bit),
        #                        "auto", or None
        #   tau                : trust gate, nats/letter (0.2 is tuned for these
        #                        weights; re-tune it for any other model)
        #   mask               : "corpus" (the default), "all", or an int bitmask
        #   inventory_sources  : optional extra inventory sources
        #   align_cells        : () standard (default), or "extended"
        #   lexicon            : dict or TSV path, force-only override
        self._chunk_chars = chunk_chars
        if niqqud not in ("strip", "use"):
            raise ValueError('niqqud must be "strip" or "use"')
        self.niqqud = niqqud
        downloaded = model_path is None
        if downloaded:
            model_path = download_model(repo_id=repo_id, filename=filename,
                                        cache_dir=cache_dir)
        self.model_path = str(model_path)
        if datastore == "auto":
            _side = os.path.join(os.path.dirname(os.path.abspath(self.model_path)),
                                 DEFAULT_DATASTORE_NAME)
            if not os.path.exists(_side) and downloaded:
                # Same repo, same snapshot directory as the weights we just
                # fetched -- so the side lookup finds it on the next line.
                _side = download_model(repo_id=repo_id,
                                       filename=DEFAULT_DATASTORE_NAME,
                                       cache_dir=cache_dir)
            datastore = _side if os.path.exists(_side) else None
        self.datastore_path = datastore
        self._session = ort.InferenceSession(self.model_path, session_options)
        meta = self._session.get_modelmeta().custom_metadata_map
        self._vocab: dict[str, int] = json.loads(meta["vocab"])
        self._consonant_vocab: dict[int, str] = {int(k): v for k, v in json.loads(meta["consonant_vocab"]).items()}
        self._vowel_vocab: dict[int, str] = {int(k): v for k, v in json.loads(meta["vowel_vocab"]).items()}
        self._cls_id = int(meta["cls_token_id"])
        self._sep_id = int(meta["sep_token_id"])
        self._letter_constraints: dict[str, list[int]] = {
            k: v for k, v in json.loads(meta["letter_consonant_constraints"]).items()
        }
        self._geresh_map: dict[str, str] = json.loads(meta.get("geresh_map", "{}"))
        # Label -> id, for turning the niqqud constraints into energy masks.
        self._consonant_ids = {v: k for k, v in self._consonant_vocab.items()}
        self._vowel_ids = {v: k for k, v in self._vowel_vocab.items()}
        # Models exported with --gender-inputs expose FiLM speaker conditioning.
        # Older 2-input models don't; phonemize() then ignores the arguments.
        input_names = {i.name for i in self._session.get_inputs()}
        self.supports_gender = {"speaker", "target_speaker"} <= input_names

        # Exact structured-cascade MAP decode. Needs the conditioning column
        # blocks of the cascade heads, which models exported before this feature
        # don't carry; those fall back to greedy argmax + margin stress.
        cond_keys = ("vowel_cond_consonant", "stress_cond_consonant", "stress_cond_vowel")
        self.supports_exact_map = all(k in meta for k in cond_keys)
        self.exact_map_default = exact_map and self.supports_exact_map
        if self.supports_exact_map:
            # [C, V], [C, S], [V, S]
            self._wv_c = np.asarray(json.loads(meta["vowel_cond_consonant"]), dtype=np.float32)
            self._ws_c = np.asarray(json.loads(meta["stress_cond_consonant"]), dtype=np.float32)
            self._ws_v = np.asarray(json.loads(meta["stress_cond_vowel"]), dtype=np.float32)
            self._cond_softmax = meta.get("cascade_cond", "softmax") == "softmax"
            n_cons = self._wv_c.shape[0]
            # [letter, consonant] True = this consonant class is illegal here.
            self._forbidden = np.ones((TAF_ORD - ALEF_ORD + 1, n_cons), dtype=bool)
            for letter, allowed in self._letter_constraints.items():
                self._forbidden[ord(letter) - ALEF_ORD, list(allowed)] = False

        # ---- the rescoring layer --------------------------------------------
        self._align_cells = resolve_align_cells(align_cells)
        self._tau = float(tau)
        if isinstance(mask, str):
            key = mask.strip().lower()
            if key not in ("corpus", "all"):
                raise ValueError("mask must be 'corpus', 'all' or an int bitmask")
            self._mask = MASK_CORPUS if key == "corpus" else MASK_ALL
        else:
            self._mask = int(mask)
        self._surfaces = None
        self.inventory_stats: dict | None = None
        self._cand_cache: dict = {}
        if datastore is not None:
            if not self.supports_exact_map:
                raise ValueError(
                    "the tau rescorer scores candidates against the exact-MAP cascade "
                    "energy; this model was exported without the cascade conditioning "
                    "metadata, so the rescorer cannot be enabled on it")
            srcs = dict(inventory_sources or {})
            unknown = set(srcs) - {"lexnikud", "homograph_inventory", "old_lexicon"}
            if unknown:
                raise ValueError(f"unknown inventory_sources key(s): {sorted(unknown)}")
            self._surfaces, self.inventory_stats = build_inventory(datastore, **srcs)
        self.last_touched: list = []
        self.last_forced: list = []
        self._lexicon = None
        if lexicon is not None:
            self._lexicon = ForceLexicon(lexicon, align_cells=self._align_cells,
                                         on_invalid=on_invalid_lexicon)

    @property
    def rescoring_enabled(self) -> bool:
        return self._surfaces is not None

    def _cand_labels(self, surface: str, ipa: str):
        key = (surface, ipa)
        if key not in self._cand_cache:
            try:
                self._cand_cache[key] = candidate_labels(surface, ipa, self._align_cells)
            except Exception:
                self._cand_cache[key] = None
        return self._cand_cache[key]

    def _propose(self, text: str, base: str, E: np.ndarray, best_u: np.ndarray,
                 best_s: np.ndarray, char_tok: dict, stressed: set) -> list[dict]:
        """Candidate loop over the energies the exact-MAP decode already computed."""
        props: list[dict] = []
        base_words = base.split()
        spans = list(re.finditer(r"\S+", text))
        if len(base_words) != len(spans):
            return props
        for wi, mt in enumerate(spans):
            word = mt.group()
            if "-" in word:
                continue  # hyphen compounds are two prosodic words; out of scope
            core = "".join(ch for ch in word if _is_hebrew(ch))
            cands_map = self._surfaces.get(core)
            if not core or not cands_map:
                continue
            idxs = [i for i in range(mt.start(), mt.end()) if _is_hebrew(text[i])]
            if len(idxs) != len(core) or idxs != list(range(idxs[0], idxs[0] + len(core))):
                continue
            if any(i not in char_tok for i in idxs):
                continue
            toks = [char_tok[i] for i in idxs]
            free = sum(float(best_s[t] if t in stressed else best_u[t]) for t in toks)
            base_ipa = base_words[wi].strip(PUNCT_STRIP)
            out_c = []
            for ipa, src in cands_map.items():
                labels = self._cand_labels(core, ipa)
                if labels is None or len(labels) != len(toks):
                    continue
                sc = 0.0
                ok = True
                for t, (c, v, s) in zip(toks, labels):
                    e = float(E[t, c, v, s])
                    if e < NEG / 2:
                        ok = False
                        break
                    sc += e
                if not ok:
                    continue
                out_c.append([ipa, round((sc - free) / len(toks), 5), src])
            if out_c:
                out_c.sort(key=lambda x: -x[1])
                props.append({"wi": wi, "surface": core, "base_ipa": base_ipa,
                              "cands": out_c[:6]})
        return props

    def _tokenize(self, text: str) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        """Tokenize character by character, return ids, mask, and offset mapping."""
        normalized = unicodedata.normalize("NFD", text)
        unk_id = self._vocab.get("[UNK]", 0)
        ids = [self._cls_id]
        offsets = [(0, 0)]  # CLS
        for i, c in enumerate(normalized):
            ids.append(self._vocab.get(c, unk_id))
            offsets.append((i, i + 1))
        ids.append(self._sep_id)
        offsets.append((0, 0))  # SEP
        mask = [1] * len(ids)
        return ids, mask, offsets

    def _best_stress_per_word(
        self,
        offsets: list[tuple[int, int]],
        text: str,
        stress_logits: np.ndarray,
        vowel_preds: np.ndarray,
    ) -> set[int]:
        word_spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        words: dict[int, list[int]] = {i: [] for i in range(len(word_spans))}
        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                continue
            for word_idx, (ws, we) in enumerate(word_spans):
                if ws <= start < we:
                    words[word_idx].append(tok_idx)
                    break
        stressed: set[int] = set()
        for toks in words.values():
            if not toks:
                continue
            # Stress must sit with a vowel (CˈV); never emit trailing ˈ.
            vowel_toks = [
                t for t in toks
                if self._vowel_vocab.get(int(vowel_preds[t]), "∅") != "∅"
            ]
            if not vowel_toks:
                continue
            # Margin (yes − no), not the raw yes-logit: raw logits differ in scale
            # across tokens.
            stressed.add(max(vowel_toks, key=lambda t: stress_logits[t, 1] - stress_logits[t, 0]))
        return stressed

    def _exact_map(
        self,
        offsets: list[tuple[int, int]],
        text: str,
        consonant_logits: np.ndarray,
        vowel_logits: np.ndarray,
        stress_logits: np.ndarray,
        return_energy: bool = False,
        constraints: dict | None = None,
    ):
        """Closed-form MAP over the joint cascade energy.

            E(c, v, s) = log P(c) + log P(v | c) + log P(s | c, v)

        The graph emits vowel/stress logits conditioned on the model's own
        upstream softmax; subtracting that conditioning back out recovers the
        unconditioned head logits, after which conditioning on ANY (c, v) is an
        exact column addition. No early argmax anywhere, so the one-stress-per-
        word constraint is free to flip both the vowel and the consonant.
        """
        cond_c = _softmax(consonant_logits) if self._cond_softmax else _onehot(
            consonant_logits.argmax(-1), consonant_logits.shape[-1]
        )
        base_v = vowel_logits - cond_c @ self._wv_c              # [S, V]
        cond_v = _softmax(vowel_logits) if self._cond_softmax else _onehot(
            vowel_logits.argmax(-1), vowel_logits.shape[-1]
        )
        base_s = stress_logits - cond_c @ self._ws_c - cond_v @ self._ws_v  # [S, 2]

        logc = _log_softmax(consonant_logits)                    # [S, C]
        logv = _log_softmax(base_v[:, None, :] + self._wv_c[None, :, :])  # [S, C, V]
        logs = _log_softmax(
            base_s[:, None, None, :]
            + self._ws_c[None, :, None, :]
            + self._ws_v[None, None, :, :]
        )                                                        # [S, C, V, 2]
        E = logc[:, :, None, None] + logv[:, :, :, None] + logs  # [S, C, V, 2]

        # Per-letter consonant legality, and "stress needs a vowel".
        for t, (start, end) in enumerate(offsets):
            if end - start == 1 and _is_hebrew(text[start:end]):
                E[t, self._forbidden[ord(text[start]) - ALEF_ORD]] = NEG

        # Niqqud read off the input, as hard constraints. A constraint that
        # contradicts the letter's own legality would leave the token with no
        # legal reading at all, so it is dropped rather than allowed to pick
        # garbage out of an all-NEG row.
        if constraints:
            char_to_tok = {
                start: t
                for t, (start, end) in enumerate(offsets)
                if end - start == 1
            }
            for idx, (cons, vows) in constraints.items():
                t = char_to_tok.get(idx)
                if t is None:
                    continue
                saved = E[t].copy()
                if cons is not None:
                    keep = [self._consonant_ids[c] for c in cons if c in self._consonant_ids]
                    block = np.ones(E.shape[1], dtype=bool)
                    block[keep] = False
                    E[t, block] = NEG
                if vows is not None:
                    keep = [self._vowel_ids[v] for v in vows if v in self._vowel_ids]
                    block = np.ones(E.shape[2], dtype=bool)
                    block[keep] = False
                    E[t, :, block] = NEG
                if E[t].max() < NEG / 2:
                    E[t] = saved

        E[:, :, 0, 1] = NEG

        S, C, V = E.shape[:3]
        flat_u = E[:, :, :, 0].reshape(S, C * V)
        flat_s = E[:, :, :, 1].reshape(S, C * V)
        arg_u, arg_s = flat_u.argmax(-1), flat_s.argmax(-1)
        best_u = flat_u[np.arange(S), arg_u]
        best_s = flat_s[np.arange(S), arg_s]
        gain = best_s - best_u

        char_tok = {
            start: t
            for t, (start, end) in enumerate(offsets)
            if end - start == 1 and _is_hebrew(text[start:end])
        }
        stressed: set[int] = set()
        for m in re.finditer(r"\S+", text):
            toks = [char_tok[i] for i in range(m.start(), m.end()) if i in char_tok]
            cands = [t for t in toks if best_s[t] > NEG / 2]
            if cands:
                stressed.add(max(cands, key=lambda t: gain[t]))

        flat = np.where(np.isin(np.arange(S), list(stressed)), arg_s, arg_u)
        if return_energy:
            # the rescorer scores candidates against exactly these energies.
            return flat // V, flat % V, stressed, E, best_u, best_s, char_tok
        return flat // V, flat % V, stressed

    def phonemize(
        self,
        text: str,
        speaker: int = 0,
        target_speaker: int = 0,
        exact_map: bool | None = None,
        on_hebrew_leak: str = "warn",
        number_norm: str = "auto",
        niqqud: str | None = None,
    ) -> str:
        """Convert unvocalized Hebrew text to IPA.

        Inputs longer than `chunk_chars` (default 2,046 — the encoder's
        2,048-position table minus CLS/SEP, measured) are split on sentence, then
        comma, then word boundaries and decoded window by window. Inputs at or
        below it are byte-for-byte unchanged from the pre-wrapper. A
        `[א-ת]`-in-output backstop then reports (default) or raises on any raw
        Hebrew that survives into the IPA string.

        `niqqud` overrides the constructor's setting for this call: "strip"
        ignores points in the input, "use" reads them as hard constraints.
        """
        if number_norm == "auto" and contains_digit(text):
            # Hebrew number front end: digits never appear
            # in training text, so a digit reaching the model is guaranteed-wrong
            # output (and mis-aligns the word after it). Runs BEFORE windowing —
            # the expander lengthens text (up to ~3.9x on digit-dense input).
            text = normalize_numbers(text)
        text = normalize_graphemes(text)
        normalized, constraints = self._read_input(unicodedata.normalize("NFD", text),
                                                   niqqud)
        windows = (split_for_decode(normalized, limit=self._chunk_chars)
                   if self._chunk_chars else [normalized])
        # The windows are contiguous substrings, so concatenating their outputs
        # preserves the original separators by construction.
        result = "".join(
            self._phonemize_window(w, speaker, target_speaker, exact_map, c)
            for w, c in _windows_with_constraints(windows, constraints)
        )
        return check_no_raw_hebrew(result, text, mode=on_hebrew_leak,
                                   where="renikud_onnx.G2P.phonemize")

    def _read_input(self, normalized: str, niqqud: str | None):
        """Skeleton + decode constraints for one already-NFD input.

        `niqqud="strip"` drops the points; `"use"` reads them. Both return the
        bare text the model is actually fed -- the points never reach the
        tokenizer either way, since they are not in its character vocabulary.
        """
        mode = self.niqqud if niqqud is None else niqqud
        if mode not in ("strip", "use"):
            raise ValueError('niqqud must be "strip" or "use"')
        if mode == "strip":
            return strip_niqqud(normalized), None
        return parse_niqqud(normalized)

    def _forward(
        self,
        normalized: str,
        speaker: int = 0,
        target_speaker: int = 0,
        exact_map: bool | None = None,
        constraints: dict | None = None,
    ):
        """One session run over an already normalized/NFD window.

        Returns the per-token labels both output renderings are built from:
        ``(offsets, consonant_preds, vowel_preds, stressed_positions,
        consonant_logits, energy)``. `energy` is the exact-MAP cascade tensors
        the rescorer needs, or None when the rescorer is off.
        """
        ids, mask, offsets = self._tokenize(normalized)

        feeds = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
        }
        if self.supports_gender:
            feeds["speaker"] = np.array([speaker], dtype=np.int64)
            feeds["target_speaker"] = np.array([target_speaker], dtype=np.int64)

        consonant_logits, vowel_logits, stress_logits = self._session.run(
            ["consonant_logits", "vowel_logits", "stress_logits"],
            feeds,
        )
        # logits shape: [1, seq_len, num_classes]
        use_exact = self.exact_map_default if exact_map is None else exact_map
        if constraints and not use_exact:
            raise ValueError(
                'niqqud="use" pins the input\'s points into the exact-MAP cascade '
                "energy; it cannot run with exact_map=False"
            )
        if use_exact and not self.supports_exact_map:
            raise ValueError(
                "this model was exported without the cascade conditioning metadata; "
                "re-export it, or pass exact_map=False for the greedy decode"
            )
        energy = None
        if use_exact:
            out = self._exact_map(
                offsets,
                normalized,
                consonant_logits[0].astype(np.float32),
                vowel_logits[0].astype(np.float32),
                stress_logits[0].astype(np.float32),
                return_energy=self._surfaces is not None,
                constraints=constraints,
            )
            if self._surfaces is not None:
                (consonant_preds, vowel_preds, stressed_positions,
                 _E, _best_u, _best_s, _char_tok) = out
                energy = (_E, _best_u, _best_s, _char_tok)
            else:
                consonant_preds, vowel_preds, stressed_positions = out
        else:
            if self._surfaces is not None:
                raise ValueError(
                    "the tau rescorer is defined on the exact-MAP cascade energies; "
                    "it cannot run with exact_map=False")

            consonant_preds = consonant_logits[0].argmax(axis=-1)
            vowel_preds = vowel_logits[0].argmax(axis=-1)
            stressed_positions = self._best_stress_per_word(
                offsets, normalized, stress_logits[0], vowel_preds
            )

        return (offsets, consonant_preds, vowel_preds, stressed_positions,
                consonant_logits[0], energy)

    def _labels_at(self, char: str, tok_idx: int, consonant_preds, vowel_preds,
                   consonant_logits) -> tuple[str, str]:
        """The (consonant, vowel) the model chose for one Hebrew letter.

        Under the exact-MAP decode the per-letter legality constraint is already
        baked into the energy; the greedy path needs the fallback applied here.
        """
        cid = int(consonant_preds[tok_idx])
        allowed = self._letter_constraints.get(char)
        if allowed is not None and cid not in allowed:
            cid = max(allowed, key=lambda x: consonant_logits[tok_idx, x])
        consonant = self._consonant_vocab.get(cid, "∅")
        vowel = self._vowel_vocab.get(int(vowel_preds[tok_idx]), "∅")
        return consonant, vowel

    def _phonemize_window(
        self,
        normalized: str,
        speaker: int = 0,
        target_speaker: int = 0,
        exact_map: bool | None = None,
        constraints: dict | None = None,
    ) -> str:
        """Decode one window to IPA."""
        (offsets, consonant_preds, vowel_preds, stressed_positions,
         consonant_logits, energy) = self._forward(
            normalized, speaker, target_speaker, exact_map, constraints)

        result = []
        prev_end = 0

        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                # CLS, SEP — skip
                if end > start:
                    prev_end = end
                continue

            # Pass through any characters skipped by the tokenizer
            if start > prev_end:
                result.append(normalized[prev_end:start])

            char = normalized[start:end]
            prev_end = end

            if not _is_hebrew(char):
                if char in ORTHOGRAPHIC_MARKERS:
                    pass
                else:
                    result.append(char)
                continue

            consonant, vowel = self._labels_at(char, tok_idx, consonant_preds,
                                               vowel_preds, consonant_logits)

            # Geresh rule: if next char is apostrophe, force geresh consonant variant
            if char in self._geresh_map and end < len(normalized) and normalized[end] == "'":
                consonant = self._geresh_map[char]

            stress = tok_idx in stressed_positions

            # Assemble IPA chunk: [consonant][ˈ][vowel]
            # Exception: word-final ח with vowel a — furtive patah flips to [ˈ]aχ
            word_final = end >= len(normalized) or not normalized[end].isalpha()
            chunk = ""
            if char == "ח" and word_final and vowel == "a":
                if stress:
                    chunk += STRESS_MARK
                chunk += "aχ"
            else:
                if consonant != "∅":
                    chunk += consonant
                if stress and vowel != "∅":
                    chunk += STRESS_MARK
                if vowel != "∅":
                    chunk += vowel
            result.append(chunk)

        if prev_end < len(normalized):
            result.append(normalized[prev_end:])

        out_str = "".join(result)

        # ---- post-passes (no-ops unless enabled at construction) ------------
        # Both operate on whitespace-delimited words, so they normalize runs of
        # inter-word whitespace to single spaces exactly as the reference
        # path does (the gate ends in `" ".join(words)`). Leading and
        # trailing whitespace of the window is preserved so that concatenating
        # windows stays lossless; the reference texts carry none, so a
        # single-window decode is bit-identical to the reference.
        if self._surfaces is not None or self._lexicon is not None:
            lead_ws = out_str[:len(out_str) - len(out_str.lstrip())]
            trail_ws = out_str[len(out_str.rstrip()):]
            body = out_str.strip()
            if self._surfaces is not None and energy is not None:
                _E, _best_u, _best_s, _char_tok = energy
                props = self._propose(normalized, body, _E, _best_u, _best_s,
                                      _char_tok, stressed_positions)
                body, self.last_touched = apply_gate(
                    {"base": body, "props": props}, self._tau, self._mask)
            if self._lexicon is not None:
                body, self.last_forced = apply_force_lexicon(
                    normalized.strip(), body, self._lexicon)
            out_str = lead_ws + body + trail_ws

        return out_str

    def vocalize(
        self,
        text: str,
        speaker: int = 0,
        target_speaker: int = 0,
        exact_map: bool | None = None,
        number_norm: str = "auto",
        niqqud: str | None = None,
    ) -> str:
        """Add niqqud (vowel diacritics) to Hebrew ``text``; non-Hebrew is unchanged.

        Renders the same per-letter (consonant, vowel) predictions as `phonemize`,
        but as niqqud -- vowel signs plus dagesh for hard b/k/p and the shin/sin
        dot -- instead of IPA. Useful for TTS engines that read pointed Hebrew
        natively but ignore phoneme markup. Diacritization is phonetically faithful
        but not publication-grade (e.g. shva in clusters is omitted), and niqqud has
        no stress mark, so the stress the model predicts is not represented here.

        The IPA-side post-passes (the attested-reading rescorer and the force
        lexicon) rewrite IPA strings, so they do not apply to this output; niqqud
        comes straight off the cascade decode.

        With `niqqud="use"` the points in the input constrain that decode, so
        this re-points pointed text rather than re-guessing it.
        """
        if number_norm == "auto" and contains_digit(text):
            text = normalize_numbers(text)
        text = normalize_graphemes(text)
        normalized, constraints = self._read_input(unicodedata.normalize("NFD", text),
                                                   niqqud)
        windows = (split_for_decode(normalized, limit=self._chunk_chars)
                   if self._chunk_chars else [normalized])
        return "".join(
            self._vocalize_window(w, speaker, target_speaker, exact_map, c)
            for w, c in _windows_with_constraints(windows, constraints)
        )

    def _vocalize_window(
        self,
        normalized: str,
        speaker: int = 0,
        target_speaker: int = 0,
        exact_map: bool | None = None,
        constraints: dict | None = None,
    ) -> str:
        """Decode one window to pointed Hebrew."""
        (offsets, consonant_preds, vowel_preds, _stressed,
         consonant_logits, _energy) = self._forward(
            normalized, speaker, target_speaker, exact_map, constraints)

        out: list[str] = []
        # One record per Hebrew letter: [char, consonant, vowel, out_index, start].
        records: list[list] = []
        prev_end = 0
        for tok_idx, (start, end) in enumerate(offsets):
            if end - start != 1:
                if end > start:
                    prev_end = end
                continue
            if start > prev_end:
                out.append(normalized[prev_end:start])
            char = normalized[start:end]
            prev_end = end
            if not _is_hebrew(char):
                # Keep everything non-Hebrew, including geresh markers (ג׳ -> ג').
                out.append(char)
                continue

            consonant, vowel = self._labels_at(char, tok_idx, consonant_preds,
                                               vowel_preds, consonant_logits)
            records.append([char, consonant, vowel, len(out), start])
            out.append("")  # placeholder, filled after the mater-lectionis fixup

        if prev_end < len(normalized):
            out.append(normalized[prev_end:])

        # Mater lectionis fixup — a vowel-letter (ו/י/ה/א) is silent but the vowel
        # it represents belongs on a neighbouring letter. Both cases are guarded to
        # adjacent letters in the same word.
        #  - Silent final ה/א (consonant ∅) took the preceding consonant's vowel;
        #    shift it back so the mark never lands on a silent letter (TTS would
        #    voice it).
        #  - A silent ו should itself carry an adjacent /o/ or /u/, so it renders as
        #    holam male (וֹ) or shuruk (וּ). Otherwise the vav is left bare and TTS
        #    may voice it as /v/ (שֻׁולחַן read as "shuvlchan"). י as a mater already
        #    renders correctly (hiriq male, ִי), so it needs no fixup.
        for i, record in enumerate(records):
            char, consonant, vowel, _idx, start = record
            if i == 0:
                continue
            previous = records[i - 1]
            adjacent = previous[4] + 1 == start
            if char in ("ה", "א") and consonant == "∅" and vowel != "∅":
                if previous[2] == "∅" and adjacent:
                    previous[2] = vowel
                    record[2] = "∅"
            elif char == "ו" and consonant == "∅" and vowel == "∅":
                if adjacent and previous[2] in ("o", "u"):
                    record[2] = previous[2]
                    previous[2] = "∅"

        for char, consonant, vowel, idx, _start in records:
            if char == "ו" and consonant == "∅" and vowel in ("o", "u"):
                # Vav as a vowel letter: shuruk (וּ) for /u/, holam male (וֹ) for /o/.
                point = _DAGESH if vowel == "u" else _NIQQUD_VOWEL["o"]
                out[idx] = unicodedata.normalize("NFC", char + point)
            else:
                point = _consonant_point(char, consonant)
                out[idx] = unicodedata.normalize("NFC", char + point + _NIQQUD_VOWEL.get(vowel, ""))

        return "".join(out)
