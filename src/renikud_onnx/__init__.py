"""ReNikud Plus: Hebrew grapheme-to-phoneme inference via ONNX."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import onnxruntime as ort

ALEF_ORD = ord("א")
TAF_ORD = ord("ת")
STRESS_MARK = "ˈ"
ORTHOGRAPHIC_MARKERS = ("'", '"')

DEFAULT_HF_REPO = "notmax123/RenikudPlus"
DEFAULT_MODEL_FILENAME = "model.onnx"


def download_model(
    repo_id: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_MODEL_FILENAME,
    cache_dir: str | Path | None = None,
) -> str:
    """Download the ONNX weights from Hugging Face (cached after the first call)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=None if cache_dir is None else str(cache_dir),
    )

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


def _is_hebrew(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


def _consonant_point(letter: str, consonant: str) -> str:
    """Dagesh or shin/sin dot implied by the consonant the model chose for `letter`."""
    if letter == "ש":
        return _SHIN_DOT if consonant == "ʃ" else _SIN_DOT
    if (letter, consonant) in _DAGESH_PAIRS:
        return _DAGESH
    return ""


def normalize_graphemes(text: str) -> str:
    text = re.sub(r"[׳'`´]", "'", text)
    text = re.sub(r'[״""]', '"', text)
    return text


class G2P:
    def __init__(
        self,
        model_path: str | Path | None = None,
        session_options: ort.SessionOptions | None = None,
        *,
        repo_id: str = DEFAULT_HF_REPO,
        filename: str = DEFAULT_MODEL_FILENAME,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Load the ONNX model.

        If `model_path` is omitted, the default weights are downloaded from
        Hugging Face (`notmax123/RenikudPlus`) and cached locally.

        `session_options` is passed straight to onnxruntime. In a CPU-limited
        container, set `intra_op_num_threads` to the CPU quota -- onnxruntime
        otherwise sizes its thread pool from the host core count and
        oversubscribes, which measurably slows inference on a small pod.
        """
        if model_path is None:
            model_path = download_model(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        self.model_path = str(model_path)
        self._session = ort.InferenceSession(self.model_path, session_options)
        self._input_names = {input_.name for input_ in self._session.get_inputs()}
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

    def _tokenize(self, text: str) -> tuple[list[int], list[int], list[tuple[int, int]]]:
        """Tokenize character by character, return ids, mask, and offset mapping."""
        normalized = unicodedata.normalize("NFD", text)
        unk_id = self._vocab.get("[UNK]", 0)
        ids = [self._cls_id]
        offsets = [(0, 0)]
        for i, char in enumerate(normalized):
            ids.append(self._vocab.get(char, unk_id))
            offsets.append((i, i + 1))
        ids.append(self._sep_id)
        offsets.append((0, 0))
        return ids, [1] * len(ids), offsets

    def _best_stress_per_word(
        self,
        offsets: list[tuple[int, int]],
        text: str,
        stress_logits: np.ndarray,
        vowel_predictions: np.ndarray,
    ) -> set[int]:
        word_spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        words: dict[int, list[int]] = {i: [] for i in range(len(word_spans))}
        for token_index, (start, end) in enumerate(offsets):
            if end - start != 1:
                continue
            for word_index, (word_start, word_end) in enumerate(word_spans):
                if word_start <= start < word_end:
                    words[word_index].append(token_index)
                    break
        stressed: set[int] = set()
        for token_indexes in words.values():
            vowel_token_indexes = [
                token_index
                for token_index in token_indexes
                if self._vowel_vocab.get(int(vowel_predictions[token_index]), "∅") != "∅"
            ]
            if vowel_token_indexes:
                stressed.add(
                    max(vowel_token_indexes, key=lambda token_index: stress_logits[token_index, 1])
                )
        return stressed

    def _predict(
        self, text: str, speaker: int = 0, target_speaker: int = 0
    ) -> tuple[str, list[tuple[int, int]], np.ndarray, np.ndarray, set[int]]:
        """Run the model once and return the per-character predictions shared by
        `phonemize` (IPA) and `vocalize` (niqqud).

        Gender-conditioned models accept speaker IDs: 0 for unknown, 1 for male,
        and 2 for female. Legacy models only support the default unknown values.

        Returns ``(normalized_text, offsets, consonant_logits[seq, C],
        vowel_predictions[seq], stressed_token_positions)``.
        """
        if speaker not in (0, 1, 2) or target_speaker not in (0, 1, 2):
            raise ValueError("speaker and target_speaker must be 0 (unknown), 1 (male), or 2 (female)")
        supports_gender = {"speaker", "target_speaker"} <= self._input_names
        if not supports_gender and (speaker or target_speaker):
            raise ValueError(
                "This ONNX model is not gender-conditioned; export a model with speaker and target_speaker inputs."
            )

        text = normalize_graphemes(text)
        normalized = unicodedata.normalize("NFD", text)
        ids, mask, offsets = self._tokenize(text)
        inputs: dict[str, np.ndarray] = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([mask], dtype=np.int64),
        }
        if supports_gender:
            inputs["speaker"] = np.array([speaker], dtype=np.int64)
            inputs["target_speaker"] = np.array([target_speaker], dtype=np.int64)
        consonant_logits, vowel_logits, stress_logits = self._session.run(
            ["consonant_logits", "vowel_logits", "stress_logits"],
            inputs,
        )
        vowel_predictions = vowel_logits[0].argmax(axis=-1)
        stressed_positions = self._best_stress_per_word(
            offsets, normalized, stress_logits[0], vowel_predictions
        )
        return normalized, offsets, consonant_logits[0], vowel_predictions, stressed_positions

    def phonemize(self, text: str, speaker: int = 0, target_speaker: int = 0) -> str:
        """Convert text to IPA.

        Gender-conditioned models accept speaker IDs: 0 for unknown, 1 for male,
        and 2 for female. Legacy models only support the default unknown values.
        """
        normalized, offsets, consonant_logits, vowel_predictions, stressed_positions = self._predict(
            text, speaker, target_speaker
        )
        consonant_predictions = consonant_logits.argmax(axis=-1)

        result: list[str] = []
        previous_end = 0
        for token_index, (start, end) in enumerate(offsets):
            if end - start != 1:
                continue
            if start > previous_end:
                result.append(normalized[previous_end:start])

            char = normalized[start:end]
            previous_end = end
            if not _is_hebrew(char):
                if char not in ORTHOGRAPHIC_MARKERS:
                    result.append(char)
                continue

            consonant_id = int(consonant_predictions[token_index])
            allowed = self._letter_constraints.get(char)
            if allowed is not None and consonant_id not in allowed:
                consonant_id = max(allowed, key=lambda value: consonant_logits[token_index, value])
            consonant = self._consonant_vocab.get(consonant_id, "∅")
            if char in self._geresh_map and end < len(normalized) and normalized[end] == "'":
                consonant = self._geresh_map[char]

            vowel = self._vowel_vocab.get(int(vowel_predictions[token_index]), "∅")
            stress = token_index in stressed_positions
            word_final = end >= len(normalized) or not normalized[end].isalpha()
            if char == "ח" and word_final and vowel == "a":
                result.append(f"{STRESS_MARK if stress else ''}aχ")
                continue

            chunk = consonant if consonant != "∅" else ""
            if stress and vowel != "∅":
                chunk += STRESS_MARK
            if vowel != "∅":
                chunk += vowel
            result.append(chunk)

        if previous_end < len(normalized):
            result.append(normalized[previous_end:])
        return "".join(result)

    def vocalize(self, text: str, speaker: int = 0, target_speaker: int = 0) -> str:
        """Add niqqud (vowel diacritics) to Hebrew ``text``; non-Hebrew is unchanged.

        Renders the same per-letter (consonant, vowel) predictions as `phonemize`,
        but as niqqud -- vowel signs plus dagesh for hard b/k/p and the shin/sin
        dot -- instead of IPA. Useful for TTS engines that read pointed Hebrew
        natively but ignore phoneme markup. Diacritization is phonetically faithful
        but not publication-grade (e.g. shva in clusters is omitted), and niqqud has
        no stress mark, so the stress the model predicts is not represented here.

        Gender-conditioned models accept the same speaker IDs as `phonemize`.
        """
        normalized, offsets, consonant_logits, vowel_predictions, _stressed = self._predict(
            text, speaker, target_speaker
        )
        consonant_predictions = consonant_logits.argmax(axis=-1)

        out: list[str] = []
        # One record per Hebrew letter: [char, consonant, vowel, out_index, start].
        records: list[list] = []
        previous_end = 0
        for token_index, (start, end) in enumerate(offsets):
            if end - start != 1:
                continue
            if start > previous_end:
                out.append(normalized[previous_end:start])
            char = normalized[start:end]
            previous_end = end
            if not _is_hebrew(char):
                # Keep everything non-Hebrew, including geresh markers (ג׳ -> ג').
                out.append(char)
                continue

            consonant_id = int(consonant_predictions[token_index])
            allowed = self._letter_constraints.get(char)
            if allowed is not None and consonant_id not in allowed:
                consonant_id = max(allowed, key=lambda value: consonant_logits[token_index, value])
            consonant = self._consonant_vocab.get(consonant_id, "∅")
            vowel = self._vowel_vocab.get(int(vowel_predictions[token_index]), "∅")

            records.append([char, consonant, vowel, len(out), start])
            out.append("")  # placeholder, filled after the mater-lectionis fixup

        if previous_end < len(normalized):
            out.append(normalized[previous_end:])

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
