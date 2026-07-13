"""ReNikud Plus: Hebrew grapheme-to-phoneme inference via ONNX."""

from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import onnxruntime as ort

ALEF_ORD = ord("א")
TAF_ORD = ord("ת")
STRESS_MARK = "ˈ"
ORTHOGRAPHIC_MARKERS = ("'", '"')


def _is_hebrew(char: str) -> bool:
    return ALEF_ORD <= ord(char) <= TAF_ORD


def normalize_graphemes(text: str) -> str:
    text = re.sub(r"[׳'`´]", "'", text)
    text = re.sub(r'[״""]', '"', text)
    return text


class G2P:
    def __init__(self, model_path: str) -> None:
        self._session = ort.InferenceSession(model_path)
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

    def phonemize(self, text: str, speaker: int = 0, target_speaker: int = 0) -> str:
        """Convert text to IPA.

        Gender-conditioned models accept speaker IDs: 0 for unknown, 1 for male,
        and 2 for female. Legacy models only support the default unknown values.
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
        consonant_predictions = consonant_logits[0].argmax(axis=-1)
        vowel_predictions = vowel_logits[0].argmax(axis=-1)
        stressed_positions = self._best_stress_per_word(
            offsets, normalized, stress_logits[0], vowel_predictions
        )

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
                consonant_id = max(allowed, key=lambda value: consonant_logits[0][token_index, value])
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
