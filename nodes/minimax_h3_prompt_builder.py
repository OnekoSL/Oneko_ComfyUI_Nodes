import hashlib
import os

from .random_vocab_string_list import (
    USER_VOCAB_LABEL,
    _available_vocab_files,
    _dedupe_preserve_order,
    _load_words,
    _resolve_vocab_path,
    _shuffle_bag_sample,
)


SECTIONS = (
    ("scene", "Scene"),
    ("character", "Character"),
    ("action", "Action"),
    ("camera", "Camera"),
    ("visual_style", "Visual Style"),
    ("audio", "Audio"),
)
DEFAULT_SECTION_VOCAB_FILES = {
    "scene": "resources/minimax_h3_scenes.csv",
    "character": "resources/minimax_h3_characters.csv",
    "action": "resources/minimax_h3_actions.csv",
    "camera": "resources/minimax_h3_cameras.csv",
    "visual_style": "resources/minimax_h3_visual_styles.csv",
    "audio": "resources/minimax_h3_audio.csv",
}


def _normalize_prose(value):
    return " ".join(str(value or "").split())


def _join_text_and_phrases(text, phrases):
    text = _normalize_prose(text)
    phrases = [_normalize_prose(phrase) for phrase in phrases]
    phrases = [phrase for phrase in phrases if phrase]
    vocab_text = ", ".join(phrases)

    if not text:
        return vocab_text
    if not vocab_text:
        return text

    separator = " " if text.endswith((".", "!", "?", ",", ";", ":")) else ", "
    return f"{text}{separator}{vocab_text}"


def _append_sentence(text, sentence):
    text = _normalize_prose(text)
    sentence = _normalize_prose(sentence)
    if not text:
        return sentence
    if not sentence:
        return text

    separator = " " if text.endswith((".", "!", "?", ",", ";", ":")) else ". "
    return f"{text}{separator}{sentence}"


def _quoted_dialogue(dialogue):
    dialogue = str(dialogue or "").strip()
    if not dialogue:
        return ""

    # Preserve individually quoted multi-speaker lines. Wrapping their whole
    # value again would create invalid nested double quotes for downstream
    # dialogue protection and translation.
    quote_count = dialogue.count('"')
    if quote_count >= 2 and quote_count % 2 == 0:
        return dialogue

    quoted = f'"{dialogue}"'
    if dialogue.endswith((".", "!", "?")):
        return quoted
    return f"{quoted}."


def _voice_phrase(voice):
    voice = _normalize_prose(voice)
    if not voice:
        return "A voice"
    if voice.casefold().startswith(("a ", "an ", "the ")):
        return voice[0].upper() + voice[1:]
    return f"A {voice}"


def _sentence_case(text):
    text = _normalize_prose(text)
    if not text:
        return ""
    return text[0].upper() + text[1:]


def _ensure_sentence(text):
    text = _sentence_case(text)
    if text and not text.endswith((".", "!", "?")):
        text += "."
    return text


def _dialogue_sentences(dialogue, language, voice, delivery):
    quoted = _quoted_dialogue(dialogue)
    if not quoted:
        return "", ""

    language = _normalize_prose(language)
    language_clause = f" in {language}" if language else ""
    action_sentence = f"The character says{language_clause}: {quoted}"

    audio_parts = [f"{_voice_phrase(voice)} says{language_clause}: {quoted}"]
    delivery_sentence = _ensure_sentence(delivery)
    if delivery_sentence:
        audio_parts.append(delivery_sentence)
    audio_parts.append("No other dialogue.")
    return action_sentence, " ".join(audio_parts)


def _sample_section(vocab_file, amount, word_index, section_index):
    amount = max(0, int(amount))
    if amount == 0:
        return []

    words = _dedupe_preserve_order(_load_words(vocab_file))
    return _shuffle_bag_sample(words, amount, word_index, section_index)


class OnekoMiniMaxH3PromptBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        available_files = _available_vocab_files()
        default_file = USER_VOCAB_LABEL if USER_VOCAB_LABEL in available_files else available_files[0]
        required = {}

        for section_name, section_label in SECTIONS:
            section_default_file = DEFAULT_SECTION_VOCAB_FILES.get(section_name, default_file)
            if section_default_file not in available_files:
                section_default_file = default_file
            required[f"{section_name}_text"] = (
                "STRING",
                {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": f"Fixed natural-language content for [{section_label}].",
                },
            )
            required[f"{section_name}_vocab_file"] = (
                available_files,
                {
                    "default": section_default_file,
                    "tooltip": f"Optional comma-separated phrase list for [{section_label}].",
                },
            )
            required[f"{section_name}_amount"] = (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 10000,
                    "tooltip": f"Number of vocabulary phrases for [{section_label}]. Use 0 to disable sampling.",
                },
            )
            required[f"{section_name}_word_index"] = (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFFFFFFFFFF,
                    "control_after_generate": True,
                    "tooltip": f"Deterministic shuffle-bag block cursor for [{section_label}].",
                },
            )

        required.update(
            {
                "spoken_dialogue": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "Exact spoken line. It is quoted in both [Action] and [Audio].",
                    },
                ),
                "dialogue_language": (
                    "STRING",
                    {
                        "default": "German",
                        "multiline": False,
                        "tooltip": "Language used to describe the spoken dialogue.",
                    },
                ),
                "dialogue_voice": (
                    "STRING",
                    {
                        "default": "clear young female voice",
                        "multiline": False,
                        "tooltip": "Voice description inserted into [Audio].",
                    },
                ),
                "dialogue_delivery": (
                    "STRING",
                    {
                        "default": "natural speech, calm and slightly cautious tone",
                        "multiline": False,
                        "tooltip": "Speech delivery and tone inserted into [Audio].",
                    },
                ),
            }
        )
        return {"required": required}

    RETURN_TYPES = ("STRING",) * 7
    RETURN_NAMES = ("prompt", "scene", "character", "action", "camera", "visual_style", "audio")
    FUNCTION = "generate"
    CATEGORY = "Oneko/02 Text/Vocabulary"
    DESCRIPTION = "Builds a structured six-section MiniMax H3 video prompt with optional deterministic vocabulary sampling and spoken dialogue."

    def generate(self, **kwargs):
        section_outputs = {}
        for section_index, (section_name, _section_label) in enumerate(SECTIONS, start=1):
            phrases = _sample_section(
                kwargs.get(f"{section_name}_vocab_file", USER_VOCAB_LABEL),
                kwargs.get(f"{section_name}_amount", 0),
                kwargs.get(f"{section_name}_word_index", 0),
                section_index,
            )
            section_outputs[section_name] = _join_text_and_phrases(
                kwargs.get(f"{section_name}_text", ""),
                phrases,
            )

        action_dialogue, audio_dialogue = _dialogue_sentences(
            kwargs.get("spoken_dialogue", ""),
            kwargs.get("dialogue_language", "German"),
            kwargs.get("dialogue_voice", "clear young female voice"),
            kwargs.get("dialogue_delivery", "natural speech, calm and slightly cautious tone"),
        )
        section_outputs["action"] = _append_sentence(section_outputs["action"], action_dialogue)
        section_outputs["audio"] = _append_sentence(section_outputs["audio"], audio_dialogue)

        blocks = []
        for section_name, section_label in SECTIONS:
            content = section_outputs[section_name]
            if content:
                blocks.append(f"[{section_label}]\n{content}")
        prompt = "\n\n".join(blocks)

        return (
            prompt,
            section_outputs["scene"],
            section_outputs["character"],
            section_outputs["action"],
            section_outputs["camera"],
            section_outputs["visual_style"],
            section_outputs["audio"],
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        digest = hashlib.sha256()

        for section_name, _section_label in SECTIONS:
            for suffix, default in (
                ("text", ""),
                ("vocab_file", USER_VOCAB_LABEL),
                ("amount", 0),
                ("word_index", 0),
            ):
                digest.update(str(kwargs.get(f"{section_name}_{suffix}", default)).encode("utf-8"))

            vocab_file = kwargs.get(f"{section_name}_vocab_file", USER_VOCAB_LABEL)
            try:
                path = _resolve_vocab_path(vocab_file)
            except RuntimeError:
                path = ""
            if os.path.exists(path):
                with open(path, "rb") as vocab_file_handle:
                    digest.update(vocab_file_handle.read())
            else:
                digest.update(b"missing")

        for field_name, default in (
            ("spoken_dialogue", ""),
            ("dialogue_language", "German"),
            ("dialogue_voice", "clear young female voice"),
            ("dialogue_delivery", "natural speech, calm and slightly cautious tone"),
        ):
            digest.update(str(kwargs.get(field_name, default)).encode("utf-8"))

        return digest.hexdigest()


NODE_CLASS_MAPPINGS = {
    "OnekoMiniMaxH3PromptBuilder": OnekoMiniMaxH3PromptBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoMiniMaxH3PromptBuilder": "MiniMax H3 Prompt Builder (Oneko)"
}
