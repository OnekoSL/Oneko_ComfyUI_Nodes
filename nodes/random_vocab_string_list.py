import hashlib
import os
import random

import folder_paths

RESOURCE_EXTENSIONS = (".csv", ".txt", ".json")
USER_VOCAB_LABEL = "user/vocab.json"
MULTI_SLOT_COUNT = 4


def _vocab_path():
    return os.path.join(folder_paths.get_user_directory(), "vocab.json")


def _resources_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "resources"))


def _list_resource_vocab_files():
    resources_dir = _resources_dir()
    if not os.path.isdir(resources_dir):
        return []

    files = []
    for file_name in os.listdir(resources_dir):
        if file_name.lower().endswith(RESOURCE_EXTENSIONS):
            files.append(f"resources/{file_name}")
    return sorted(files)


def _available_vocab_files():
    files = []
    user_path = _vocab_path()
    if os.path.exists(user_path):
        files.append(USER_VOCAB_LABEL)
    files.extend(_list_resource_vocab_files())
    if not files:
        files.append(USER_VOCAB_LABEL)
    return files


def _resolve_vocab_path(vocab_file):
    if vocab_file == USER_VOCAB_LABEL:
        user_path = _vocab_path()
        if os.path.exists(user_path):
            return user_path

        resource_files = _list_resource_vocab_files()
        if resource_files:
            return _resolve_vocab_path(resource_files[0])

        raise RuntimeError(f"Random Vocab String List: vocab file not found: {user_path}")

    if vocab_file.startswith("resources/"):
        file_name = os.path.basename(vocab_file)
        path = os.path.abspath(os.path.join(_resources_dir(), file_name))
        resources_dir = _resources_dir()
        if os.path.commonpath([resources_dir, path]) != resources_dir:
            raise RuntimeError(f"Random Vocab String List: invalid vocab file: {vocab_file}")
        if os.path.exists(path):
            return path

    raise RuntimeError(f"Random Vocab String List: vocab file not found: {vocab_file}")


def _load_words(vocab_file):
    path = _resolve_vocab_path(vocab_file)
    with open(path, "r", encoding="utf-8") as vocab_file:
        text = vocab_file.read()

    words = [part.strip() for part in text.split(",") if part.strip()]
    if not words:
        raise RuntimeError(f"Random Vocab String List: vocab file is empty: {path}")

    return words


def _dedupe_preserve_order(words):
    seen = set()
    result = []
    for word in words:
        if word in seen:
            continue
        seen.add(word)
        result.append(word)
    return result


def _slot_seed(seed, slot_index):
    return (int(seed) + (slot_index * 1000003)) & 0xffffffffffffffff


def _shuffle_bag_sample(words, amount, word_index, slot_index):
    words = _dedupe_preserve_order(words)
    amount = min(max(0, int(amount)), len(words))
    if amount <= 0:
        return []

    cursor = int(word_index) * amount
    selected = []
    while len(selected) < amount:
        cycle_index = cursor // len(words)
        cycle_offset = cursor % len(words)
        shuffled = list(words)
        random.Random(_slot_seed(cycle_index, slot_index)).shuffle(shuffled)
        take_count = min(amount - len(selected), len(words) - cycle_offset)
        selected.extend(shuffled[cycle_offset : cycle_offset + take_count])
        cursor += take_count

    return selected


def _generate_slot(vocab_file, amount, word_index, slot_index):
    amount = max(0, int(amount))
    if amount <= 0:
        return ""

    words = _dedupe_preserve_order(_load_words(vocab_file))
    return " ".join(_shuffle_bag_sample(words, amount, word_index, slot_index))


class OnekoRandomVocabStringList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "chain": (
                    "STRING",
                    {
                        "default": "",
                        "defaultInput": True,
                        "forceInput": True,
                        "multiline": False,
                        "tooltip": "Optional incoming string. When connected, generated words are appended after it.",
                    },
                ),
            },
            "required": {
                "vocab_file": (
                    _available_vocab_files(),
                    {
                        "default": USER_VOCAB_LABEL if USER_VOCAB_LABEL in _available_vocab_files() else _available_vocab_files()[0],
                        "tooltip": "Comma-separated word list to sample from. Add .csv, .txt, or .json files to this node pack's resources folder.",
                    },
                ),
                "amount": (
                    "INT",
                    {
                        "default": 10,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "Number of random words to output. Clamped to the number of available vocab words.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xffffffffffffffff,
                        "control_after_generate": True,
                        "tooltip": "Shuffle-bag block cursor for deterministic random words. Increment this to walk the vocabulary without repeats before reshuffling.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "Oneko/02 Text/Vocabulary"
    DESCRIPTION = "Outputs a deterministic shuffle-bag word list from a selectable vocabulary file, optionally appended to an incoming string."

    def generate(self, vocab_file, amount, seed, chain=""):
        generated = _generate_slot(vocab_file, amount, seed, 1)
        chain = chain.strip()
        if chain and generated:
            return (f"{chain} {generated}",)
        if chain:
            return (chain,)
        return (generated,)

    @classmethod
    def IS_CHANGED(cls, vocab_file, amount, seed, chain=""):
        try:
            path = _resolve_vocab_path(vocab_file)
        except RuntimeError:
            path = ""

        digest = hashlib.sha256()
        digest.update(str(chain).encode("utf-8"))
        digest.update(str(vocab_file).encode("utf-8"))
        digest.update(str(int(amount)).encode("utf-8"))
        digest.update(str(int(seed)).encode("utf-8"))

        if os.path.exists(path):
            with open(path, "rb") as vocab_file:
                digest.update(vocab_file.read())
        else:
            digest.update(b"missing")

        return digest.hexdigest()


class OnekoVocabMultiStringList:
    @classmethod
    def INPUT_TYPES(cls):
        available_files = _available_vocab_files()
        default_file = USER_VOCAB_LABEL if USER_VOCAB_LABEL in available_files else available_files[0]

        required = {}

        for slot in range(1, MULTI_SLOT_COUNT + 1):
            required[f"vocab_file_{slot}"] = (
                available_files,
                {
                    "default": default_file,
                    "tooltip": f"Comma-separated word list for slot {slot}.",
                },
            )
            required[f"amount_{slot}"] = (
                "INT",
                {
                    "default": 3,
                    "min": 0,
                    "max": 10000,
                    "tooltip": f"Number of words to output from slot {slot}. Use 0 to disable the slot.",
                },
            )
            required[f"word_index_{slot}"] = (
                "INT",
                {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "control_prefix": f"word_index_{slot}",
                    "tooltip": f"Block cursor for slot {slot}. Increment/decrement/randomize this to choose the next word block.",
                },
            )

        return {
            "optional": {
                "chain": (
                    "STRING",
                    {
                        "default": "",
                        "defaultInput": True,
                        "forceInput": True,
                        "multiline": False,
                        "tooltip": "Optional incoming string. When connected, the combined output is appended after it.",
                    },
                ),
            },
            "required": required,
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("combined", "slot_1", "slot_2", "slot_3", "slot_4")
    FUNCTION = "generate"
    CATEGORY = "Oneko/02 Text/Vocabulary"
    DESCRIPTION = "Combines four selectable vocabulary files into one deterministic prompt string using per-slot incrementable word cursors."

    def generate(self, chain="", **kwargs):
        slot_outputs = []
        for slot in range(1, MULTI_SLOT_COUNT + 1):
            slot_outputs.append(
                _generate_slot(
                    kwargs[f"vocab_file_{slot}"],
                    kwargs[f"amount_{slot}"],
                    kwargs[f"word_index_{slot}"],
                    slot,
                )
            )

        combined_parts = [slot_output for slot_output in slot_outputs if slot_output]
        combined = " ".join(combined_parts)
        chain = chain.strip()
        if chain and combined:
            combined = f"{chain} {combined}"
        elif chain:
            combined = chain

        return (combined, *slot_outputs)

    @classmethod
    def IS_CHANGED(cls, chain="", **kwargs):
        digest = hashlib.sha256()
        digest.update(str(chain).encode("utf-8"))

        for slot in range(1, MULTI_SLOT_COUNT + 1):
            vocab_file = kwargs.get(f"vocab_file_{slot}", "")
            digest.update(str(vocab_file).encode("utf-8"))
            digest.update(str(int(kwargs.get(f"amount_{slot}", 0))).encode("utf-8"))
            digest.update(str(int(kwargs.get(f"word_index_{slot}", 0))).encode("utf-8"))

            try:
                path = _resolve_vocab_path(vocab_file)
            except RuntimeError:
                path = ""
            if os.path.exists(path):
                with open(path, "rb") as vocab_file_handle:
                    digest.update(vocab_file_handle.read())
            else:
                digest.update(b"missing")

        return digest.hexdigest()


NODE_CLASS_MAPPINGS = {
    "OnekoRandomVocabStringList": OnekoRandomVocabStringList,
    "OnekoVocabMultiStringList": OnekoVocabMultiStringList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OnekoRandomVocabStringList": "Random Vocab String List (Oneko)",
    "OnekoVocabMultiStringList": "Multi Vocab String List (Oneko)"
}
