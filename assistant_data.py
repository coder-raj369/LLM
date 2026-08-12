"""Dialogue formatting and batches for Stage 4 assistant fine-tuning.

The format deliberately stays small: a conversation is a JSON object with a
``messages`` list containing alternating ``user`` and ``assistant`` turns.
Only assistant response tokens contribute to the training loss.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from char_transformer import IGNORE_INDEX
from tokenization import BPETokenizer

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
END_TOKEN = "<|end|>"
SPECIAL_TOKENS = (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN)


@dataclass
class DialogueExample:
    """A next-token example whose labels mask non-assistant targets."""

    input_ids: list[int]
    labels: list[int]


def require_special_tokens(tokenizer: BPETokenizer) -> dict[str, int]:
    """Return role-token IDs or explain which checkpoint-preparation step is missing."""
    missing = [token for token in SPECIAL_TOKENS if token not in tokenizer.special_tokens]
    if missing:
        raise ValueError(
            "Tokenizer is missing reserved dialogue tokens: "
            f"{', '.join(missing)}. Run prepare_assistant_checkpoint.py first."
        )
    return tokenizer.special_tokens


def encode_turn(role: str, content: str, tokenizer: BPETokenizer, close_turn: bool = True) -> list[int]:
    """Encode one dialogue turn while keeping role markers outside BPE merges."""
    marker_ids = require_special_tokens(tokenizer)
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'")
    if any(marker in content for marker in SPECIAL_TOKENS):
        raise ValueError("turn content cannot contain a reserved dialogue marker")
    marker = USER_TOKEN if role == "user" else ASSISTANT_TOKEN
    try:
        content_ids = tokenizer.encode(content)
    except ValueError as error:
        raise ValueError(f"Cannot encode {role} content: {error}") from error
    token_ids = [marker_ids[marker], *content_ids]
    if close_turn:
        token_ids.append(marker_ids[END_TOKEN])
    return token_ids


def _validate_messages(messages: object) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("messages must be a list containing at least one user/assistant exchange")

    expected_role = "user"
    checked_messages = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role != expected_role:
            raise ValueError(f"message {index} must have role {expected_role!r}, not {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} needs non-empty string content")
        checked_messages.append({"role": role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    if checked_messages[-1]["role"] != "assistant":
        raise ValueError("a dialogue must end with an assistant response")
    return checked_messages


def build_dialogue_example(
    messages: object,
    tokenizer: BPETokenizer,
    block_size: int,
) -> DialogueExample:
    """Format a dialogue and mask all target tokens except assistant output.

    The input at position ``i`` predicts the token at ``i + 1``. The assistant
    marker itself is masked because the chat harness inserts it mechanically;
    the first assistant-content token and its closing end marker are trained.
    """
    marker_ids = require_special_tokens(tokenizer)
    messages = _validate_messages(messages)

    full_ids = []
    assistant_output = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        turn_ids = encode_turn(role, content, tokenizer)
        full_ids.append(turn_ids[0])
        assistant_output.append(False)
        content_ids = turn_ids[1:-1]
        full_ids.extend(content_ids)
        assistant_output.extend([role == "assistant"] * len(content_ids))
        full_ids.append(marker_ids[END_TOKEN])
        assistant_output.append(role == "assistant")

    input_ids = full_ids[:-1]
    labels = [
        token_id if should_learn else IGNORE_INDEX
        for token_id, should_learn in zip(full_ids[1:], assistant_output[1:])
    ]
    if len(input_ids) > block_size:
        raise ValueError(
            f"Dialogue has {len(input_ids)} input tokens, exceeding the {block_size}-token context window. "
            "Shorten the dialogue instead of silently cutting off its answer."
        )
    if not any(label != IGNORE_INDEX for label in labels):
        raise ValueError("Dialogue has no assistant tokens to learn from")
    return DialogueExample(input_ids=input_ids, labels=labels)


def load_dialogue_examples(path: str | Path, tokenizer: BPETokenizer, block_size: int) -> list[DialogueExample]:
    """Load one JSON object per line and convert each complete dialogue safely."""
    examples = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} must contain one JSON object")
        try:
            examples.append(build_dialogue_example(record.get("messages"), tokenizer, block_size))
        except ValueError as error:
            raise ValueError(f"Invalid dialogue on line {line_number}: {error}") from error

    if len(examples) < 2:
        raise ValueError("Need at least two complete dialogue examples for train/validation splitting")
    return examples


def split_dialogue_examples(
    examples: list[DialogueExample],
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[list[DialogueExample], list[DialogueExample]]:
    """Split by complete dialogue, never through the middle of a conversation."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(examples) < 2:
        raise ValueError("Need at least two dialogue examples")

    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count]


def make_dialogue_batch(
    examples: list[DialogueExample],
    indices: list[int],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad selected dialogues; padding targets never affect the loss.

    Right padding is safe for causal attention because valid tokens cannot see
    the padding that comes after them. The pad ID is the end-of-turn token, and
    all matching target positions use ``IGNORE_INDEX``.
    """
    selected = [examples[index] for index in indices]
    if not selected:
        raise ValueError("Cannot create a batch without dialogue examples")
    length = max(len(example.input_ids) for example in selected)
    inputs = []
    labels = []
    for example in selected:
        padding = length - len(example.input_ids)
        inputs.append(example.input_ids + [pad_token_id] * padding)
        labels.append(example.labels + [IGNORE_INDEX] * padding)

    target_tensor = torch.tensor(labels, dtype=torch.long)
    if not torch.any(target_tensor != IGNORE_INDEX):
        raise ValueError("Dialogue batch has no assistant targets")
    return torch.tensor(inputs, dtype=torch.long, device=device), target_tensor.to(device)
