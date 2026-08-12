"""Add dialogue markers to a pretrained checkpoint without changing its base rows.

Example:
    python3 prepare_assistant_checkpoint.py \
        --base-checkpoint small_gpt.pt \
        --output checkpoints/small_gpt_assistant_base.pt
"""

import argparse
from pathlib import Path

import torch

from assistant_data import SPECIAL_TOKENS
from char_transformer import CharacterTransformer
from tokenization import BPETokenizer

VOCAB_TENSORS = {
    "token_embedding.weight",
    "language_model_head.weight",
    "language_model_head.bias",
}


def extend_checkpoint(checkpoint: dict, special_tokens: tuple[str, ...], seed: int) -> dict:
    """Return a checkpoint with appended marker IDs and preserved base weights."""
    required = {"model_state", "model_config", "tokenizer"}
    if not required.issubset(checkpoint):
        raise ValueError("Base checkpoint must contain model_state, model_config, and tokenizer")
    if checkpoint["tokenizer"].get("kind") != "bpe":
        raise ValueError("Stage 4 dialogue adaptation currently requires a BPE checkpoint")

    tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    old_vocab_size = tokenizer.vocab_size
    model_config = dict(checkpoint["model_config"])
    if model_config["vocab_size"] != old_vocab_size:
        raise ValueError("Checkpoint model vocabulary and tokenizer vocabulary do not match")

    tokenizer.add_special_tokens(special_tokens)
    new_vocab_size = tokenizer.vocab_size
    model_config["vocab_size"] = new_vocab_size

    # Constructing a normal model gives new rows the same PyTorch initialization
    # as the rest of the project. Only the old rows are overwritten below.
    torch.manual_seed(seed)
    expanded_model = CharacterTransformer(**model_config)
    expanded_state = expanded_model.state_dict()
    base_state = checkpoint["model_state"]

    if set(base_state) != set(expanded_state):
        raise ValueError("Checkpoint tensors do not match the current model architecture")
    for name, base_tensor in base_state.items():
        target_tensor = expanded_state[name]
        if name in VOCAB_TENSORS:
            if target_tensor.shape[0] != new_vocab_size or base_tensor.shape[0] != old_vocab_size:
                raise ValueError(f"Unexpected vocabulary tensor shape for {name}")
            if target_tensor.shape[1:] != base_tensor.shape[1:]:
                raise ValueError(f"Unexpected non-vocabulary shape change for {name}")
            target_tensor[:old_vocab_size].copy_(base_tensor)
        else:
            if target_tensor.shape != base_tensor.shape:
                raise ValueError(f"Unexpected shape change for {name}")
            target_tensor.copy_(base_tensor)

    expanded_model.load_state_dict(expanded_state)
    for name in VOCAB_TENSORS:
        if not torch.equal(expanded_model.state_dict()[name][:old_vocab_size], base_state[name]):
            raise RuntimeError(f"Base rows were not preserved for {name}")

    return {
        "format_version": 4,
        "model_state": expanded_model.state_dict(),
        "model_config": model_config,
        "tokenizer": tokenizer.state_dict(),
        "stage4_metadata": {
            "base_vocab_size": old_vocab_size,
            "special_tokens": tokenizer.special_tokens,
            "initialization_seed": seed,
            "purpose": "assistant dialogue adaptation base",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare a pretrained BPE checkpoint for dialogue fine-tuning.")
    parser.add_argument("--base-checkpoint", required=True, help="Untouched Stage 3 checkpoint")
    parser.add_argument("--output", required=True, help="New checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="Seed used only to initialize appended rows")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists; choose another path or pass --overwrite")

    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    converted = extend_checkpoint(checkpoint, SPECIAL_TOKENS, args.seed)
    converted["stage4_metadata"]["base_checkpoint"] = str(Path(args.base_checkpoint))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output)
    print(f"Saved dialogue-ready checkpoint to {output}")
    print(f"Vocabulary: {converted['stage4_metadata']['base_vocab_size']} -> {converted['model_config']['vocab_size']}")
    print(f"Reserved tokens: {', '.join(SPECIAL_TOKENS)}")


if __name__ == "__main__":
    main()
