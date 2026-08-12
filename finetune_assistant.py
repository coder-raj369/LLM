"""Fine-tune a dialogue-ready Stage 3 checkpoint on JSONL conversations.

This is Stage 4a only. It teaches the model the turn format and optimizes loss
only on assistant replies; it intentionally does not add memory, retrieval, or
tool calling yet.
"""

import argparse
import random
from pathlib import Path

import torch

from assistant_data import (
    END_TOKEN,
    load_dialogue_examples,
    make_dialogue_batch,
    require_special_tokens,
    split_dialogue_examples,
)
from char_transformer import CharacterTransformer
from device import get_device
from tokenization import BPETokenizer


@torch.no_grad()
def evaluate(model, examples, batch_size, pad_token_id, device) -> float:
    """Return mean loss across complete dialogue examples."""
    model.eval()
    losses = []
    for start in range(0, len(examples), batch_size):
        indices = list(range(start, min(start + batch_size, len(examples))))
        inputs, targets = make_dialogue_batch(examples, indices, pad_token_id, device)
        _, loss = model(inputs, targets)
        losses.append((loss.item(), len(indices)))
    model.train()
    return sum(loss * count for loss, count in losses) / sum(count for _, count in losses)


def load_dialogue_checkpoint(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "model_config", "tokenizer"}
    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint must contain model_state, model_config, and tokenizer")
    if checkpoint["tokenizer"].get("kind") != "bpe":
        raise ValueError("Assistant fine-tuning currently requires a BPE checkpoint")

    tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    marker_ids = require_special_tokens(tokenizer)
    model_config = checkpoint["model_config"]
    if model_config["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("Checkpoint model vocabulary and tokenizer vocabulary do not match")
    model = CharacterTransformer(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint, tokenizer, marker_ids, model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a prepared checkpoint on dialogue JSONL data.")
    parser.add_argument("--base-checkpoint", required=True, help="Checkpoint made by prepare_assistant_checkpoint.py")
    parser.add_argument("--data", required=True, help="Dialogue JSONL file")
    parser.add_argument("--output", required=True, help="New fine-tuned checkpoint path")
    parser.add_argument("--epochs", type=int, default=3, help="Passes over the dialogue dataset")
    parser.add_argument("--batch-size", type=int, default=4, help="Small default for 8 GB unified memory")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Low learning rate for fine-tuning")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output file")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("--epochs, --batch-size, and --learning-rate must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} already exists; choose another path or pass --overwrite")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    base_checkpoint, tokenizer, marker_ids, model = load_dialogue_checkpoint(args.base_checkpoint, device)
    examples = load_dialogue_examples(args.data, tokenizer, model.block_size)
    train_examples, validation_examples = split_dialogue_examples(
        examples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    pad_token_id = marker_ids[END_TOKEN]

    print(f"Fine-tuning on {len(train_examples)} train and {len(validation_examples)} validation dialogues.")
    print(f"Device: {device}. Context: {model.block_size}. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_examples)))
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            inputs, targets = make_dialogue_batch(train_examples, indices, pad_token_id, device)
            _, loss = model(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        validation_loss = evaluate(model, validation_examples, args.batch_size, pad_token_id, device)
        print(
            f"epoch {epoch}/{args.epochs} | "
            f"train loss {sum(losses) / len(losses):.3f} | validation loss {validation_loss:.3f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 4,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": base_checkpoint["model_config"],
            "tokenizer": tokenizer.state_dict(),
            "stage4_metadata": {
                "base_checkpoint": str(Path(args.base_checkpoint)),
                "dialogue_data": str(Path(args.data)),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
            },
        },
        output,
    )
    print(f"Saved fine-tuned assistant checkpoint to {output}")


if __name__ == "__main__":
    main()
