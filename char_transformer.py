"""A small GPT-style decoder-only Transformer trained from local text.

The aim of this file is to show the important parts clearly.  It does not use a
tokenizer package, an API, or a pre-trained model.  PyTorch is the only
third-party dependency.

Example:
    python3 char_transformer.py --input raj_bio.txt --steps 1500
"""

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenization import BPETokenizer


class CharacterTokenizer:
    """Maps every character in the training file to an integer and back."""

    def __init__(self, text: str):
        self.characters = sorted(set(text))
        self.stoi = {character: index for index, character in enumerate(self.characters)}
        self.itos = {index: character for character, index in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.characters)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[character] for character in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[index] for index in ids)

    def state_dict(self) -> dict:
        return {"kind": "character", "characters": self.characters}

    @classmethod
    def from_state_dict(cls, state: dict):
        tokenizer = cls.__new__(cls)
        tokenizer.characters = state["characters"]
        tokenizer.stoi = {character: index for index, character in enumerate(tokenizer.characters)}
        tokenizer.itos = {index: character for character, index in tokenizer.stoi.items()}
        return tokenizer


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where a character cannot see future text."""

    def __init__(self, embed_size: int, heads: int, block_size: int, dropout: float):
        super().__init__()
        if embed_size % heads != 0:
            raise ValueError("embed_size must be divisible by heads")

        self.heads = heads
        self.head_size = embed_size // heads
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

        # Stored once, instead of creating the triangular mask on every pass.
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = x.shape

        key = self.key(x).view(batch_size, time_steps, self.heads, self.head_size)
        query = self.query(x).view(batch_size, time_steps, self.heads, self.head_size)
        value = self.value(x).view(batch_size, time_steps, self.heads, self.head_size)

        # Put the heads before the time dimension: (B, heads, T, head_size).
        key = key.transpose(1, 2)
        query = query.transpose(1, 2)
        value = value.transpose(1, 2)

        attention = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)
        attention = attention.masked_fill(~self.causal_mask[:time_steps, :time_steps], float("-inf"))
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(batch_size, time_steps, channels)
        return self.dropout(self.projection(output))


class FeedForward(nn.Module):
    """The small MLP applied independently to each character position."""

    def __init__(self, embed_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, block_size: int, dropout: float):
        super().__init__()
        self.attention = CausalSelfAttention(embed_size, heads, block_size, dropout)
        self.feed_forward = FeedForward(embed_size, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-layer normalisation with residual connections is a stable default.
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class CharacterTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        embed_size: int = 96,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        # Learned positional embeddings let the model distinguish token order.
        self.position_embedding = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_size, heads, block_size, dropout) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(embed_size)
        self.language_model_head = nn.Linear(embed_size, vocab_size)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        _, time_steps = tokens.shape
        if time_steps > self.block_size:
            raise ValueError(f"sequence has {time_steps} tokens; max is {self.block_size}")

        positions = torch.arange(time_steps, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.language_model_head(self.final_norm(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens: torch.Tensor,
        new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Adds sampled tokens to the end of a prompt."""
        self.eval()
        for _ in range(new_tokens):
            context = tokens[:, -self.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                cutoff = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


def make_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device):
    """Pick random snippets and the next-character labels for each snippet."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    inputs = torch.stack([data[start : start + block_size] for start in starts])
    targets = torch.stack([data[start + 1 : start + block_size + 1] for start in starts])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, validation_data, batch_size, block_size, device, batches=20):
    model.eval()
    losses = {}
    for name, data in {"train": train_data, "validation": validation_data}.items():
        values = []
        for _ in range(batches):
            inputs, targets = make_batch(data, batch_size, block_size, device)
            _, loss = model(inputs, targets)
            values.append(loss.item())
        losses[name] = sum(values) / len(values)
    model.train()
    return losses


def main():
    parser = argparse.ArgumentParser(description="Train a small Transformer from plain text.")
    parser.add_argument("--input", help="Plain-text training file (needed when training)")
    parser.add_argument("--steps", type=int, default=1500, help="Number of training updates")
    parser.add_argument("--preset", choices=("learning", "small-gpt"), default="learning")
    parser.add_argument("--tokenizer", choices=("character", "bpe"), default="bpe")
    parser.add_argument("--bpe-merges", type=int, help="How many BPE merges to learn")
    parser.add_argument("--context", type=int, help="Maximum number of input tokens")
    parser.add_argument("--embed-size", type=int, help="Size of each token embedding")
    parser.add_argument("--heads", type=int, help="Number of attention heads")
    parser.add_argument("--layers", type=int, help="Number of Transformer blocks")
    parser.add_argument("--dropout", type=float, help="Dropout probability")
    parser.add_argument("--batch-size", type=int, help="Sequences in each training batch")
    parser.add_argument("--learning-rate", type=float, help="AdamW learning rate")
    parser.add_argument("--eval-interval", type=int, default=250, help="Steps between validation checks")
    parser.add_argument("--eval-batches", type=int, default=20, help="Batches used for each validation estimate")
    parser.add_argument("--sample-length", type=int, default=300, help="Tokens to generate")
    parser.add_argument("--prompt", default="", help="Optional starting text made of training characters")
    parser.add_argument("--temperature", type=float, default=0.7, help="Lower is safer; higher is more varied")
    parser.add_argument("--top-k", type=int, default=12, help="Sample only from the k most likely tokens; 0 disables it")
    parser.add_argument("--chat", action="store_true", help="Talk to the model after training")
    parser.add_argument("--save", help="Path for saving a complete checkpoint")
    parser.add_argument("--load", help="Path for loading a complete checkpoint")
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps cannot be negative")
    if args.eval_interval <= 0 or args.eval_batches <= 0:
        parser.error("--eval-interval and --eval-batches must be positive")

    presets = {
        "learning": {
            "bpe_merges": 40,
            "context": 128,
            "embed_size": 128,
            "heads": 4,
            "layers": 3,
            "dropout": 0.1,
            "batch_size": 32,
            "learning_rate": 3e-4,
        },
        "small-gpt": {
            "bpe_merges": 300,
            "context": 256,
            "embed_size": 192,
            "heads": 6,
            "layers": 4,
            "dropout": 0.1,
            "batch_size": 24,
            "learning_rate": 3e-4,
        },
    }
    settings = presets[args.preset].copy()
    for name in settings:
        override = getattr(args, name)
        if override is not None:
            settings[name] = override
    if settings["embed_size"] % settings["heads"] != 0:
        parser.error("--embed-size must be divisible by --heads")

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.load:
        checkpoint = torch.load(args.load, map_location=device, weights_only=True)
        if "model_state" not in checkpoint or "tokenizer" not in checkpoint:
            raise ValueError("This is not a Stage 2 checkpoint. Train and save the model again to create one.")
        tokenizer_state = checkpoint["tokenizer"]
        tokenizer = (
            CharacterTokenizer.from_state_dict(tokenizer_state)
            if tokenizer_state["kind"] == "character"
            else BPETokenizer.from_state_dict(tokenizer_state)
        )
        model_config = checkpoint["model_config"]
    else:
        if not args.input:
            parser.error("--input is required unless --load is used")
        text = Path(args.input).read_text(encoding="utf-8")
        tokenizer = CharacterTokenizer(text) if args.tokenizer == "character" else BPETokenizer.train(text, settings["bpe_merges"])
        model_config = {
            "vocab_size": tokenizer.vocab_size,
            "block_size": settings["context"],
            "embed_size": settings["embed_size"],
            "heads": settings["heads"],
            "layers": settings["layers"],
            "dropout": settings["dropout"],
        }
        checkpoint = None

    model = CharacterTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint and args.steps > 0:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Loaded checkpoint from {args.load}.")

    block_size = model.block_size
    if args.steps > 0:
        if not args.input:
            parser.error("--input is required when continuing training")
        text = Path(args.input).read_text(encoding="utf-8")
        encoded_text = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        if len(encoded_text) < block_size * 3:
            raise ValueError(f"{args.input} becomes too short after tokenization. Use at least {block_size * 3} tokens.")

        # Keep a normal 10% validation set, but never make it shorter than one batch window.
        validation_size = max(int(0.1 * len(encoded_text)), block_size + 2)
        split = len(encoded_text) - validation_size
        train_data, validation_data = encoded_text[:split], encoded_text[split:]

        print(f"Preset: {args.preset}. Training on {len(encoded_text):,} tokens with a vocabulary of {tokenizer.vocab_size} tokens.")
        print(f"Context: {block_size}. Batch size: {settings['batch_size']}. Device: {device}. Parameters: {sum(p.numel() for p in model.parameters()):,}")
        for step in range(args.steps):
            inputs, targets = make_batch(train_data, batch_size=settings["batch_size"], block_size=block_size, device=device)
            _, loss = model(inputs, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if step % args.eval_interval == 0 or step == args.steps - 1:
                losses = estimate_loss(
                    model,
                    train_data,
                    validation_data,
                    settings["batch_size"],
                    block_size,
                    device,
                    args.eval_batches,
                )
                print(f"step {step:4d} | train loss {losses['train']:.3f} | validation loss {losses['validation']:.3f}")

    if args.save:
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "model_config": model_config,
                "tokenizer": tokenizer.state_dict(),
            },
            args.save,
        )
        print(f"Saved complete checkpoint to {args.save}.")

    if args.chat:
        print("\nChat is ready. Type 'quit' to stop.")
        while True:
            question = input("You: ").strip()
            if question.lower() in {"quit", "exit"}:
                break
            if not question:
                continue

            prompt = f"User: {question}\nAssistant:"
            unknown = set(prompt) - set(tokenizer.characters)
            if unknown:
                print("Assistant: I cannot use these characters yet:", "".join(sorted(unknown)))
                continue
            start = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            answer = tokenizer.decode(model.generate(start, 120, args.temperature, args.top_k)[0].tolist())
            # Only print the new part, not the prompt the user already typed.
            print("Assistant:" + answer[len(prompt):].split("\nUser:")[0].strip())
        return

    if args.prompt:
        unknown = set(args.prompt) - set(tokenizer.characters)
        if unknown:
            raise ValueError(f"Prompt has characters not found in the training file: {sorted(unknown)}")
        start = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    else:
        start = torch.zeros((1, 1), dtype=torch.long, device=device)

    generated = model.generate(start, args.sample_length, args.temperature, args.top_k)[0].tolist()
    print("\n--- Generated text ---")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
