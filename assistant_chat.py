"""Terminal chat for a Stage 4 dialogue-fine-tuned checkpoint.

The runner owns the conversation format: it inserts role markers, keeps recent
complete turns inside the context window, and stops sampling at ``<|end|>``.
It intentionally has no long-term memory or tools yet.
"""

import argparse
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from assistant_data import ASSISTANT_TOKEN, END_TOKEN, USER_TOKEN, encode_turn, require_special_tokens
from char_transformer import CharacterTransformer
from device import get_device
from tokenization import BPETokenizer


@dataclass
class Turn:
    role: str
    content: str
    token_ids: list[int]


def load_assistant(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state", "model_config", "tokenizer"}
    if not required.issubset(checkpoint):
        raise ValueError("Checkpoint must contain model_state, model_config, and tokenizer")
    tokenizer = BPETokenizer.from_state_dict(checkpoint["tokenizer"])
    marker_ids = require_special_tokens(tokenizer)
    if checkpoint["model_config"]["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("Checkpoint model vocabulary and tokenizer vocabulary do not match")
    model = CharacterTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, tokenizer, marker_ids


def context_for_response(history: list[Turn], user_turn: Turn, assistant_token_id: int, block_size: int) -> list[int]:
    """Keep the newest complete history turns that fit before an assistant reply."""
    pending_ids = [*user_turn.token_ids, assistant_token_id]
    if len(pending_ids) > block_size:
        raise ValueError(
            f"Your message uses {len(pending_ids) - 1} tokens, exceeding the {block_size}-token context window. "
            "Please send a shorter message."
        )

    if len(history) % 2:
        raise ValueError("Conversation history must contain complete user/assistant exchanges")

    context = pending_ids.copy()
    # Keep a full user/assistant exchange or drop it; never start context with
    # an orphaned assistant reply whose question has already been discarded.
    for index in range(len(history) - 2, -1, -2):
        exchange_ids = [*history[index].token_ids, *history[index + 1].token_ids]
        if len(exchange_ids) + len(context) > block_size:
            break
        context = [*exchange_ids, *context]
    return context


def sample_token(logits: torch.Tensor, temperature: float, top_k: int, top_p: float, blocked_ids: set[int]) -> int:
    """Sample one token using optional top-k and nucleus (top-p) filtering."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = logits.clone() / temperature
    for token_id in blocked_ids:
        logits[token_id] = float("-inf")

    if top_k > 0:
        cutoff = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < cutoff] = float("-inf")
    if 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_indices[remove]] = float("-inf")

    probabilities = F.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum().item() == 0:
        raise RuntimeError("Sampling removed every possible token")
    return torch.multinomial(probabilities, num_samples=1).item()


@torch.no_grad()
def generate_response(
    model: CharacterTransformer,
    context_ids: list[int],
    marker_ids: dict[str, int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device,
) -> tuple[list[int], bool]:
    """Generate until a closing end marker or the configured token limit."""
    tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
    generated = []
    blocked_ids = {marker_ids[USER_TOKEN], marker_ids[ASSISTANT_TOKEN]}
    for _ in range(max_new_tokens):
        logits, _ = model(tokens[:, -model.block_size :])
        next_id = sample_token(logits[0, -1], temperature, top_k, top_p, blocked_ids)
        if next_id == marker_ids[END_TOKEN]:
            return generated, True
        generated.append(next_id)
        tokens = torch.cat((tokens, torch.tensor([[next_id]], device=device)), dim=1)
    return generated, False


def main():
    parser = argparse.ArgumentParser(description="Chat with a Stage 4 dialogue-fine-tuned checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Output from finetune_assistant.py")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.temperature <= 0 or args.top_k < 0 or not 0 < args.top_p <= 1 or args.max_new_tokens <= 0:
        parser.error("Use positive temperature/max-new-tokens, non-negative top-k, and top-p in (0, 1]")

    device = get_device()
    model, tokenizer, marker_ids = load_assistant(args.checkpoint, device)
    history: list[Turn] = []
    print("Assistant chat is ready. Type 'quit' or 'exit' to stop.")
    print("This small model only supports characters learned by its Stage 3 tokenizer.")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            break
        if not user_text:
            continue
        try:
            user_ids = encode_turn("user", user_text, tokenizer)
            user_turn = Turn("user", user_text, user_ids)
            context_ids = context_for_response(history, user_turn, marker_ids[ASSISTANT_TOKEN], model.block_size)
            response_ids, ended = generate_response(
                model,
                context_ids,
                marker_ids,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                device,
            )
        except (ValueError, RuntimeError) as error:
            print(f"Assistant: I cannot process that message: {error}")
            continue

        response = tokenizer.decode(response_ids).strip()
        print(f"Assistant: {response}")
        if not ended:
            print("[Response stopped at the token limit before <|end|>.]")
        assistant_ids = [marker_ids[ASSISTANT_TOKEN], *response_ids, marker_ids[END_TOKEN]]
        history.extend((user_turn, Turn("assistant", response, assistant_ids)))


if __name__ == "__main__":
    main()
