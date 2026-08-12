"""Checks for the Stage 4 dialogue-adaptation foundations.

Run with: python3 test_stage4.py
"""

import torch

from assistant_data import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    IGNORE_INDEX,
    USER_TOKEN,
    build_dialogue_example,
    make_dialogue_batch,
)
from assistant_chat import Turn, context_for_response, generate_response
from char_transformer import CharacterTransformer
from prepare_assistant_checkpoint import extend_checkpoint
from tokenization import BPETokenizer


def main():
    text = "Hello. How are you. I am fine. " * 30
    tokenizer = BPETokenizer.train(text, max_merges=12)
    plain_ids = tokenizer.encode("Hello. How are you.")
    model_config = {"vocab_size": tokenizer.vocab_size, "block_size": 32, "embed_size": 32, "heads": 4, "layers": 2, "dropout": 0.0}
    model = CharacterTransformer(**model_config)
    base_checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "tokenizer": tokenizer.state_dict(),
    }

    prepared = extend_checkpoint(base_checkpoint, (USER_TOKEN, ASSISTANT_TOKEN, END_TOKEN), seed=7)
    prepared_tokenizer = BPETokenizer.from_state_dict(prepared["tokenizer"])
    assert prepared_tokenizer.vocab_size == tokenizer.vocab_size + 3
    assert prepared_tokenizer.encode("Hello. How are you.") == plain_ids
    dialogue_text = "<|user|>Hello.<|end|><|assistant|>I am fine.<|end|>"
    assert prepared_tokenizer.decode(prepared_tokenizer.encode(dialogue_text)) == dialogue_text

    for name in ("token_embedding.weight", "language_model_head.weight", "language_model_head.bias"):
        old_rows = base_checkpoint["model_state"][name]
        new_rows = prepared["model_state"][name][: tokenizer.vocab_size]
        assert torch.equal(old_rows, new_rows)

    messages = [
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "I am fine."},
    ]
    example = build_dialogue_example(messages, prepared_tokenizer, block_size=32)
    answer_targets = [token_id for token_id in example.labels if token_id != IGNORE_INDEX]
    assert prepared_tokenizer.decode(answer_targets) == "I am fine.<|end|>"
    inputs, targets = make_dialogue_batch(
        [example, example],
        [0, 1],
        prepared_tokenizer.special_tokens[END_TOKEN],
        torch.device("cpu"),
    )
    prepared_model = CharacterTransformer(**prepared["model_config"])
    prepared_model.load_state_dict(prepared["model_state"])
    _, loss = prepared_model(inputs, targets)
    assert torch.isfinite(loss)

    user_turn = Turn("user", "Hello.", [1, 2])
    assistant_turn = Turn("assistant", "I am fine.", [3, 4])
    next_user_turn = Turn("user", "How are you.", [5, 6])
    assert context_for_response([user_turn, assistant_turn], next_user_turn, 7, block_size=6) == [5, 6, 7]
    assert context_for_response([user_turn, assistant_turn], next_user_turn, 7, block_size=8) == [1, 2, 3, 4, 5, 6, 7]

    # A strongly preferred end marker proves generation stops on the required token.
    prepared_model.language_model_head.weight.data.zero_()
    prepared_model.language_model_head.bias.data.fill_(-100.0)
    prepared_model.language_model_head.bias.data[prepared_tokenizer.special_tokens[END_TOKEN]] = 100.0
    response_ids, ended = generate_response(
        prepared_model,
        example.input_ids,
        prepared_tokenizer.special_tokens,
        max_new_tokens=3,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        device=torch.device("cpu"),
    )
    assert response_ids == [] and ended
    print("All Stage 4 foundation checks passed.")


if __name__ == "__main__":
    main()
