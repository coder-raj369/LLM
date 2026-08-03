"""Quick checks for the character-level Transformer.

Run with: python3 test_tiny.py
"""

import torch

from char_transformer import CharacterTokenizer, CharacterTransformer, make_batch
from tokenization import BPETokenizer


def main():
    text = "hello transformer\n" * 30
    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    model = CharacterTransformer(tokenizer.vocab_size, block_size=16, embed_size=32, heads=4, layers=2)

    inputs, targets = make_batch(data, batch_size=4, block_size=16, device=torch.device("cpu"))
    logits, loss = model(inputs, targets)

    assert logits.shape == (4, 16, tokenizer.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    generated = model.generate(inputs[:1, :1], new_tokens=10)
    assert generated.shape == (1, 11)

    bpe = BPETokenizer.train(text, max_merges=10)
    assert bpe.decode(bpe.encode(text)) == text
    assert bpe.vocab_size > tokenizer.vocab_size
    print("All basic checks passed.")


if __name__ == "__main__":
    main()
