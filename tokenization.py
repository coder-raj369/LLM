"""Small tokenizers used by the learning Transformer project."""

from collections import Counter
from collections.abc import Iterable


class BPETokenizer:
    """Character BPE with optional reserved tokens outside the merge vocabulary."""

    def __init__(
        self,
        token_to_id: dict[str, int],
        merges: list[tuple[str, str]],
        special_tokens: dict[str, int] | None = None,
    ):
        self.token_to_id = dict(token_to_id)
        self.id_to_token = {token_id: token for token, token_id in token_to_id.items()}
        self.merges = merges
        self.special_tokens = dict(special_tokens or {})

        if len(self.id_to_token) != len(self.token_to_id):
            raise ValueError("Every tokenizer token must have a unique ID.")
        for token, token_id in self.special_tokens.items():
            if self.token_to_id.get(token) != token_id:
                raise ValueError(f"Special token {token!r} is missing or has the wrong ID.")

        # Longest match first makes the behaviour unambiguous if token names overlap.
        self._special_token_strings = tuple(sorted(self.special_tokens, key=len, reverse=True))

    @classmethod
    def train(cls, text: str, max_merges: int = 100):
        if not text:
            raise ValueError("Cannot train a tokenizer on an empty file.")

        symbols = list(text)
        token_to_id = {character: index for index, character in enumerate(sorted(set(symbols)))}
        merges = []

        for _ in range(max_merges):
            pair_counts = Counter(zip(symbols, symbols[1:]))
            if not pair_counts:
                break

            pair, count = pair_counts.most_common(1)[0]
            merged_token = pair[0] + pair[1]
            if count < 2 or merged_token in token_to_id:
                break

            token_to_id[merged_token] = len(token_to_id)
            merges.append(pair)

            merged_symbols = []
            index = 0
            while index < len(symbols):
                if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
                    merged_symbols.append(merged_token)
                    index += 2
                else:
                    merged_symbols.append(symbols[index])
                    index += 1
            symbols = merged_symbols

        return cls(token_to_id, merges)

    def add_special_tokens(self, tokens: Iterable[str]) -> dict[str, int]:
        """Reserve IDs for literal markers that must never be BPE-split.

        This mutates the tokenizer so that existing token IDs stay unchanged and
        new IDs are appended to the vocabulary. It is intended for markers such
        as ``<|user|>`` and ``<|assistant|>``.
        """
        added = {}
        for token in tokens:
            if not token:
                raise ValueError("A special token cannot be empty.")
            if token in self.special_tokens:
                added[token] = self.special_tokens[token]
                continue
            if token in self.token_to_id:
                raise ValueError(f"{token!r} already exists as a normal BPE token.")

            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            self.special_tokens[token] = token_id
            added[token] = token_id

        self._special_token_strings = tuple(sorted(self.special_tokens, key=len, reverse=True))
        return added

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def characters(self) -> set[str]:
        """The original character alphabet, useful for checking chat prompts."""
        return {token for token in self.token_to_id if len(token) == 1}

    def _encode_plain_text(self, text: str) -> list[int]:
        """Encode ordinary text only; special markers are handled separately."""
        unknown = set(text) - self.characters
        if unknown:
            raise ValueError(f"Characters not in the tokenizer vocabulary: {sorted(unknown)}")

        symbols = list(text)
        for pair in self.merges:
            merged_token = pair[0] + pair[1]
            merged_symbols = []
            index = 0
            while index < len(symbols):
                if index < len(symbols) - 1 and (symbols[index], symbols[index + 1]) == pair:
                    merged_symbols.append(merged_token)
                    index += 2
                else:
                    merged_symbols.append(symbols[index])
                    index += 1
            symbols = merged_symbols
        return [self.token_to_id[symbol] for symbol in symbols]

    def encode(self, text: str) -> list[int]:
        """Encode text while preserving registered special tokens as single IDs."""
        if not self.special_tokens:
            return self._encode_plain_text(text)

        encoded_ids = []
        text_start = 0
        index = 0
        while index < len(text):
            special_token = next(
                (token for token in self._special_token_strings if text.startswith(token, index)),
                None,
            )
            if special_token is None:
                index += 1
                continue

            encoded_ids.extend(self._encode_plain_text(text[text_start:index]))
            encoded_ids.append(self.special_tokens[special_token])
            index += len(special_token)
            text_start = index

        encoded_ids.extend(self._encode_plain_text(text[text_start:]))
        return encoded_ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[token_id] for token_id in ids)

    def state_dict(self) -> dict:
        return {
            "kind": "bpe",
            "token_to_id": self.token_to_id,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_state_dict(cls, state: dict):
        return cls(
            state["token_to_id"],
            [tuple(pair) for pair in state["merges"]],
            state.get("special_tokens"),
        )
