from collections import Counter


class SimpleBPETokenizer:

    def __init__(self):
        # 1. Base vocabulary: Maps basic characters to integers
        self.vocab = {chr(i): i for i in range(256)}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, text, num_merges):
        """Finds frequent patterns in your training data and creates new integer IDs."""
        # Split text into character sequences
        sequences = [list(word) + ["</w>"] for word in text.split()]

        for _ in range(num_merges):
            # Count pairs
            pairs = Counter()
            for seq in sequences:
                for i in range(len(seq) - 1):
                    pairs[(seq[i], seq[i + 1])] += 1

            if not pairs:
                break

            # Find the most common pair (e.g., 't' and 'h')
            best_pair = pairs.most_common(1)[0][0]

            # Assign a new unique integer ID to this pair
            new_token = "".join(best_pair)
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.inverse_vocab[new_id] = new_token

            # Update sequences with the merged pair
            new_sequences = []
            for seq in sequences:
                new_seq = []
                i = 0
                while i < len(seq):
                    if (
                        i < len(seq) - 1
                        and (seq[i], seq[i + 1]) == best_pair
                    ):
                        new_seq.append(new_token)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                sequences = new_sequences

    def encode(self, text):
        """Converts text into a list of integer IDs using your vocabulary."""
        # Simple fallback lookup matching largest possible chunks
        tokens = text.split()
        encoded_ids = []

        for token in tokens:
            token_with_end = token + "</w>"
            # Greedily match against vocab
            if token_with_end in self.vocab:
                encoded_ids.append(self.vocab[token_with_end])
            else:
                # Fallback to individual character IDs if the word is unknown
                for char in token:
                    encoded_ids.append(self.vocab.get(char, 0))
        return encoded_ids

    def decode(self, ids):
        """Converts integer IDs back into readable text."""
        tokens = [self.inverse_vocab.get(i, "") for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()
