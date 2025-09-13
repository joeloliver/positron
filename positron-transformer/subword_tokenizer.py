"""
Subword Tokenizer using Byte-Pair Encoding (BPE)

This tokenizer is much more effective than character-level tokenization
for language modeling, as it learns meaningful subword units.

Author: Joel Oliver
"""

import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set
import os

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer implementation

    BPE learns to merge the most frequent pairs of characters/subwords,
    creating a vocabulary of meaningful subword units.
    """

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.word_freqs = Counter()
        self.splits = {}
        self.merges = {}
        self.vocab = {}
        self.special_tokens = {
            'pad_token': '<PAD>',
            'unk_token': '<UNK>',
            'bos_token': '<BOS>',
            'eos_token': '<EOS>'
        }

    def _get_word_freqs(self, corpus: List[str]) -> Counter:
        """Count word frequencies in corpus"""
        word_freqs = Counter()
        for text in corpus:
            # Simple preprocessing: lowercase and split on whitespace
            words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
            word_freqs.update(words)
        return word_freqs

    def _get_splits(self, word_freqs: Counter) -> Dict[str, List[str]]:
        """Split words into characters initially"""
        splits = {}
        for word in word_freqs:
            splits[word] = list(word)
        return splits

    def _get_pair_freqs(self, splits: Dict[str, List[str]], word_freqs: Counter) -> Counter:
        """Count frequencies of adjacent pairs"""
        pair_freqs = Counter()
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) < 2:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    def _merge_pair(self, pair: Tuple[str, str], splits: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Merge the most frequent pair in all splits"""
        new_splits = {}
        for word in splits:
            split = splits[word]
            if len(split) < 2:
                new_splits[word] = split
                continue

            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    # Merge the pair
                    new_split.append(split[i] + split[i + 1])
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits

    def train(self, corpus: List[str]) -> None:
        """
        Train the BPE tokenizer on a corpus

        Args:
            corpus: List of text strings to train on
        """
        print(f"Training BPE tokenizer on {len(corpus)} texts...")

        # Get word frequencies
        self.word_freqs = self._get_word_freqs(corpus)
        print(f"Found {len(self.word_freqs)} unique words")

        # Initial splits (character level)
        self.splits = self._get_splits(self.word_freqs)

        # Build vocabulary starting with special tokens
        self.vocab = {}
        for i, (token_type, token) in enumerate(self.special_tokens.items()):
            self.vocab[token] = i

        # Add all characters that appear in the corpus
        all_chars = set()
        for word in self.word_freqs:
            all_chars.update(word)

        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        print(f"Starting with {len(self.vocab)} base tokens")

        # Perform BPE merges
        num_merges = self.vocab_size - len(self.vocab)
        self.merges = {}

        for i in range(num_merges):
            # Find most frequent pair
            pair_freqs = self._get_pair_freqs(self.splits, self.word_freqs)
            if not pair_freqs:
                print(f"No more pairs to merge. Stopping at {len(self.vocab)} tokens.")
                break

            most_frequent_pair = pair_freqs.most_common(1)[0][0]

            # Merge the pair
            self.splits = self._merge_pair(most_frequent_pair, self.splits)
            self.merges[most_frequent_pair] = most_frequent_pair[0] + most_frequent_pair[1]

            # Add merged token to vocabulary
            new_token = most_frequent_pair[0] + most_frequent_pair[1]
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            if (i + 1) % 1000 == 0:
                print(f"Completed {i + 1}/{num_merges} merges...")

        # Create reverse vocabulary
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        print(f"BPE training completed. Final vocabulary size: {self.vocab_size}")

    def _split_word(self, word: str) -> List[str]:
        """Split a word using learned BPE merges"""
        if not self.merges:
            return list(word)

        # Start with character-level split
        split = list(word)

        # Apply merges in the order they were learned
        for pair, merged in self.merges.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == pair[0] and split[i + 1] == pair[1]:
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            split = new_split

        return split

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs

        Args:
            text: Text to encode
            add_special_tokens: Whether to add BOS/EOS tokens

        Returns:
            List of token IDs
        """
        # Tokenize into words
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())

        token_ids = []
        if add_special_tokens:
            token_ids.append(self.vocab[self.special_tokens['bos_token']])

        for word in words:
            # Split word using BPE
            subwords = self._split_word(word)
            for subword in subwords:
                if subword in self.vocab:
                    token_ids.append(self.vocab[subword])
                else:
                    # Handle unknown tokens by splitting into characters
                    for char in subword:
                        if char in self.vocab:
                            token_ids.append(self.vocab[char])
                        else:
                            token_ids.append(self.vocab[self.special_tokens['unk_token']])

        if add_special_tokens:
            token_ids.append(self.vocab[self.special_tokens['eos_token']])

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # Skip special tokens in output
                if token not in self.special_tokens.values():
                    tokens.append(token)

        # Join tokens with spaces for readability
        # This is a simplification - real tokenizers handle whitespace more carefully
        return ' '.join(tokens).replace(' ##', '')

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)

    def save_vocab(self, vocab_path: str) -> None:
        """Save vocabulary to file"""
        vocab_data = {
            'vocab': self.vocab,
            'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }

        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)

        print(f"Vocabulary saved to {vocab_path}")

    def load_vocab(self, vocab_path: str) -> None:
        """Load vocabulary from file"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        self.vocab = vocab_data['vocab']
        self.special_tokens = vocab_data['special_tokens']
        self.vocab_size = vocab_data['vocab_size']

        # Reconstruct merges
        self.merges = {}
        for pair_str, merged in vocab_data['merges'].items():
            pair_parts = pair_str.split('|||')
            if len(pair_parts) == 2:
                self.merges[(pair_parts[0], pair_parts[1])] = merged

        # Create reverse vocabulary
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        print(f"Vocabulary loaded from {vocab_path} (size: {self.vocab_size})")

    @property
    def pad_token_id(self) -> int:
        """Get PAD token ID"""
        return self.vocab[self.special_tokens['pad_token']]

    @property
    def unk_token_id(self) -> int:
        """Get UNK token ID"""
        return self.vocab[self.special_tokens['unk_token']]

    @property
    def bos_token_id(self) -> int:
        """Get BOS token ID"""
        return self.vocab[self.special_tokens['bos_token']]

    @property
    def eos_token_id(self) -> int:
        """Get EOS token ID"""
        return self.vocab[self.special_tokens['eos_token']]