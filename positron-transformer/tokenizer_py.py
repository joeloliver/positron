"""
Text tokenization for Pure Python Transformer Implementation

This module provides tokenization functionality for converting text to
numerical tokens and vice versa. Implements character-level, word-level,
and simple BPE tokenization strategies.

Key Features:
- Character-level tokenization (simple and effective for learning)
- Word-level tokenization with vocabulary management
- Basic BPE (Byte Pair Encoding) implementation
- Special token handling (PAD, UNK, BOS, EOS)
- Efficient encoding/decoding with NumPy arrays

Author: Joel Oliver
Based on: Experience with data preprocessing and optimization
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
import re
import json
import os
from collections import Counter, defaultdict
from abc import ABC, abstractmethod

# =============================================================================
# ABSTRACT BASE TOKENIZER
# =============================================================================

class BaseTokenizer(ABC):
    """
    Abstract base class for all tokenizers.
    
    Defines the interface that all tokenizers must implement. This design
    allows for easy swapping between different tokenization strategies.
    """
    
    def __init__(self, vocab_size: int = 8000, 
                 pad_token: str = '<PAD>',
                 unk_token: str = '<UNK>',
                 bos_token: str = '<BOS>',
                 eos_token: str = '<EOS>'):
        """
        Initialize base tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            pad_token: Padding token
            unk_token: Unknown token
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
        """
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        # Special token IDs (will be set during vocab creation)
        self.pad_token_id: Optional[int] = None
        self.unk_token_id: Optional[int] = None
        self.bos_token_id: Optional[int] = None
        self.eos_token_id: Optional[int] = None
        
        # Vocabulary mappings
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        
        # Training state
        self.is_trained = False
    
    @abstractmethod
    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
        """
        pass
    
    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            Array of token IDs
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: Union[np.ndarray, List[int]], 
               skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Array or list of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        pass
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_id)
    
    def save_vocab(self, vocab_path: str) -> None:
        """
        Save vocabulary to file.
        
        Args:
            vocab_path: Path to save vocabulary JSON file
        """
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': {str(k): v for k, v in self.id_to_token.items()},
            'special_tokens': {
                'pad_token': self.pad_token,
                'unk_token': self.unk_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token,
                'pad_token_id': self.pad_token_id,
                'unk_token_id': self.unk_token_id,
                'bos_token_id': self.bos_token_id,
                'eos_token_id': self.eos_token_id,
            },
            'vocab_size': self.vocab_size,
            'tokenizer_type': self.__class__.__name__
        }
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    def load_vocab(self, vocab_path: str) -> None:
        """
        Load vocabulary from file.
        
        Args:
            vocab_path: Path to vocabulary JSON file
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        
        special_tokens = vocab_data['special_tokens']
        self.pad_token = special_tokens['pad_token']
        self.unk_token = special_tokens['unk_token']
        self.bos_token = special_tokens['bos_token']
        self.eos_token = special_tokens['eos_token']
        self.pad_token_id = special_tokens['pad_token_id']
        self.unk_token_id = special_tokens['unk_token_id']
        self.bos_token_id = special_tokens['bos_token_id']
        self.eos_token_id = special_tokens['eos_token_id']
        
        self.vocab_size = vocab_data['vocab_size']
        self.is_trained = True

# =============================================================================
# CHARACTER-LEVEL TOKENIZER
# =============================================================================

class CharacterTokenizer(BaseTokenizer):
    """
    Character-level tokenizer.
    
    This tokenizer treats each character as a separate token. It's simple
    and effective for learning, especially with smaller datasets. Good for
    understanding transformer mechanics without complex preprocessing.
    
    Advantages:
    - Simple implementation
    - No out-of-vocabulary issues (except rare Unicode characters)
    - Works well for any language
    - Small vocabulary size
    
    Disadvantages:
    - Longer sequences
    - May not capture semantic meaning as well as word-level
    """
    
    def __init__(self, vocab_size: int = 1000, **kwargs):
        """
        Initialize character tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            **kwargs: Arguments passed to base tokenizer
        """
        super().__init__(vocab_size=vocab_size, **kwargs)
        
        # Character frequency counter
        self.char_counts = Counter()
    
    def train(self, texts: List[str]) -> None:
        """
        Train character tokenizer on texts.
        
        Builds vocabulary from most frequent characters in training corpus.
        
        Args:
            texts: List of training texts
        """
        print(f"Training character tokenizer on {len(texts)} texts...")
        
        # Count character frequencies
        for text in texts:
            self.char_counts.update(text)
        
        print(f"Found {len(self.char_counts)} unique characters")
        
        # Create vocabulary starting with special tokens
        vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Add most frequent characters (reserve space for special tokens)
        most_frequent_chars = self.char_counts.most_common(self.vocab_size - len(vocab))
        vocab.extend([char for char, _ in most_frequent_chars])
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}
        
        # Set special token IDs
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        
        self.is_trained = True
        
        print(f"Vocabulary size: {len(self.token_to_id)}")
        print(f"Special token IDs: PAD={self.pad_token_id}, UNK={self.unk_token_id}, "
              f"BOS={self.bos_token_id}, EOS={self.eos_token_id}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        """
        Encode text to character token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            Array of token IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        # Convert characters to IDs
        token_ids = []
        
        # Add BOS token
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        # Convert each character
        for char in text:
            if char in self.token_to_id:
                token_ids.append(self.token_to_id[char])
            else:
                token_ids.append(self.unk_token_id)
        
        # Add EOS token
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return np.array(token_ids, dtype=np.int32)
    
    def decode(self, token_ids: Union[np.ndarray, List[int]], 
               skip_special_tokens: bool = True) -> str:
        """
        Decode character token IDs to text.
        
        Args:
            token_ids: Array or list of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        # Convert to list if numpy array
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Special token IDs to skip
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        if not skip_special_tokens:
            special_ids = set()
        
        # Convert IDs to characters
        chars = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            if token_id in self.id_to_token:
                chars.append(self.id_to_token[token_id])
            else:
                chars.append(self.unk_token)  # Handle invalid IDs
        
        return ''.join(chars)

# =============================================================================
# WORD-LEVEL TOKENIZER
# =============================================================================

class WordTokenizer(BaseTokenizer):
    """
    Word-level tokenizer with vocabulary management.
    
    This tokenizer splits text into words and builds a vocabulary based on
    word frequency. More semantic than character-level but may have more
    out-of-vocabulary issues.
    
    Advantages:
    - Captures word-level semantics
    - Shorter sequences than character-level
    - Good for well-defined vocabularies
    
    Disadvantages:
    - Larger vocabulary requirements
    - Out-of-vocabulary issues
    - Language-specific word boundaries
    """
    
    def __init__(self, vocab_size: int = 8000, min_freq: int = 2, **kwargs):
        """
        Initialize word tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for word inclusion
            **kwargs: Arguments passed to base tokenizer
        """
        super().__init__(vocab_size=vocab_size, **kwargs)
        self.min_freq = min_freq
        self.word_counts = Counter()
        
        # Regex for basic word tokenization (can be customized)
        self.word_pattern = re.compile(r'\b\w+\b|[^\w\s]')
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        # Convert to lowercase and find all words/punctuation
        text = text.lower()
        tokens = self.word_pattern.findall(text)
        return tokens
    
    def train(self, texts: List[str]) -> None:
        """
        Train word tokenizer on texts.
        
        Args:
            texts: List of training texts
        """
        print(f"Training word tokenizer on {len(texts)} texts...")
        
        # Count word frequencies
        for text in texts:
            words = self._tokenize_text(text)
            self.word_counts.update(words)
        
        print(f"Found {len(self.word_counts)} unique words")
        
        # Filter words by minimum frequency
        filtered_words = {word: count for word, count in self.word_counts.items() 
                         if count >= self.min_freq}
        
        print(f"After filtering (min_freq={self.min_freq}): {len(filtered_words)} words")
        
        # Create vocabulary starting with special tokens
        vocab = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Add most frequent words (reserve space for special tokens)
        most_frequent_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        vocab.extend([word for word, _ in most_frequent_words[:self.vocab_size - len(vocab)]])
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab)}
        
        # Set special token IDs
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        
        self.is_trained = True
        
        print(f"Final vocabulary size: {len(self.token_to_id)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        """
        Encode text to word token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            Array of token IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        # Tokenize text into words
        words = self._tokenize_text(text)
        
        # Convert words to IDs
        token_ids = []
        
        # Add BOS token
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        # Convert each word
        for word in words:
            if word in self.token_to_id:
                token_ids.append(self.token_to_id[word])
            else:
                token_ids.append(self.unk_token_id)
        
        # Add EOS token
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return np.array(token_ids, dtype=np.int32)
    
    def decode(self, token_ids: Union[np.ndarray, List[int]], 
               skip_special_tokens: bool = True) -> str:
        """
        Decode word token IDs to text.
        
        Args:
            token_ids: Array or list of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        # Convert to list if numpy array
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Special token IDs to skip
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        if not skip_special_tokens:
            special_ids = set()
        
        # Convert IDs to words
        words = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            if token_id in self.id_to_token:
                words.append(self.id_to_token[token_id])
            else:
                words.append(self.unk_token)
        
        # Join words with spaces (simple reconstruction)
        return ' '.join(words)

# =============================================================================
# SIMPLE BPE TOKENIZER
# =============================================================================

class SimpleBPETokenizer(BaseTokenizer):
    """
    Simple Byte Pair Encoding (BPE) tokenizer.
    
    BPE iteratively merges the most frequent pair of characters/tokens to
    create a vocabulary that balances between character-level and word-level
    tokenization.
    
    This is a simplified version of BPE that demonstrates the core concept
    without the full complexity of modern implementations like GPT-2's BPE.
    
    Advantages:
    - Good balance between vocabulary size and sequence length
    - Handles out-of-vocabulary words better than word-level
    - Subword information preservation
    
    Disadvantages:
    - More complex than character/word level
    - Training can be slower
    - May create semantically meaningless subwords
    """
    
    def __init__(self, vocab_size: int = 8000, num_merges: int = 1000, **kwargs):
        """
        Initialize simple BPE tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            num_merges: Number of BPE merge operations
            **kwargs: Arguments passed to base tokenizer
        """
        super().__init__(vocab_size=vocab_size, **kwargs)
        self.num_merges = min(num_merges, vocab_size - 4)  # Reserve space for special tokens
        
        # BPE merge rules (pair -> merged_token)
        self.bpe_merges = []
        self.merge_rules = {}
        
    def _get_word_tokens(self, word: str) -> List[str]:
        """
        Split word into character tokens with end-of-word marker.
        
        Args:
            word: Input word
            
        Returns:
            List of character tokens
        """
        if not word:
            return []
        
        # Split into characters and add end-of-word marker to last character
        chars = list(word)
        chars[-1] = chars[-1] + '</w>'
        return chars
    
    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """
        Get all adjacent pairs in word tokens.
        
        Args:
            word_tokens: List of token strings
            
        Returns:
            Set of adjacent pairs
        """
        pairs = set()
        prev_char = word_tokens[0]
        
        for char in word_tokens[1:]:
            pairs.add((prev_char, char))
            prev_char = char
            
        return pairs
    
    def _merge_pair(self, word_tokens: List[str], pair: Tuple[str, str]) -> List[str]:
        """
        Merge a specific pair in word tokens.
        
        Args:
            word_tokens: List of token strings
            pair: Pair to merge
            
        Returns:
            List with merged tokens
        """
        new_tokens = []
        i = 0
        
        while i < len(word_tokens):
            try:
                j = word_tokens.index(pair[0], i)
                new_tokens.extend(word_tokens[i:j])
                i = j
            except ValueError:
                new_tokens.extend(word_tokens[i:])
                break
            
            if i < len(word_tokens) - 1 and word_tokens[i + 1] == pair[1]:
                # Merge the pair
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        
        return new_tokens
    
    def train(self, texts: List[str]) -> None:
        """
        Train BPE tokenizer on texts.
        
        Args:
            texts: List of training texts
        """
        print(f"Training BPE tokenizer on {len(texts)} texts...")
        
        # Create initial word vocabulary with character-level tokens
        word_freqs = Counter()
        
        # Count word frequencies and split into characters
        for text in texts:
            # Simple word tokenization (lowercase and split on whitespace/punctuation)
            words = re.findall(r'\b\w+\b', text.lower())
            word_freqs.update(words)
        
        # Convert words to character tokens
        vocab = defaultdict(int)
        for word, freq in word_freqs.items():
            word_tokens = self._get_word_tokens(word)
            vocab[' '.join(word_tokens)] += freq
        
        # Get initial character vocabulary
        all_chars = set()
        for word_tokens in vocab.keys():
            all_chars.update(word_tokens.split())
        
        print(f"Initial character vocabulary size: {len(all_chars)}")
        
        # Perform BPE merges
        for merge_idx in range(self.num_merges):
            # Count all pairs
            pairs = defaultdict(int)
            for word_tokens, freq in vocab.items():
                word_token_list = word_tokens.split()
                word_pairs = self._get_pairs(word_token_list)
                
                for pair in word_pairs:
                    pairs[pair] += freq
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the best pair in all words
            new_vocab = {}
            for word_tokens, freq in vocab.items():
                word_token_list = word_tokens.split()
                merged_tokens = self._merge_pair(word_token_list, best_pair)
                new_vocab[' '.join(merged_tokens)] = freq
            
            vocab = new_vocab
            
            # Store merge rule
            self.bpe_merges.append(best_pair)
            self.merge_rules[best_pair] = best_pair[0] + best_pair[1]
            
            if (merge_idx + 1) % 100 == 0:
                print(f"Completed {merge_idx + 1} merges")
        
        # Create final vocabulary
        final_vocab = set()
        for word_tokens in vocab.keys():
            final_vocab.update(word_tokens.split())
        
        # Build token mappings
        vocab_list = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        vocab_list.extend(sorted(final_vocab))
        
        self.token_to_id = {token: idx for idx, token in enumerate(vocab_list)}
        self.id_to_token = {idx: token for idx, token in enumerate(vocab_list)}
        
        # Set special token IDs
        self.pad_token_id = self.token_to_id[self.pad_token]
        self.unk_token_id = self.token_to_id[self.unk_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        
        self.is_trained = True
        
        print(f"BPE training completed:")
        print(f"  - Number of merges: {len(self.bpe_merges)}")
        print(f"  - Final vocabulary size: {len(self.token_to_id)}")
    
    def _apply_bpe(self, word: str) -> List[str]:
        """
        Apply BPE merges to a word.
        
        Args:
            word: Input word
            
        Returns:
            List of BPE tokens
        """
        if not word:
            return []
        
        # Start with character-level tokens
        word_tokens = self._get_word_tokens(word)
        
        # Apply each merge rule in order
        for pair in self.bpe_merges:
            if len(word_tokens) == 1:
                break
            
            word_tokens = self._merge_pair(word_tokens, pair)
        
        return word_tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        """
        Encode text using BPE tokenization.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            
        Returns:
            Array of token IDs
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before encoding")
        
        # Extract words from text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Apply BPE to each word
        all_tokens = []
        
        # Add BOS token
        if add_special_tokens:
            all_tokens.append(self.bos_token_id)
        
        for word in words:
            bpe_tokens = self._apply_bpe(word)
            
            for token in bpe_tokens:
                if token in self.token_to_id:
                    all_tokens.append(self.token_to_id[token])
                else:
                    all_tokens.append(self.unk_token_id)
        
        # Add EOS token
        if add_special_tokens:
            all_tokens.append(self.eos_token_id)
        
        return np.array(all_tokens, dtype=np.int32)
    
    def decode(self, token_ids: Union[np.ndarray, List[int]], 
               skip_special_tokens: bool = True) -> str:
        """
        Decode BPE token IDs to text.
        
        Args:
            token_ids: Array or list of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text
        """
        if not self.is_trained:
            raise ValueError("Tokenizer must be trained before decoding")
        
        # Convert to list if numpy array
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        # Special token IDs to skip
        special_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        if not skip_special_tokens:
            special_ids = set()
        
        # Convert IDs to tokens
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.unk_token)
        
        # Reconstruct text from BPE tokens
        text = ''.join(tokens)
        
        # Replace end-of-word markers with spaces
        text = text.replace('</w>', ' ')
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text

# =============================================================================
# TOKENIZER FACTORY AND UTILITIES
# =============================================================================

def create_tokenizer(tokenizer_type: str, **kwargs) -> BaseTokenizer:
    """
    Factory function to create tokenizer instances.
    
    Args:
        tokenizer_type: Type of tokenizer ('character', 'word', 'bpe')
        **kwargs: Arguments passed to tokenizer constructor
        
    Returns:
        Tokenizer instance
        
    Raises:
        ValueError: If tokenizer type is not supported
    """
    tokenizers = {
        'character': CharacterTokenizer,
        'word': WordTokenizer,
        'bpe': SimpleBPETokenizer,
    }
    
    if tokenizer_type not in tokenizers:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}. "
                        f"Available: {list(tokenizers.keys())}")
    
    return tokenizers[tokenizer_type](**kwargs)

def load_tokenizer(vocab_path: str) -> BaseTokenizer:
    """
    Load a trained tokenizer from vocabulary file.
    
    Args:
        vocab_path: Path to vocabulary JSON file
        
    Returns:
        Loaded tokenizer instance
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    tokenizer_type = vocab_data['tokenizer_type'].lower()
    
    if tokenizer_type == 'charactertokenizer':
        tokenizer = CharacterTokenizer()
    elif tokenizer_type == 'wordtokenizer':
        tokenizer = WordTokenizer()
    elif tokenizer_type == 'simplebpetokenizer':
        tokenizer = SimpleBPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type in vocab file: {tokenizer_type}")
    
    tokenizer.load_vocab(vocab_path)
    return tokenizer

def batch_encode(tokenizer: BaseTokenizer, texts: List[str], 
                max_length: Optional[int] = None, 
                padding: bool = True,
                truncation: bool = True,
                add_special_tokens: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batch encode multiple texts with padding and truncation.
    
    Args:
        tokenizer: Trained tokenizer instance
        texts: List of texts to encode
        max_length: Maximum sequence length (None for auto)
        padding: Whether to pad sequences
        truncation: Whether to truncate long sequences
        add_special_tokens: Whether to add BOS/EOS tokens
        
    Returns:
        Tuple of (token_ids, attention_mask)
        token_ids: Array of shape (batch_size, seq_len)
        attention_mask: Array of shape (batch_size, seq_len) with 1 for real tokens, 0 for padding
    """
    # Encode all texts
    encoded_texts = [tokenizer.encode(text, add_special_tokens=add_special_tokens) 
                    for text in texts]
    
    # Determine sequence length
    if max_length is None:
        max_length = max(len(tokens) for tokens in encoded_texts)
    
    # Initialize output arrays
    batch_size = len(texts)
    token_ids = np.full((batch_size, max_length), tokenizer.pad_token_id, dtype=np.int32)
    attention_mask = np.zeros((batch_size, max_length), dtype=np.float32)
    
    # Fill arrays
    for i, tokens in enumerate(encoded_texts):
        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Copy tokens and set attention mask
        seq_len = len(tokens)
        token_ids[i, :seq_len] = tokens
        attention_mask[i, :seq_len] = 1.0
    
    return token_ids, attention_mask

def print_tokenizer_stats(tokenizer: BaseTokenizer, sample_text: str = None) -> None:
    """
    Print statistics about a trained tokenizer.
    
    Args:
        tokenizer: Trained tokenizer instance
        sample_text: Optional sample text to analyze
    """
    if not tokenizer.is_trained:
        print("Tokenizer is not trained yet.")
        return
    
    print(f"\n=== {tokenizer.__class__.__name__} Statistics ===")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Special tokens:")
    print(f"  PAD: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  UNK: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    print(f"  BOS: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"  EOS: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    if sample_text:
        print(f"\nSample encoding:")
        print(f"Original text: '{sample_text}'")
        encoded = tokenizer.encode(sample_text)
        print(f"Encoded: {encoded}")
        print(f"Length: {len(encoded)} tokens")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: '{decoded}'")
        print(f"Round-trip successful: {sample_text.lower().strip() == decoded.lower().strip()}")

# =============================================================================
# SIMPLE TOKENIZER ALIAS (DEFAULT)
# =============================================================================

# Create alias for the default/recommended tokenizer
SimpleTokenizer = CharacterTokenizer

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage of different tokenizers
    
    # Sample training texts
    sample_texts = [
        "Once upon a time, there was a little cat named Whiskers.",
        "The cat loved to play in the sunny garden every day.",
        "Whiskers found a ball of yarn and played happily.",
        "The end of the story came when night fell.",
    ]
    
    sample_test_text = "The little cat played with the yarn ball in the garden."
    
    print("=" * 60)
    print("TOKENIZER COMPARISON")
    print("=" * 60)
    
    # Test Character Tokenizer
    print("\n1. CHARACTER TOKENIZER")
    print("-" * 30)
    char_tokenizer = CharacterTokenizer(vocab_size=200)
    char_tokenizer.train(sample_texts)
    print_tokenizer_stats(char_tokenizer, sample_test_text)
    
    # Test Word Tokenizer
    print("\n2. WORD TOKENIZER")
    print("-" * 30)
    word_tokenizer = WordTokenizer(vocab_size=100, min_freq=1)
    word_tokenizer.train(sample_texts)
    print_tokenizer_stats(word_tokenizer, sample_test_text)
    
    # Test BPE Tokenizer
    print("\n3. BPE TOKENIZER")
    print("-" * 30)
    bpe_tokenizer = SimpleBPETokenizer(vocab_size=150, num_merges=50)
    bpe_tokenizer.train(sample_texts)
    print_tokenizer_stats(bpe_tokenizer, sample_test_text)
    
    # Test batch encoding
    print("\n4. BATCH ENCODING TEST")
    print("-" * 30)
    test_texts = [
        "Short text.",
        "This is a longer text with more words.",
        "Medium length text here."
    ]
    
    token_ids, attention_mask = batch_encode(
        char_tokenizer, test_texts, 
        max_length=50, padding=True, truncation=True
    )
    
    print(f"Batch token IDs shape: {token_ids.shape}")
    print(f"Attention mask shape: {attention_mask.shape}")
    print("Sample batch (first text):")
    print(f"  Tokens: {token_ids[0]}")
    print(f"  Mask: {attention_mask[0]}")
    print(f"  Decoded: '{char_tokenizer.decode(token_ids[0])}'")
    
    print("\n" + "=" * 60)
    print("All tokenizer tests completed successfully!")
    print("=" * 60)

# Convenience alias for simple usage
SimpleTokenizer = CharacterTokenizer