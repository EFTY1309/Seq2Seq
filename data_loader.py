"""
Data loading and preprocessing for CodeSearchNet dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import re
from typing import List, Tuple, Dict
import pickle
import os


class Vocabulary:
    """Builds and manages vocabulary for source and target sequences"""
    
    def __init__(self, max_vocab_size=10000):
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        self.max_vocab_size = max_vocab_size
        
        # Special tokens
        self.PAD_token = 0
        self.SOS_token = 1
        self.EOS_token = 2
        self.UNK_token = 3
        
        self.word2idx['<PAD>'] = self.PAD_token
        self.word2idx['<SOS>'] = self.SOS_token
        self.word2idx['<EOS>'] = self.EOS_token
        self.word2idx['<UNK>'] = self.UNK_token
        
        self.idx2word[self.PAD_token] = '<PAD>'
        self.idx2word[self.SOS_token] = '<SOS>'
        self.idx2word[self.EOS_token] = '<EOS>'
        self.idx2word[self.UNK_token] = '<UNK>'
        
        self.n_words = 4
    
    def add_sentence(self, sentence: str):
        """Add all words in a sentence to vocabulary"""
        for word in self.tokenize(sentence):
            self.word_counts[word] += 1
    
    def build_vocab(self):
        """Build vocabulary from word counts, keeping most common words"""
        # Get most common words
        most_common = self.word_counts.most_common(self.max_vocab_size - 4)
        
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = self.n_words
                self.idx2word[self.n_words] = word
                self.n_words += 1
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with some preprocessing"""
        # Basic preprocessing
        text = text.lower().strip()
        # Split on whitespace and basic punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return tokens
    
    def encode(self, sentence: str) -> List[int]:
        """Convert sentence to list of indices"""
        tokens = self.tokenize(sentence)
        return [self.word2idx.get(token, self.UNK_token) for token in tokens]
    
    def decode(self, indices: List[int]) -> str:
        """Convert list of indices back to sentence"""
        words = []
        for idx in indices:
            if idx == self.EOS_token:
                break
            if idx not in [self.PAD_token, self.SOS_token]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def save(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts,
                'max_vocab_size': self.max_vocab_size,
                'n_words': self.n_words
            }, f)
    
    def load(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_counts = data['word_counts']
            self.max_vocab_size = data['max_vocab_size']
            self.n_words = data['n_words']


class CodeSearchNetDataset(Dataset):
    """PyTorch Dataset for CodeSearchNet data"""
    
    def __init__(self, data, src_vocab, tgt_vocab, max_src_len, max_tgt_len):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode source (docstring)
        src_indices = self.src_vocab.encode(item['docstring'])
        # Truncate if too long
        src_indices = src_indices[:self.max_src_len]
        # Add EOS token
        src_indices.append(self.src_vocab.EOS_token)
        
        # Encode target (code)
        tgt_indices = self.tgt_vocab.encode(item['code'])
        # Truncate if too long
        tgt_indices = tgt_indices[:self.max_tgt_len]
        # Add SOS and EOS tokens
        tgt_indices = [self.tgt_vocab.SOS_token] + tgt_indices + [self.tgt_vocab.EOS_token]
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt': torch.tensor(tgt_indices, dtype=torch.long),
            'src_text': item['docstring'],
            'tgt_text': item['code']
        }


def collate_fn(batch):
    """Custom collate function to pad sequences in a batch"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]
    
    # Pad sequences
    src_lengths = torch.tensor([len(s) for s in src_batch])
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch])
    
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': src_lengths,
        'tgt_lengths': tgt_lengths,
        'src_texts': src_texts,
        'tgt_texts': tgt_texts
    }


def load_and_prepare_data(config):
    """
    Load CodeSearchNet dataset and prepare vocabularies
    
    Args:
        config: Configuration dictionary
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)
    """
    print("Loading CodeSearchNet dataset...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset(
        config['dataset']['name'],
        split='train',
        cache_dir=config['dataset']['cache_dir']
    )
    
    print(f"Total dataset size: {len(dataset)}")
    
    # Take larger subset initially (we'll filter and then select what we need)
    total_needed = config['dataset']['train_size'] + config['dataset']['val_size'] + config['dataset']['test_size']
    # Get 5x more than needed to account for filtering
    initial_sample = min(total_needed * 5, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(initial_sample))
    
    # Filter out examples that are too long or empty
    def filter_fn(example):
        try:
            # Handle different possible field names
            doc = example.get('func_documentation_string') or example.get('docstring') or example.get('doc')
            code = example.get('func_code_string') or example.get('code') or example.get('function')
            
            if not doc or not code:
                return False
            
            # Check if strings are not empty after stripping
            doc = str(doc).strip()
            code = str(code).strip()
            
            if not doc or not code:
                return False
            
            doc_len = len(Vocabulary.tokenize(doc))
            code_len = len(Vocabulary.tokenize(code))
            
            return (doc_len > 2 and doc_len <= config['dataset']['max_docstring_length'] and
                    code_len > 2 and code_len <= config['dataset']['max_code_length'])
        except Exception as e:
            return False
    
    print("Filtering dataset...")
    dataset = dataset.filter(filter_fn)
    print(f"After filtering: {len(dataset)} examples")
    
    # Check if we have enough data
    if len(dataset) == 0:
        raise ValueError("No examples passed filtering! The dataset might have different field names or all examples were too long.")
    
    # Convert to simpler format
    processed_data = []
    for item in dataset:
        # Handle different possible field names
        doc = item.get('func_documentation_string') or item.get('docstring') or item.get('doc')
        code = item.get('func_code_string') or item.get('code') or item.get('function')
        
        processed_data.append({
            'docstring': str(doc).strip(),
            'code': str(code).strip()
        })
        
        # Stop if we have enough examples
        if len(processed_data) >= total_needed:
            break
    
    # Split into train/val/test
    train_size = config['dataset']['train_size']
    val_size = config['dataset']['val_size']
    
    train_data = processed_data[:train_size]
    val_data = processed_data[train_size:train_size + val_size]
    test_data = processed_data[train_size + val_size:train_size + val_size + config['dataset']['test_size']]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Check if we have enough data
    if len(train_data) == 0:
        raise ValueError(f"No training data! Got {len(processed_data)} examples after filtering, but needed at least {train_size}. Try reducing the dataset sizes in config.yaml or increasing max_docstring_length/max_code_length.")
    
    if len(train_data) < train_size:
        print(f"⚠️  Warning: Only got {len(train_data)} training examples (requested {train_size})")
        print(f"   Continuing with available data...")
    
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab = Vocabulary(max_vocab_size=config['model']['max_vocab_size'])
    tgt_vocab = Vocabulary(max_vocab_size=config['model']['max_vocab_size'])
    
    # Add all sentences to vocabulary
    for item in train_data:
        src_vocab.add_sentence(item['docstring'])
        tgt_vocab.add_sentence(item['code'])
    
    src_vocab.build_vocab()
    tgt_vocab.build_vocab()
    
    print(f"Source vocabulary size: {src_vocab.n_words}")
    print(f"Target vocabulary size: {tgt_vocab.n_words}")
    
    # Create datasets
    train_dataset = CodeSearchNetDataset(
        train_data, src_vocab, tgt_vocab,
        config['dataset']['max_docstring_length'],
        config['dataset']['max_code_length']
    )
    val_dataset = CodeSearchNetDataset(
        val_data, src_vocab, tgt_vocab,
        config['dataset']['max_docstring_length'],
        config['dataset']['max_code_length']
    )
    test_dataset = CodeSearchNetDataset(
        test_data, src_vocab, tgt_vocab,
        config['dataset']['max_docstring_length'],
        config['dataset']['max_code_length']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Save vocabularies
    os.makedirs(config['paths']['checkpoints'], exist_ok=True)
    src_vocab.save(os.path.join(config['paths']['checkpoints'], 'src_vocab.pkl'))
    tgt_vocab.save(os.path.join(config['paths']['checkpoints'], 'tgt_vocab.pkl'))
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab


if __name__ == '__main__':
    import yaml
    
    # Test data loading
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_and_prepare_data(config)
    
    # Print sample batch
    for batch in train_loader:
        print("Source shape:", batch['src'].shape)
        print("Target shape:", batch['tgt'].shape)
        print("Source text:", batch['src_texts'][0])
        print("Target text:", batch['tgt_texts'][0])
        break
