import pickle

# Load source vocab
with open('./checkpoints/src_vocab.pkl', 'rb') as f:
    src_data = pickle.load(f)
    print(f'Source vocab: n_words={src_data["n_words"]}, max_size={src_data["max_vocab_size"]}')

# Load target vocab
with open('./checkpoints/tgt_vocab.pkl', 'rb') as f:
    tgt_data = pickle.load(f)
    print(f'Target vocab: n_words={tgt_data["n_words"]}, max_size={tgt_data["max_vocab_size"]}')
