"""
Vanilla RNN-based Seq2Seq Model
"""
import torch
import torch.nn as nn
import random


class EncoderRNN(nn.Module):
    """Vanilla RNN Encoder"""
    
    def __init__(self, input_size, embedding_dim, hidden_dim, dropout=0.3):
        super(EncoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_seq, input_lengths):
        """
        Args:
            input_seq: (batch_size, seq_len)
            input_lengths: (batch_size,)
        
        Returns:
            outputs: (batch_size, seq_len, hidden_dim)
            hidden: (1, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(input_seq))
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, hidden = self.rnn(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden


class DecoderRNN(nn.Module):
    """Vanilla RNN Decoder"""
    
    def __init__(self, output_size, embedding_dim, hidden_dim, dropout=0.3):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden):
        """
        Args:
            input_token: (batch_size, 1)
            hidden: (1, batch_size, hidden_dim)
        
        Returns:
            output: (batch_size, output_size)
            hidden: (1, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(input_token))
        
        output, hidden = self.rnn(embedded, hidden)
        
        output = self.out(output.squeeze(1))
        
        return output, hidden


class VanillaSeq2Seq(nn.Module):
    """Vanilla RNN Seq2Seq Model"""
    
    def __init__(self, encoder, decoder, device):
        super(VanillaSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
            tgt: (batch_size, tgt_len)
            teacher_forcing_ratio: probability of using teacher forcing
        
        Returns:
            outputs: (batch_size, tgt_len, output_size)
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encode
        _, hidden = self.encoder(src, src_lengths)
        
        # First input to decoder is SOS token
        input_token = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            output, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = output
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def generate(self, src, src_lengths, max_length, sos_token):
        """
        Generate output sequence
        
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
            max_length: maximum length of generated sequence
            sos_token: start of sequence token
        
        Returns:
            outputs: (batch_size, max_length)
        """
        batch_size = src.shape[0]
        
        # Encode
        _, hidden = self.encoder(src, src_lengths)
        
        # Start with SOS token
        input_token = torch.tensor([[sos_token]] * batch_size).to(self.device)
        
        outputs = []
        
        for _ in range(max_length):
            output, hidden = self.decoder(input_token, hidden)
            top1 = output.argmax(1)
            outputs.append(top1.unsqueeze(1))
            input_token = top1.unsqueeze(1)
        
        return torch.cat(outputs, dim=1)


def create_vanilla_seq2seq(src_vocab_size, tgt_vocab_size, config, device):
    """Create Vanilla RNN Seq2Seq model"""
    encoder = EncoderRNN(
        src_vocab_size,
        config['model']['embedding_dim'],
        config['model']['hidden_dim'],
        config['model']['dropout']
    )
    
    decoder = DecoderRNN(
        tgt_vocab_size,
        config['model']['embedding_dim'],
        config['model']['hidden_dim'],
        config['model']['dropout']
    )
    
    model = VanillaSeq2Seq(encoder, decoder, device).to(device)
    
    return model


if __name__ == '__main__':
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = {
        'model': {
            'embedding_dim': 256,
            'hidden_dim': 256,
            'dropout': 0.3
        }
    }
    
    model = create_vanilla_seq2seq(5000, 5000, config, device)
    
    # Test forward pass
    src = torch.randint(0, 5000, (32, 20)).to(device)
    tgt = torch.randint(0, 5000, (32, 30)).to(device)
    src_lengths = torch.tensor([20] * 32)
    
    outputs = model(src, src_lengths, tgt)
    print("Output shape:", outputs.shape)
    
    # Test generation
    generated = model.generate(src, src_lengths, 30, 1)
    print("Generated shape:", generated.shape)
