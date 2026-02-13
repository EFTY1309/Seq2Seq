"""
LSTM with Bahdanau Attention Seq2Seq Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class EncoderBiLSTM(nn.Module):
    """Bidirectional LSTM Encoder"""
    
    def __init__(self, input_size, embedding_dim, hidden_dim, dropout=0.3):
        super(EncoderBiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(input_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        
        # Linear layer to project bidirectional hidden state to decoder hidden size
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, input_seq, input_lengths):
        """
        Args:
            input_seq: (batch_size, seq_len)
            input_lengths: (batch_size,)
        
        Returns:
            outputs: (batch_size, seq_len, hidden_dim * 2)
            hidden: (1, batch_size, hidden_dim)
            cell: (1, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(input_seq))
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, (hidden, cell) = self.lstm(packed)
        
        # Unpack
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # hidden and cell are (2, batch_size, hidden_dim) for bidirectional
        # Concatenate forward and backward and project to decoder size
        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[0], hidden[1]), dim=1))).unsqueeze(0)
        cell = torch.tanh(self.fc_cell(torch.cat((cell[0], cell[1]), dim=1))).unsqueeze(0)
        
        return outputs, hidden, cell


class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention Mechanism"""
    
    def __init__(self, hidden_dim, encoder_dim):
        super(BahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        
        # Attention layers
        self.attn_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.attn_encoder = nn.Linear(encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Args:
            hidden: (batch_size, hidden_dim) - decoder hidden state
            encoder_outputs: (batch_size, src_len, encoder_dim) - all encoder outputs
            mask: (batch_size, src_len) - mask for padding
        
        Returns:
            context: (batch_size, encoder_dim) - weighted context vector
            attention_weights: (batch_size, src_len) - attention weights
        """
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # (batch_size, src_len, hidden_dim)
        
        # Calculate attention energy
        energy = torch.tanh(
            self.attn_hidden(hidden) + self.attn_encoder(encoder_outputs)
        )  # (batch_size, src_len, hidden_dim)
        
        # Calculate attention scores
        attention = self.v(energy).squeeze(2)  # (batch_size, src_len)
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention, dim=1)  # (batch_size, src_len)
        
        # Calculate context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # (batch_size, encoder_dim)
        
        return context, attention_weights


class AttentionDecoderLSTM(nn.Module):
    """LSTM Decoder with Attention"""
    
    def __init__(self, output_size, embedding_dim, hidden_dim, encoder_dim, dropout=0.3):
        super(AttentionDecoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.encoder_dim = encoder_dim
        
        self.embedding = nn.Embedding(output_size, embedding_dim, padding_idx=0)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        
        # LSTM input is embedding + context vector
        self.lstm = nn.LSTM(embedding_dim + encoder_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.out = nn.Linear(hidden_dim + encoder_dim + embedding_dim, output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        """
        Args:
            input_token: (batch_size, 1)
            hidden: (1, batch_size, hidden_dim)
            cell: (1, batch_size, hidden_dim)
            encoder_outputs: (batch_size, src_len, encoder_dim)
            mask: (batch_size, src_len)
        
        Returns:
            output: (batch_size, output_size)
            hidden: (1, batch_size, hidden_dim)
            cell: (1, batch_size, hidden_dim)
            attention_weights: (batch_size, src_len)
        """
        embedded = self.dropout(self.embedding(input_token))  # (batch_size, 1, embedding_dim)
        
        # Calculate attention
        context, attention_weights = self.attention(
            hidden.squeeze(0),
            encoder_outputs,
            mask
        )  # context: (batch_size, encoder_dim)
        
        # Concatenate embedding and context
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # (batch_size, 1, embedding_dim + encoder_dim)
        
        # LSTM forward
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # Concatenate output, context, and embedding for final prediction
        output = output.squeeze(1)  # (batch_size, hidden_dim)
        embedded = embedded.squeeze(1)  # (batch_size, embedding_dim)
        
        pred_input = torch.cat((output, context, embedded), dim=1)  # (batch_size, hidden_dim + encoder_dim + embedding_dim)
        prediction = self.out(pred_input)  # (batch_size, output_size)
        
        return prediction, hidden, cell, attention_weights


class AttentionSeq2Seq(nn.Module):
    """LSTM with Attention Seq2Seq Model"""
    
    def __init__(self, encoder, decoder, device):
        super(AttentionSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def create_mask(self, src, src_lengths):
        """Create mask for padding"""
        mask = torch.zeros_like(src, dtype=torch.bool)
        for i, length in enumerate(src_lengths):
            mask[i, :length] = 1
        return mask
    
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len)
            src_lengths: (batch_size,)
            tgt: (batch_size, tgt_len)
            teacher_forcing_ratio: probability of using teacher forcing
        
        Returns:
            outputs: (batch_size, tgt_len, output_size)
            attentions: (batch_size, tgt_len, src_len)
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_size
        src_len = src.shape[1]
        
        # Tensors to store decoder outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, tgt_len, src_len).to(self.device)
        
        # Create mask
        mask = self.create_mask(src, src_lengths).to(self.device)
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # First input to decoder is SOS token
        input_token = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, tgt_len):
            output, hidden, cell, attention_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            outputs[:, t] = output
            attentions[:, t] = attention_weights
            
            # Teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs, attentions
    
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
            attentions: (batch_size, max_length, src_len)
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Create mask
        mask = self.create_mask(src, src_lengths).to(self.device)
        
        # Encode
        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # Start with SOS token
        input_token = torch.tensor([[sos_token]] * batch_size).to(self.device)
        
        outputs = []
        attentions = torch.zeros(batch_size, max_length, src_len).to(self.device)
        
        for t in range(max_length):
            output, hidden, cell, attention_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )
            attentions[:, t] = attention_weights
            top1 = output.argmax(1)
            outputs.append(top1.unsqueeze(1))
            input_token = top1.unsqueeze(1)
        
        return torch.cat(outputs, dim=1), attentions


def create_attention_seq2seq(src_vocab_size, tgt_vocab_size, config, device):
    """Create LSTM with Attention Seq2Seq model"""
    encoder = EncoderBiLSTM(
        src_vocab_size,
        config['model']['embedding_dim'],
        config['model']['hidden_dim'],
        config['model']['dropout']
    )
    
    decoder = AttentionDecoderLSTM(
        tgt_vocab_size,
        config['model']['embedding_dim'],
        config['model']['hidden_dim'],
        config['model']['hidden_dim'] * 2,  # bidirectional encoder
        config['model']['dropout']
    )
    
    model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    
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
    
    model = create_attention_seq2seq(5000, 5000, config, device)
    
    # Test forward pass
    src = torch.randint(0, 5000, (32, 20)).to(device)
    tgt = torch.randint(0, 5000, (32, 30)).to(device)
    src_lengths = torch.tensor([20] * 32)
    
    outputs, attentions = model(src, src_lengths, tgt)
    print("Output shape:", outputs.shape)
    print("Attention shape:", attentions.shape)
    
    # Test generation
    generated, gen_attentions = model.generate(src, src_lengths, 30, 1)
    print("Generated shape:", generated.shape)
    print("Generated attention shape:", gen_attentions.shape)
