"""
Training script for Seq2Seq models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
import os
import argparse
import json
import numpy as np

from data_loader import load_and_prepare_data, Vocabulary
from models import create_vanilla_seq2seq, create_lstm_seq2seq, create_attention_seq2seq


class Trainer:
    """Trainer class for Seq2Seq models"""
    
    def __init__(self, model, config, device, model_name, src_vocab, tgt_vocab):
        self.model = model
        self.config = config
        self.device = device
        self.model_name = model_name
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Create directories
        self.checkpoint_dir = os.path.join(config['paths']['checkpoints'], model_name)
        self.log_dir = os.path.join(config['paths']['logs'], model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            src_lengths = batch['src_lengths']
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if 'attention' in self.model_name:
                outputs, _ = self.model(
                    src, src_lengths, tgt,
                    teacher_forcing_ratio=self.config['training']['teacher_forcing_ratio']
                )
            else:
                outputs = self.model(
                    src, src_lengths, tgt,
                    teacher_forcing_ratio=self.config['training']['teacher_forcing_ratio']
                )
            
            # Calculate loss
            # Reshape: outputs (batch_size, tgt_len, vocab_size) -> (batch_size * tgt_len, vocab_size)
            # tgt (batch_size, tgt_len) -> (batch_size * tgt_len)
            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            loss = self.criterion(outputs, tgt)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return epoch_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                src_lengths = batch['src_lengths']
                
                # Forward pass
                if 'attention' in self.model_name:
                    outputs, _ = self.model(src, src_lengths, tgt, teacher_forcing_ratio=0)
                else:
                    outputs = self.model(src, src_lengths, tgt, teacher_forcing_ratio=0)
                
                # Calculate loss
                output_dim = outputs.shape[-1]
                outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
                tgt = tgt[:, 1:].contiguous().view(-1)
                
                loss = self.criterion(outputs, tgt)
                epoch_loss += loss.item()
        
        return epoch_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs):
        """Full training loop"""
        print(f"\nTraining {self.model_name}...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss:   {val_loss:.4f}')
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_every'] == 0:
                self.save_checkpoint(epoch, val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
                print(f'  New best model saved!')
        
        # Save training history
        self.save_training_history()
        
        print(f'\nTraining completed for {self.model_name}!')
        print(f'Best validation loss: {best_val_loss:.4f}')
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
            'src_vocab': self.src_vocab,
            'tgt_vocab': self.tgt_vocab
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
        
        torch.save(checkpoint, path)
    
    def save_training_history(self):
        """Save training history"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        path = os.path.join(self.log_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Train Seq2Seq models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='all',
                        choices=['vanilla', 'lstm', 'attention', 'all'],
                        help='Which model to train')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_and_prepare_data(config)
    
    src_vocab_size = src_vocab.n_words
    tgt_vocab_size = tgt_vocab.n_words
    
    print(f'Source vocabulary size: {src_vocab_size}')
    print(f'Target vocabulary size: {tgt_vocab_size}')
    
    # Determine which models to train
    models_to_train = []
    if args.model == 'all':
        models_to_train = ['vanilla', 'lstm', 'attention']
    else:
        models_to_train = [args.model]
    
    # Train models
    for model_name in models_to_train:
        print(f'\n{"="*60}')
        print(f'Training {model_name.upper()} model')
        print(f'{"="*60}')
        
        # Create model
        if model_name == 'vanilla':
            model = create_vanilla_seq2seq(src_vocab_size, tgt_vocab_size, config, device)
        elif model_name == 'lstm':
            model = create_lstm_seq2seq(src_vocab_size, tgt_vocab_size, config, device)
        elif model_name == 'attention':
            model = create_attention_seq2seq(src_vocab_size, tgt_vocab_size, config, device)
        
        # Create trainer and train
        trainer = Trainer(model, config, device, model_name, src_vocab, tgt_vocab)
        trainer.train(train_loader, val_loader, config['training']['num_epochs'])
    
    print('\n' + '='*60)
    print('All training completed!')
    print('='*60)


if __name__ == '__main__':
    main()
