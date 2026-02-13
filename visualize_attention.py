"""
Attention visualization script
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import yaml
import os
import argparse
import numpy as np
from tqdm import tqdm

from data_loader import load_and_prepare_data, Vocabulary
from models import create_attention_seq2seq


class AttentionVisualizer:
    """Visualize attention weights"""
    
    def __init__(self, model, config, device, src_vocab, tgt_vocab):
        self.model = model
        self.config = config
        self.device = device
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Visualization directory
        self.viz_dir = os.path.join(config['paths']['visualizations'], 'attention')
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]+1}')
    
    def visualize_attention(self, src_tokens, tgt_tokens, attention_weights, example_id):
        """
        Visualize attention weights as a heatmap
        
        Args:
            src_tokens: list of source tokens
            tgt_tokens: list of target tokens
            attention_weights: (tgt_len, src_len) attention matrix
            example_id: identifier for saving the plot
        """
        try:
            # Limit visualization to reasonable size
            max_tgt_tokens = 50
            max_src_tokens = 30
            
            if len(tgt_tokens) > max_tgt_tokens:
                tgt_tokens = tgt_tokens[:max_tgt_tokens]
                attention_weights = attention_weights[:max_tgt_tokens, :]
                print(f'  Note: Truncated visualization to first {max_tgt_tokens} target tokens')
            
            if len(src_tokens) > max_src_tokens:
                src_tokens = src_tokens[:max_src_tokens]
                attention_weights = attention_weights[:, :max_src_tokens]
                print(f'  Note: Truncated visualization to first {max_src_tokens} source tokens')
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(10, len(src_tokens) * 0.5), 
                                           max(8, len(tgt_tokens) * 0.4)))
            
            # Create heatmap
            sns.heatmap(
                attention_weights,
                xticklabels=src_tokens,
                yticklabels=tgt_tokens,
                cmap='YlOrRd',
                cbar=True,
                ax=ax,
                vmin=0,
                vmax=1,
                square=False
            )
            
            ax.set_xlabel('Source Sequence (Docstring)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Target Sequence (Generated Code)', fontsize=12, fontweight='bold')
            ax.set_title(f'Attention Weights - Example {example_id}', fontsize=14, fontweight='bold')
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            
            plt.tight_layout()
            
            # Save figure
            save_path = os.path.join(self.viz_dir, f'attention_example_{example_id}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f'Saved attention visualization: {save_path}')
        except Exception as e:
            print(f'Error creating visualization: {e}')
            plt.close('all')
    
    def visualize_example(self, src, tgt, src_text, tgt_text, example_id):
        """Visualize attention for a single example"""
        self.model.eval()
        
        with torch.no_grad():
            src = src.unsqueeze(0).to(self.device)
            src_lengths = torch.tensor([src.shape[1]])
            
            # Generate with attention
            max_length = self.config['dataset']['max_code_length'] + 2
            generated, attention_weights = self.model.generate(
                src, src_lengths, max_length, self.tgt_vocab.SOS_token
            )
            
            # Get generated sequence
            generated_indices = generated[0].cpu().tolist()
            generated_text = self.tgt_vocab.decode(generated_indices)
            
            # Get attention weights (remove batch dimension)
            attention_weights = attention_weights[0].cpu().numpy()
            
            # Get tokens
            src_tokens = self.src_vocab.tokenize(src_text)
            tgt_tokens = self.tgt_vocab.tokenize(generated_text)
            
            # Truncate attention to actual lengths
            attention_weights = attention_weights[:len(tgt_tokens), :len(src_tokens)]
            
            # Print example
            print(f'\n{"="*80}')
            print(f'Example {example_id}')
            print(f'{"="*80}')
            print(f'Source (Docstring):\n  {src_text}')
            print(f'\nReference Code:\n  {tgt_text}')
            print(f'\nGenerated Code:\n  {generated_text}')
            print(f'\nAttention matrix shape: {attention_weights.shape}')
            
            # Visualize attention
            self.visualize_attention(src_tokens, tgt_tokens, attention_weights, example_id)
            
            # Save detailed info
            info = {
                'source': src_text,
                'reference': tgt_text,
                'generated': generated_text,
                'src_tokens': src_tokens,
                'tgt_tokens': tgt_tokens,
                'attention_shape': attention_weights.shape
            }
            
            return info, attention_weights
    
    def analyze_attention_patterns(self, attention_weights, src_tokens, tgt_tokens):
        """Analyze attention patterns"""
        print(f'\nAttention Analysis:')
        
        # Find max attention for each target token
        max_attentions = np.max(attention_weights, axis=1)
        max_src_indices = np.argmax(attention_weights, axis=1)
        
        print(f'\nTarget -> Most Attended Source:')
        for i, (tgt_token, src_idx, max_att) in enumerate(zip(tgt_tokens, max_src_indices, max_attentions)):
            if src_idx < len(src_tokens):
                print(f'  {tgt_token:<15} -> {src_tokens[src_idx]:<15} (weight: {max_att:.3f})')
        
        # Calculate attention entropy (measure of focus)
        epsilon = 1e-10
        entropy = -np.sum(attention_weights * np.log(attention_weights + epsilon), axis=1)
        avg_entropy = np.mean(entropy)
        
        print(f'\nAverage Attention Entropy: {avg_entropy:.3f}')
        print(f'  (Lower entropy = more focused attention)')
    
    def visualize_multiple_examples(self, data_loader, num_examples=5):
        """Visualize attention for multiple examples"""
        print(f'\nVisualizing attention for {num_examples} examples...')
        
        examples_found = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if examples_found >= num_examples:
                    break
                
                src = batch['src']
                tgt = batch['tgt']
                src_texts = batch['src_texts']
                tgt_texts = batch['tgt_texts']
                
                for i in range(src.shape[0]):
                    if examples_found >= num_examples:
                        break
                    
                    info, attention_weights = self.visualize_example(
                        src[i],
                        tgt[i],
                        src_texts[i],
                        tgt_texts[i],
                        examples_found + 1
                    )
                    
                    # Analyze attention patterns
                    self.analyze_attention_patterns(
                        attention_weights,
                        info['src_tokens'],
                        info['tgt_tokens']
                    )
                    
                    examples_found += 1
        
        print(f'\n{"="*80}')
        print(f'Visualized {examples_found} examples')
        print(f'Visualizations saved to: {self.viz_dir}')
        print(f'{"="*80}')
    
    def create_summary_visualization(self, data_loader, num_samples=100):
        """Create summary visualization of attention statistics"""
        print(f'\nCreating attention summary visualization...')
        
        all_entropies = []
        all_max_attentions = []
        
        self.model.eval()
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Processing'):
                if samples_processed >= num_samples:
                    break
                
                src = batch['src'].to(self.device)
                src_lengths = batch['src_lengths']
                
                max_length = self.config['dataset']['max_code_length'] + 2
                _, attention_weights = self.model.generate(
                    src, src_lengths, max_length, self.tgt_vocab.SOS_token
                )
                
                # Calculate statistics
                for i in range(attention_weights.shape[0]):
                    if samples_processed >= num_samples:
                        break
                    
                    att = attention_weights[i].cpu().numpy()
                    
                    # Entropy
                    epsilon = 1e-10
                    entropy = -np.sum(att * np.log(att + epsilon), axis=1)
                    all_entropies.extend(entropy.tolist())
                    
                    # Max attention
                    max_att = np.max(att, axis=1)
                    all_max_attentions.extend(max_att.tolist())
                    
                    samples_processed += 1
        
        # Create summary plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Entropy distribution
        axes[0].hist(all_entropies, bins=50, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Attention Entropy', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Attention Entropy', fontsize=14, fontweight='bold')
        axes[0].axvline(np.mean(all_entropies), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_entropies):.3f}')
        axes[0].legend()
        
        # Max attention distribution
        axes[1].hist(all_max_attentions, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].set_xlabel('Maximum Attention Weight', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Maximum Attention Weights', fontsize=14, fontweight='bold')
        axes[1].axvline(np.mean(all_max_attentions), color='red', linestyle='--',
                       label=f'Mean: {np.mean(all_max_attentions):.3f}')
        axes[1].legend()
        
        plt.tight_layout()
        
        save_path = os.path.join(self.viz_dir, 'attention_statistics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved attention statistics: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize attention weights')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of examples to visualize')
    parser.add_argument('--summary', action='store_true',
                        help='Create summary statistics visualization')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load checkpoint first to check if it exists
    checkpoint_path = os.path.join(
        config['paths']['checkpoints'],
        'attention',
        'best_model.pt'
    )
    
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint not found: {checkpoint_path}')
        print('Please train the attention model first.')
        print('\nTo train the model, run:')
        print('  python train.py --config config_quick.yaml --model attention')
        return
    
    # Load checkpoint to get vocabularies and config
    print('Loading checkpoint...')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract vocabularies from checkpoint
    if 'src_vocab' not in checkpoint or 'tgt_vocab' not in checkpoint:
        print('Error: Vocabularies not found in checkpoint!')
        print('Please retrain the model with the updated train.py')
        return
    
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    print(f'Loaded vocabularies from checkpoint: src={src_vocab.n_words}, tgt={tgt_vocab.n_words}')
    
    # Use config from checkpoint (has correct dimensions)
    model_config = checkpoint.get('config', config)
    print(f'Using embedding_dim={model_config["model"]["embedding_dim"]}, hidden_dim={model_config["model"]["hidden_dim"]}')
    
    # Load test data with checkpoint's vocabularies
    print('Loading test data...')
    from datasets import load_dataset
    from data_loader import CodeSearchNetDataset, collate_fn
    from torch.utils.data import DataLoader
    
    dataset = load_dataset(
        model_config['dataset']['name'],
        split='train',
        cache_dir=model_config['dataset']['cache_dir']
    )
    
    total_needed = model_config['dataset']['train_size'] + model_config['dataset']['val_size'] + model_config['dataset']['test_size']
    dataset = dataset.shuffle(seed=42).select(range(min(total_needed, len(dataset))))
    
    splits = dataset.train_test_split(test_size=model_config['dataset']['val_size'] + model_config['dataset']['test_size'], seed=42)
    temp_splits = splits['test'].train_test_split(test_size=model_config['dataset']['test_size'], seed=42)
    test_data = temp_splits['test']
    
    max_src_len = model_config['dataset'].get('max_docstring_length', 50)
    max_tgt_len = model_config['dataset'].get('max_code_length', 100)
    test_dataset = CodeSearchNetDataset(test_data, src_vocab, tgt_vocab, max_src_len, max_tgt_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=model_config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f'Test dataset size: {len(test_dataset)}')
    
    src_vocab_size = src_vocab.n_words
    tgt_vocab_size = tgt_vocab.n_words
    
    # Create attention model with checkpoint's config and vocab sizes
    print('Creating attention model...')
    model = create_attention_seq2seq(src_vocab_size, tgt_vocab_size, model_config, device)
    
    # Load checkpoint weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]+1}, val_loss={checkpoint["val_loss"]:.4f}')
    
    # Create visualizer
    visualizer = AttentionVisualizer(model, model_config, device, src_vocab, tgt_vocab)
    
    # Visualize examples
    visualizer.visualize_multiple_examples(test_loader, num_examples=args.num_examples)
    
    # Create summary visualization
    if args.summary:
        num_samples = min(100, len(test_dataset))
        visualizer.create_summary_visualization(test_loader, num_samples=num_samples)


if __name__ == '__main__':
    main()
