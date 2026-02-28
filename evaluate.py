"""
Evaluation metrics and testing script
"""
import torch
import yaml
import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from collections import defaultdict

from data_loader import load_and_prepare_data, Vocabulary
from models import create_vanilla_seq2seq, create_lstm_seq2seq, create_attention_seq2seq


class Evaluator:
    """Evaluation class for Seq2Seq models"""
    
    def __init__(self, model, config, device, model_name, src_vocab, tgt_vocab):
        self.model = model
        self.config = config
        self.device = device
        self.model_name = model_name
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # BLEU metric
        self.bleu = BLEU()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]+1}')
        print(f'Validation loss: {checkpoint["val_loss"]:.4f}')
    
    def generate_sequences(self, data_loader):
        """Generate sequences for all examples in data loader"""
        self.model.eval()
        
        all_predictions = []
        all_references = []
        all_src_texts = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc='Generating'):
                src = batch['src'].to(self.device)
                src_lengths = batch['src_lengths']
                src_texts = batch['src_texts']
                tgt_texts = batch['tgt_texts']
                
                # Generate
                max_length = self.config['dataset']['max_code_length'] + 2
                if 'attention' in self.model_name:
                    generated, _ = self.model.generate(
                        src, src_lengths, max_length, self.tgt_vocab.SOS_token
                    )
                else:
                    generated = self.model.generate(
                        src, src_lengths, max_length, self.tgt_vocab.SOS_token
                    )
                
                # Decode generated sequences
                for i in range(generated.shape[0]):
                    pred_indices = generated[i].cpu().tolist()
                    pred_text = self.tgt_vocab.decode(pred_indices)
                    
                    all_predictions.append(pred_text)
                    all_references.append(tgt_texts[i])
                    all_src_texts.append(src_texts[i])
        
        return all_predictions, all_references, all_src_texts
    
    def calculate_token_accuracy(self, predictions, references):
        """Calculate token-level accuracy"""
        correct_tokens = 0
        total_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self.tgt_vocab.tokenize(pred)
            ref_tokens = self.tgt_vocab.tokenize(ref)
            
            # Compare tokens up to the length of prediction
            min_len = min(len(pred_tokens), len(ref_tokens))
            for i in range(min_len):
                if pred_tokens[i] == ref_tokens[i]:
                    correct_tokens += 1
                total_tokens += 1
            
            # Add penalty for length mismatch
            total_tokens += abs(len(pred_tokens) - len(ref_tokens))
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0
    
    def calculate_exact_match(self, predictions, references):
        """Calculate exact match accuracy"""
        exact_matches = 0
        
        for pred, ref in zip(predictions, references):
            # Normalize whitespace for comparison
            pred_norm = ' '.join(pred.split())
            ref_norm = ' '.join(ref.split())
            
            if pred_norm == ref_norm:
                exact_matches += 1
        
        return exact_matches / len(predictions) if predictions else 0
    
    def calculate_bleu(self, predictions, references):
        """Calculate BLEU score"""
        # Format references as list of lists (sacrebleu format)
        refs = [[ref] for ref in references]
        
        try:
            bleu_score = self.bleu.corpus_score(predictions, list(zip(*refs)))
            return bleu_score.score
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            return 0.0
    
    def analyze_by_length(self, predictions, references, src_texts):
        """Analyze performance by source sequence length"""
        length_buckets = defaultdict(lambda: {'predictions': [], 'references': []})
        
        for pred, ref, src in zip(predictions, references, src_texts):
            src_len = len(self.src_vocab.tokenize(src))
            
            # Categorize by length
            if src_len <= 10:
                bucket = '0-10'
            elif src_len <= 20:
                bucket = '11-20'
            elif src_len <= 30:
                bucket = '21-30'
            else:
                bucket = '31+'
            
            length_buckets[bucket]['predictions'].append(pred)
            length_buckets[bucket]['references'].append(ref)
        
        # Calculate metrics for each bucket
        results = {}
        for bucket, data in sorted(length_buckets.items()):
            if data['predictions']:
                bleu = self.calculate_bleu(data['predictions'], data['references'])
                token_acc = self.calculate_token_accuracy(data['predictions'], data['references'])
                exact_match = self.calculate_exact_match(data['predictions'], data['references'])
                
                results[bucket] = {
                    'count': len(data['predictions']),
                    'bleu': bleu,
                    'token_accuracy': token_acc,
                    'exact_match': exact_match
                }
        
        return results
    
    def evaluate(self, data_loader):
        """Full evaluation"""
        print(f'\nEvaluating {self.model_name}...')
        
        # Generate predictions
        predictions, references, src_texts = self.generate_sequences(data_loader)
        
        # Calculate overall metrics
        bleu_score = self.calculate_bleu(predictions, references)
        token_accuracy = self.calculate_token_accuracy(predictions, references)
        exact_match = self.calculate_exact_match(predictions, references)
        
        print(f'\nOverall Results:')
        print(f'  BLEU Score:        {bleu_score:.2f}')
        print(f'  Token Accuracy:    {token_accuracy*100:.2f}%')
        print(f'  Exact Match:       {exact_match*100:.2f}%')
        
        # Analyze by length
        length_analysis = self.analyze_by_length(predictions, references, src_texts)
        
        print(f'\nResults by Source Length:')
        for bucket, metrics in sorted(length_analysis.items()):
            print(f'  Length {bucket}: (n={metrics["count"]})')
            print(f'    BLEU:          {metrics["bleu"]:.2f}')
            print(f'    Token Acc:     {metrics["token_accuracy"]*100:.2f}%')
            print(f'    Exact Match:   {metrics["exact_match"]*100:.2f}%')
        
        # Save results
        results = {
            'model': self.model_name,
            'overall': {
                'bleu': float(bleu_score),
                'token_accuracy': float(token_accuracy),
                'exact_match': float(exact_match)
            },
            'by_length': {k: {mk: float(mv) for mk, mv in v.items()} 
                         for k, v in length_analysis.items()},
            'examples': self.get_example_predictions(predictions, references, src_texts, n=10)
        }
        
        # Save to file
        results_dir = os.path.join(self.config['paths']['results'], self.model_name)
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'\nResults saved to {results_dir}')
        
        return results
    
    def get_example_predictions(self, predictions, references, src_texts, n=10):
        """Get example predictions for analysis"""
        examples = []
        
        indices = np.random.choice(len(predictions), min(n, len(predictions)), replace=False)
        
        for idx in indices:
            examples.append({
                'source': src_texts[idx],
                'reference': references[idx],
                'prediction': predictions[idx]
            })
        
        return examples


def main():
    parser = argparse.ArgumentParser(description='Evaluate Seq2Seq models')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='all',
                        choices=['vanilla', 'lstm', 'attention', 'all'],
                        help='Which model to evaluate')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Which split to evaluate on')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Vocabularies will be loaded from checkpoints
    src_vocab = None
    tgt_vocab = None
    
    # Determine which models to evaluate
    models_to_eval = []
    if args.model == 'all':
        models_to_eval = ['vanilla', 'lstm', 'attention']
    else:
        models_to_eval = [args.model]
    
    all_results = {}
    
    # Evaluate models
    for model_name in models_to_eval:
        print(f'\n{"="*60}')
        print(f'Evaluating {model_name.upper()} model')
        print(f'{"="*60}')
        
        # Load checkpoint first to get the config it was trained with
        checkpoint_path = os.path.join(
            config['paths']['checkpoints'],
            model_name,
            'best_model.pt'
        )
        
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}')
            continue
        
        # Load checkpoint to get training config and vocabularies
        print(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract vocabularies from checkpoint
        if 'src_vocab' in checkpoint and 'tgt_vocab' in checkpoint:
            src_vocab = checkpoint['src_vocab']
            tgt_vocab = checkpoint['tgt_vocab']
            print(f'Loaded vocabularies from checkpoint: src={src_vocab.n_words}, tgt={tgt_vocab.n_words}')
        else:
            print('Error: Vocabularies not found in checkpoint!')
            print('Please retrain the model with the updated train.py')
            continue
        
        # Use the config from checkpoint (has correct model dimensions)
        model_config = checkpoint.get('config', config)
        print(f'Using embedding_dim={model_config["model"]["embedding_dim"]}, hidden_dim={model_config["model"]["hidden_dim"]}')
        
        src_vocab_size = src_vocab.n_words
        tgt_vocab_size = tgt_vocab.n_words
        
        # Create model with checkpoint's config and vocab sizes
        if model_name == 'vanilla':
            model = create_vanilla_seq2seq(src_vocab_size, tgt_vocab_size, model_config, device)
        elif model_name == 'lstm':
            model = create_lstm_seq2seq(src_vocab_size, tgt_vocab_size, model_config, device)
        elif model_name == 'attention':
            model = create_attention_seq2seq(src_vocab_size, tgt_vocab_size, model_config, device)
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from epoch {checkpoint["epoch"]+1}')
        print(f'Validation loss: {checkpoint["val_loss"]:.4f}')
        
        # Load evaluation data with checkpoint's vocabularies (only for first model)
        if 'data_loader' not in locals():
            print('Loading evaluation data...')
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
            
            eval_data = temp_splits['test'] if args.split == 'test' else temp_splits['train']
            max_src_len = model_config['dataset'].get('max_docstring_length', 50)
            max_tgt_len = model_config['dataset'].get('max_code_length', 100)
            eval_dataset = CodeSearchNetDataset(eval_data, src_vocab, tgt_vocab, max_src_len, max_tgt_len)
            data_loader = DataLoader(
                eval_dataset,
                batch_size=model_config['training']['batch_size'],
                shuffle=False,
                collate_fn=collate_fn
            )
            print(f'Evaluation dataset size: {len(eval_dataset)}')
        
        # Create evaluator and evaluate
        evaluator = Evaluator(model, model_config, device, model_name, src_vocab, tgt_vocab)
        results = evaluator.evaluate(data_loader)
        
        all_results[model_name] = {
            'overall': results['overall'],
            'by_length': results['by_length']
        }
    
    # Print comparison
    if len(all_results) > 1:
        # Overall comparison
        print(f'\n{"="*60}')
        print('Model Comparison — Overall')
        print(f'{"="*60}')
        print(f'{"Model":<15} {"BLEU":<10} {"Token Acc":<12} {"Exact Match":<12}')
        print('-' * 60)
        for model_name, data in all_results.items():
            metrics = data['overall']
            print(f'{model_name:<15} {metrics["bleu"]:<10.2f} '
                  f'{metrics["token_accuracy"]*100:<12.2f} '
                  f'{metrics["exact_match"]*100:<12.2f}')

        # Cross-model by-length comparison
        all_buckets = sorted(set(
            bucket
            for data in all_results.values()
            for bucket in data['by_length'].keys()
        ))

        for metric_key, metric_label in [
            ('bleu', 'BLEU'),
            ('token_accuracy', 'Token Accuracy (%)'),
            ('exact_match', 'Exact Match (%)'),
        ]:
            print(f'\n{"="*60}')
            print(f'Model Comparison by Docstring Length — {metric_label}')
            print(f'{"="*60}')
            model_names = list(all_results.keys())
            header = f'{"Length":<10}' + ''.join(f'{m:<15}' for m in model_names)
            print(header)
            print('-' * (10 + 15 * len(model_names)))
            for bucket in all_buckets:
                row = f'{bucket:<10}'
                for model_name in model_names:
                    bucket_data = all_results[model_name]['by_length'].get(bucket)
                    if bucket_data:
                        val = bucket_data[metric_key]
                        if metric_key != 'bleu':
                            val *= 100
                        row += f'{val:<15.2f}'
                    else:
                        row += f'{"N/A":<15}'
                print(row)


if __name__ == '__main__':
    main()
