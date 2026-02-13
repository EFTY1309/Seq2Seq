#!/usr/bin/env python
"""
Master run script for the Seq2Seq Code Generation project
"""
import argparse
import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and print status"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed!")
        return False
    else:
        print(f"\n✅ {description} completed successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Run Seq2Seq Code Generation Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run.py --all
  
  # Train specific model
  python run.py --train --model lstm
  
  # Evaluate all models
  python run.py --evaluate
  
  # Visualize attention
  python run.py --visualize
  
  # Use Docker
  python run.py --docker --train
        """
    )
    
    parser.add_argument('--docker', action='store_true',
                        help='Run using Docker')
    parser.add_argument('--train', action='store_true',
                        help='Train models')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate models')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize attention')
    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline (train + evaluate + visualize)')
    parser.add_argument('--model', type=str, default='all',
                        choices=['vanilla', 'lstm', 'attention', 'all'],
                        help='Which model to train/evaluate')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all steps
    if args.all:
        args.train = True
        args.evaluate = True
        args.visualize = True
    
    # If no action specified, show help
    if not (args.train or args.evaluate or args.visualize):
        parser.print_help()
        return
    
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Seq2Seq Code Generation - Text to Python                     ║
║                  RNN | LSTM | Attention                              ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    success = True
    
    # Training
    if args.train:
        if args.docker:
            cmd = ['docker-compose', 'up', '--build', 'seq2seq']
        else:
            cmd = ['python', 'train.py', '--model', args.model, '--config', args.config]
        
        success = run_command(cmd, "Training Models")
        if not success:
            return
    
    # Evaluation
    if args.evaluate:
        if args.docker:
            cmd = ['docker-compose', 'run', '--rm', 'evaluate', 
                   'python', 'evaluate.py', '--model', args.model]
        else:
            cmd = ['python', 'evaluate.py', '--model', args.model, '--config', args.config]
        
        success = run_command(cmd, "Evaluating Models")
        if not success:
            return
    
    # Visualization
    if args.visualize:
        if args.docker:
            cmd = ['docker-compose', 'run', '--rm', 'visualize']
        else:
            cmd = ['python', 'visualize_attention.py', '--num_examples', '10', 
                   '--summary', '--config', args.config]
        
        success = run_command(cmd, "Visualizing Attention")
        if not success:
            return
    
    print(f"\n{'='*80}")
    print("✅ All tasks completed successfully!")
    print(f"{'='*80}")
    print("\nResults are available in:")
    print("  📁 checkpoints/    - Trained model weights")
    print("  📁 results/        - Evaluation metrics")
    print("  📁 visualizations/ - Attention heatmaps")
    print("  📁 logs/           - Training logs")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
