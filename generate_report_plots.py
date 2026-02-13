"""
Generate plots and figures for the report
"""
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

sns.set_style('whitegrid')
sns.set_palette('husl')


def plot_training_curves(model_names, config):
    """Plot training and validation loss curves"""
    fig, axes = plt.subplots(1, len(model_names), figsize=(6*len(model_names), 5))
    
    if len(model_names) == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(model_names):
        # Load training history
        history_path = os.path.join(config['paths']['logs'], model_name, 'training_history.json')
        
        if not os.path.exists(history_path):
            print(f"Warning: History not found for {model_name}")
            continue
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        axes[idx].plot(epochs, history['train_losses'], label='Train Loss', linewidth=2)
        axes[idx].plot(epochs, history['val_losses'], label='Val Loss', linewidth=2)
        axes[idx].set_xlabel('Epoch', fontsize=12)
        axes[idx].set_ylabel('Loss', fontsize=12)
        axes[idx].set_title(f'{model_name.upper()} - Training Progress', 
                           fontsize=14, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(config['paths']['results'], 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves: {save_path}")
    plt.close()


def plot_model_comparison(model_names, config):
    """Plot comparison of models across metrics"""
    metrics_data = {
        'model': [],
        'BLEU': [],
        'Token Accuracy': [],
        'Exact Match': []
    }
    
    for model_name in model_names:
        results_path = os.path.join(config['paths']['results'], model_name, 'evaluation_results.json')
        
        if not os.path.exists(results_path):
            print(f"Warning: Results not found for {model_name}")
            continue
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        metrics_data['model'].append(model_name.upper())
        metrics_data['BLEU'].append(results['overall']['bleu'])
        metrics_data['Token Accuracy'].append(results['overall']['token_accuracy'] * 100)
        metrics_data['Exact Match'].append(results['overall']['exact_match'] * 100)
    
    # Create grouped bar chart
    x = np.arange(len(metrics_data['model']))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, metrics_data['BLEU'], width, label='BLEU Score')
    rects2 = ax.bar(x, metrics_data['Token Accuracy'], width, label='Token Accuracy (%)')
    rects3 = ax.bar(x + width, metrics_data['Exact Match'], width, label='Exact Match (%)')
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Comparison Across Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data['model'])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(config['paths']['results'], 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison: {save_path}")
    plt.close()


def plot_performance_by_length(model_names, config):
    """Plot performance by input length"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metric_names = ['bleu', 'token_accuracy', 'exact_match']
    titles = ['BLEU Score by Length', 'Token Accuracy by Length', 'Exact Match by Length']
    
    for metric_idx, (metric, title) in enumerate(zip(metric_names, titles)):
        for model_name in model_names:
            results_path = os.path.join(config['paths']['results'], model_name, 
                                       'evaluation_results.json')
            
            if not os.path.exists(results_path):
                continue
            
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Extract length buckets
            buckets = sorted(results['by_length'].keys())
            values = []
            
            for bucket in buckets:
                val = results['by_length'][bucket][metric]
                if metric != 'bleu':
                    val *= 100
                values.append(val)
            
            axes[metric_idx].plot(buckets, values, marker='o', linewidth=2, 
                                 label=model_name.upper(), markersize=8)
        
        axes[metric_idx].set_xlabel('Source Length (tokens)', fontsize=12)
        axes[metric_idx].set_ylabel('Score', fontsize=12)
        axes[metric_idx].set_title(title, fontsize=14, fontweight='bold')
        axes[metric_idx].legend()
        axes[metric_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(config['paths']['results'], 'performance_by_length.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved performance by length: {save_path}")
    plt.close()


def generate_results_table(model_names, config):
    """Generate LaTeX table of results"""
    table = "\\begin{table}[h]\n"
    table += "\\centering\n"
    table += "\\caption{Model Performance Comparison}\n"
    table += "\\begin{tabular}{|l|c|c|c|}\n"
    table += "\\hline\n"
    table += "\\textbf{Model} & \\textbf{BLEU} & \\textbf{Token Acc (\\%)} & \\textbf{Exact Match (\\%)} \\\\\n"
    table += "\\hline\n"
    
    for model_name in model_names:
        results_path = os.path.join(config['paths']['results'], model_name, 
                                   'evaluation_results.json')
        
        if not os.path.exists(results_path):
            continue
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        bleu = results['overall']['bleu']
        token_acc = results['overall']['token_accuracy'] * 100
        exact_match = results['overall']['exact_match'] * 100
        
        table += f"{model_name.upper()} & {bleu:.2f} & {token_acc:.2f} & {exact_match:.2f} \\\\\n"
    
    table += "\\hline\n"
    table += "\\end{tabular}\n"
    table += "\\end{table}\n"
    
    # Save
    save_path = os.path.join(config['paths']['results'], 'results_table.tex')
    with open(save_path, 'w') as f:
        f.write(table)
    
    print(f"Saved LaTeX table: {save_path}")
    print("\nLaTeX table:")
    print(table)


def main():
    import yaml
    
    parser = argparse.ArgumentParser(description='Generate plots for report')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    model_names = ['vanilla', 'lstm', 'attention']
    
    print("Generating plots for report...")
    print("="*60)
    
    # Generate all plots
    plot_training_curves(model_names, config)
    plot_model_comparison(model_names, config)
    plot_performance_by_length(model_names, config)
    generate_results_table(model_names, config)
    
    print("="*60)
    print("All plots generated successfully!")
    print(f"Check the '{config['paths']['results']}' directory")


if __name__ == '__main__':
    main()
