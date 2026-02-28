"""
Interactive Model Comparison
Usage:  python interactive_compare.py --config config_quick.yaml
        python interactive_compare.py --config config_quick.yaml --input "sort a list of numbers"
"""
import ast
import argparse
import os
import sys
import torch
import yaml
from sacrebleu.metrics import BLEU

from models import create_vanilla_seq2seq, create_lstm_seq2seq, create_attention_seq2seq

# ── ANSI colours (work on most terminals) ──────────────────────────────────────
RESET  = '\033[0m'
BOLD   = '\033[1m'
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
RED    = '\033[91m'
DIM    = '\033[2m'

MODEL_COLOURS = {
    'vanilla':   '\033[94m',   # blue
    'lstm':      '\033[95m',   # magenta
    'attention': '\033[92m',   # green
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_model(model_name, config, device):
    """Load a trained model from its best checkpoint."""
    checkpoint_path = os.path.join(
        config['paths']['checkpoints'], model_name, 'best_model.pt'
    )
    if not os.path.exists(checkpoint_path):
        print(f'{RED}Checkpoint not found for "{model_name}": {checkpoint_path}{RESET}')
        return None, None, None, None

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'src_vocab' not in ckpt or 'tgt_vocab' not in ckpt:
        print(f'{RED}Vocabularies missing in checkpoint for "{model_name}".  Retrain first.{RESET}')
        return None, None, None, None

    src_vocab  = ckpt['src_vocab']
    tgt_vocab  = ckpt['tgt_vocab']
    model_cfg  = ckpt.get('config', config)

    if model_name == 'vanilla':
        model = create_vanilla_seq2seq(src_vocab.n_words, tgt_vocab.n_words, model_cfg, device)
    elif model_name == 'lstm':
        model = create_lstm_seq2seq(src_vocab.n_words, tgt_vocab.n_words, model_cfg, device)
    elif model_name == 'attention':
        model = create_attention_seq2seq(src_vocab.n_words, tgt_vocab.n_words, model_cfg, device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch    = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f'  {GREEN}✓{RESET} {model_name:<12} epoch={epoch+1 if isinstance(epoch,int) else epoch}  val_loss={val_loss:.4f}')
    return model, src_vocab, tgt_vocab, model_cfg


def encode_input(text, src_vocab, max_len, device):
    """Tokenise and encode user text into a tensor."""
    tokens = src_vocab.tokenize(text)[:max_len]
    indices = [src_vocab.word2idx.get(t, src_vocab.UNK_token) for t in tokens]
    indices.append(src_vocab.EOS_token)
    src   = torch.tensor([indices], dtype=torch.long).to(device)
    src_l = torch.tensor([len(indices)])
    return src, src_l


def generate(model, model_name, src, src_lengths, tgt_vocab, max_len, device):
    """Run the model and return (generated_text, attention_weights_or_None)."""
    with torch.no_grad():
        if 'attention' in model_name:
            generated, attn = model.generate(src, src_lengths, max_len, tgt_vocab.SOS_token)
            attn = attn[0].cpu().numpy()
        else:
            generated = model.generate(src, src_lengths, max_len, tgt_vocab.SOS_token)
            attn = None

    indices = generated[0].cpu().tolist()
    text    = tgt_vocab.decode(indices)
    return text, attn


# ── scoring helpers ────────────────────────────────────────────────────────────

def score_syntax(code):
    """Return True if code is parseable Python."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def score_bleu(prediction, reference, bleu_scorer):
    """Return corpus BLEU (single sentence)."""
    try:
        return bleu_scorer.corpus_score([prediction], [[reference]]).score
    except Exception:
        return 0.0


def score_token_overlap(pred_tokens, ref_tokens):
    """Simple token-level F1 overlap (unordered)."""
    pred_set = set(pred_tokens)
    ref_set  = set(ref_tokens)
    if not pred_set or not ref_set:
        return 0.0
    precision = len(pred_set & ref_set) / len(pred_set)
    recall    = len(pred_set & ref_set) / len(ref_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── display ────────────────────────────────────────────────────────────────────

def print_separator(char='─', width=70):
    print(DIM + char * width + RESET)


def print_header(text, width=70):
    pad = (width - len(text) - 2) // 2
    print(BOLD + '─' * pad + f' {text} ' + '─' * (width - pad - len(text) - 2) + RESET)


def display_results(input_text, reference, models_output, bleu_scorer, tgt_vocab):
    """Pretty-print all model outputs with scoring."""
    width = 72

    print()
    print_header('INPUT DOCSTRING', width)
    print(f'  {CYAN}{input_text}{RESET}')

    if reference:
        print_header('REFERENCE CODE', width)
        for line in reference.splitlines():
            print(f'  {DIM}{line}{RESET}')

    print_header('MODEL OUTPUTS', width)

    scores = {}
    for model_name, (code, attn) in models_output.items():
        colour    = MODEL_COLOURS.get(model_name, '')
        valid_py  = score_syntax(code)
        ref_tok   = tgt_vocab.tokenize(reference) if reference else []
        pred_tok  = tgt_vocab.tokenize(code)
        f1        = score_token_overlap(pred_tok, ref_tok) if reference else None
        bleu      = score_bleu(code, reference, bleu_scorer) if reference else None

        scores[model_name] = {
            'syntax': valid_py,
            'bleu':   bleu,
            'f1':     f1,
            'tokens': len(pred_tok),
        }

        print()
        print(f'{BOLD}{colour}▶ {model_name.upper()}{RESET}')
        print_separator()

        if code.strip():
            for line in code.splitlines():
                print(f'  {line}')
        else:
            print(f'  {RED}(empty output){RESET}')

        # inline mini-scorecard
        syn_str  = f'{GREEN}✓ valid Python{RESET}' if valid_py else f'{RED}✗ syntax error{RESET}'
        tok_str  = f'{len(pred_tok)} tokens'
        bleu_str = f'BLEU {bleu:.1f}' if bleu is not None else ''
        f1_str   = f'Token-F1 {f1*100:.1f}%' if f1 is not None else ''
        details  = '  '.join(filter(None, [syn_str, tok_str, bleu_str, f1_str]))
        print(f'  {DIM}[{details}{DIM}]{RESET}')

    # ── winner determination ──────────────────────────────────────────────────
    if reference:
        print()
        print_header('COMPARISON SUMMARY', width)
        print(f'  {"Model":<12} {"Valid Python":<15} {"BLEU":>7}  {"Token-F1":>10}  {"Tokens":>7}')
        print_separator()

        ranked = sorted(
            scores.items(),
            key=lambda x: (x[1]['syntax'], x[1]['bleu'] or 0, x[1]['f1'] or 0),
            reverse=True
        )
        for rank, (mn, s) in enumerate(ranked):
            colour   = MODEL_COLOURS.get(mn, '')
            syn_icon = f'{GREEN}✓{RESET}' if s['syntax'] else f'{RED}✗{RESET}'
            bleu_v   = f"{s['bleu']:>7.2f}" if s['bleu'] is not None else '    N/A'
            f1_v     = f"{s['f1']*100:>9.1f}%" if s['f1'] is not None else '      N/A'
            medal    = ['🥇', '🥈', '🥉'][rank] if rank < 3 else '  '
            print(f'  {colour}{medal} {mn:<10}{RESET}  {syn_icon:<13}  {bleu_v}  {f1_v}  {s["tokens"]:>7}')

        winner = ranked[0][0]
        print()
        print(f'  {BOLD}Best output: {MODEL_COLOURS[winner]}{winner.upper()}{RESET}', end='')
        reasons = []
        if scores[winner]['syntax']:
            reasons.append('valid Python')
        if scores[winner]['bleu'] is not None:
            reasons.append(f'BLEU {scores[winner]["bleu"]:.1f}')
        if reasons:
            print(f'  {DIM}({", ".join(reasons)}){RESET}')
        else:
            print()

    print_separator('═', width)


# ── main loop ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Interactively compare all three Seq2Seq models')
    parser.add_argument('--config', type=str, default='config_quick.yaml')
    parser.add_argument('--input',  type=str, default=None,
                        help='Run once with this docstring instead of interactive mode')
    parser.add_argument('--reference', type=str, default=None,
                        help='Optional reference code for scoring')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{BOLD}Loading models  (device: {device}){RESET}')
    print_separator()

    models = {}
    vocabs = {}
    for name in ['vanilla', 'lstm', 'attention']:
        model, src_vocab, tgt_vocab, model_cfg = load_model(name, config, device)
        if model is not None:
            models[name] = (model, src_vocab, tgt_vocab, model_cfg)
            vocabs[name] = (src_vocab, tgt_vocab)

    if not models:
        print(f'{RED}No models loaded — train them first with:')
        print(f'  python train.py --config {args.config} --model all{RESET}')
        sys.exit(1)

    # Use the first loaded vocab as shared vocab for display
    _, shared_tgt_vocab = list(vocabs.values())[0]
    bleu_scorer = BLEU()

    max_len = config['dataset']['max_code_length'] + 2
    max_src = config['dataset']['max_docstring_length']

    print(f'\n{BOLD}All models ready.{RESET}')

    # ── single-shot mode ──────────────────────────────────────────────────────
    if args.input:
        model_outputs = {}
        for name, (model, src_vocab, tgt_vocab, _) in models.items():
            src, src_l = encode_input(args.input, src_vocab, max_src, device)
            code, attn = generate(model, name, src, src_l, tgt_vocab, max_len, device)
            model_outputs[name] = (code, attn)
        display_results(args.input, args.reference or '', model_outputs, bleu_scorer, shared_tgt_vocab)
        return

    # ── interactive REPL ─────────────────────────────────────────────────────
    print(f'\n{DIM}Type a Python function description and press Enter.')
    print(f'Optionally paste a reference answer on the next prompt.')
    print(f'Type {BOLD}quit{DIM} or {BOLD}exit{DIM} to stop.{RESET}\n')

    while True:
        try:
            raw = input(f'{BOLD}{CYAN}Docstring > {RESET}').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nBye!')
            break

        if raw.lower() in ('quit', 'exit', 'q'):
            print('Bye!')
            break
        if not raw:
            continue

        ref = input(f'{DIM}Reference code (optional, press Enter to skip) > {RESET}').strip()

        model_outputs = {}
        for name, (model, src_vocab, tgt_vocab, _) in models.items():
            src, src_l = encode_input(raw, src_vocab, max_src, device)
            code, attn = generate(model, name, src, src_l, tgt_vocab, max_len, device)
            model_outputs[name] = (code, attn)

        display_results(raw, ref, model_outputs, bleu_scorer, shared_tgt_vocab)
        print()


if __name__ == '__main__':
    main()
