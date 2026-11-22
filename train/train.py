"""
Train a small BERT masked language model on one world.

What it does
    1. Reads corpus_a.txt and corpus_b.txt of the world (or only B with --langs b).
    2. Builds a word-level tokenizer: every word of the corpus is one token.
    3. Trains a 4-layer BERT from scratch with masked language modelling
       (15% of tokens masked).
    4. Saves the model and tokenizer to <out_dir>/final.

Settings used for all results
    hidden 128, 4 layers, 4 heads, feed-forward 512, dropout 0.1
    lr 3e-4, linear decay, 50 warm-up steps, batch 64, weight decay 0, 100 epochs

Seeds
    --seed       model init and which tokens get masked
    --data_seed  order of the sentences (default: same as --seed)

Usage
    python train.py --world_dir ../worlds/nestk_k1 --seed 42
    python train.py --world_dir ../worlds/nestk_k1 --seed 42 --langs b     # B-only baseline
"""

import argparse
from pathlib import Path

from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from transformers import (
    BertConfig,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)


class SaveEvery(TrainerCallback):
    """Keep a copy of the model every n epochs (e.g. epoch_50)."""

    def __init__(self, out_dir, every):
        self.out_dir, self.every = out_dir, every

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        epoch = round(state.epoch)
        if epoch % self.every == 0:
            model.save_pretrained(str(self.out_dir / f'epoch_{epoch}'))


def build_tokenizer(corpus_files):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'], min_frequency=1)
    tokenizer.train(corpus_files, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]',
                                   cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]')


def load_corpus(path):
    return [l for l in Path(path).read_text(encoding='utf-8').split('\n') if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world_dir', required=True)
    p.add_argument('--langs', choices=['both', 'b'], default='both')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_seed', type=int, default=None)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--save_every', type=int, default=0, help='also keep a checkpoint every n epochs')
    p.add_argument('--out_dir', default=None, help='default: <world_dir>/model_<langs>_s<seed>')
    args = p.parse_args()

    set_seed(args.seed)
    world_dir = Path(args.world_dir).expanduser()
    files = [world_dir / 'corpus_b.txt']
    if args.langs == 'both':
        files.insert(0, world_dir / 'corpus_a.txt')
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else world_dir / f'model_{args.langs}_s{args.seed}'

    tokenizer = build_tokenizer([str(f) for f in files])
    sentences = [s for f in files for s in load_corpus(f)]
    print(f'langs {args.langs} | vocab {len(tokenizer)} | sentences {len(sentences)}')
    dataset = Dataset.from_dict({'text': sentences}).map(
        lambda b: tokenizer(b['text'], truncation=True, max_length=64), batched=True, remove_columns=['text'])

    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=4 * args.hidden,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=128,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = BertForMaskedLM(config)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=64,
        learning_rate=args.lr,
        warmup_steps=50,
        weight_decay=0.0,
        lr_scheduler_type='linear',
        save_strategy='no',
        logging_strategy='epoch',
        seed=args.seed,
        data_seed=args.data_seed,
        report_to='none',
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15),
        callbacks=[SaveEvery(out_dir, args.save_every)] if args.save_every else [],
    )
    trainer.train()

    trainer.save_model(str(out_dir / 'final'))
    tokenizer.save_pretrained(str(out_dir / 'final'))
    print(f'Saved to {out_dir / "final"}')


if __name__ == '__main__':
    main()
