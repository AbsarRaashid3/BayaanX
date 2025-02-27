# src/train_improved.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import TransformerSeq2Seq
from vocab import SPECIAL_TOKENS
import pickle
from tqdm import tqdm
import math


class TranslationDataset(Dataset):
    def __init__(self, pairs_file, src_vocab, tgt_vocab, max_len=128):
        self.samples = []
        with open(pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                src, tgt = line.split('\t')
                self.samples.append((src.split(), tgt.split()))
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_tokens, tgt_tokens = self.samples[idx]
        # Add start and end tokens to target
        tgt_tokens = ["<sos>"] + tgt_tokens + ["<eos>"]
        src_ids = self.src_vocab.tokenize_to_ids(src_tokens)
        tgt_ids = self.tgt_vocab.tokenize_to_ids(tgt_tokens)
        src_ids = src_ids[:self.max_len]
        tgt_ids = tgt_ids[:self.max_len]
        src_ids += [self.src_vocab.word2idx["<pad>"]] * (self.max_len - len(src_ids))
        tgt_ids += [self.tgt_vocab.word2idx["<pad>"]] * (self.max_len - len(tgt_ids))
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def create_masks(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,src_len)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch,1,1,tgt_len)
    seq_len = tgt.size(1)
    subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=tgt.device), diagonal=1).bool()
    tgt_mask = tgt_mask & ~subsequent_mask
    return src_mask, tgt_mask

def evaluate(model, dataloader, tgt_pad_idx, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    with torch.no_grad():
        for src_ids, tgt_ids in dataloader:
            src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
            dec_input = tgt_ids[:, :-1]
            dec_target = tgt_ids[:, 1:]
            src_mask, tgt_mask = create_masks(src_ids, dec_input, tgt_pad_idx)
            logits = model(src_ids, dec_input, src_mask, tgt_mask)
            logits = logits.reshape(-1, logits.size(-1))
            dec_target = dec_target.reshape(-1)
            loss = criterion(logits, dec_target)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_pairs', type=str, required=True, help='Path to training pairs file (e.g., data/train_pairs3.txt)')
    parser.add_argument('--val_pairs', type=str, required=True, help='Path to validation pairs file (e.g., data/test_pairs3.txt)')
    parser.add_argument('--src_vocab_file', type=str, required=True)
    parser.add_argument('--tgt_vocab_file', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_len', type=int, default=128)
    args = parser.parse_args()

    with open(args.src_vocab_file, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(args.tgt_vocab_file, 'rb') as f:
        tgt_vocab = pickle.load(f)

    train_dataset = TranslationDataset(args.train_pairs, src_vocab, tgt_vocab, max_len=args.max_len)
    val_dataset = TranslationDataset(args.val_pairs, src_vocab, tgt_vocab, max_len=args.max_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,
        n_heads=4,
        d_ff=512,
        num_layers=2
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.word2idx["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for src_ids, tgt_ids in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            src_ids, tgt_ids = src_ids.to(device), tgt_ids.to(device)
            dec_input = tgt_ids[:, :-1]
            dec_target = tgt_ids[:, 1:]
            src_mask, tgt_mask = create_masks(src_ids, dec_input, tgt_vocab.word2idx["<pad>"])
            optimizer.zero_grad()
            logits = model(src_ids, dec_input, src_mask, tgt_mask)
            logits = logits.reshape(-1, logits.size(-1))
            dec_target = dec_target.reshape(-1)
            loss = criterion(logits, dec_target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, tgt_vocab.word2idx["<pad>"], device)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "transformer_nmt_best.pt")
            print("Saved best model.")

    print("Training complete.")

if __name__ == "__main__":
    main()
