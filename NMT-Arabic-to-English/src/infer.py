# src/infer.py

import argparse
import torch
import pickle
from model import TransformerSeq2Seq
from vocab import Vocabulary, SPECIAL_TOKENS

def greedy_decode(model, src_ids, src_vocab, tgt_vocab, max_len=128, device='cpu'):
    model.eval()
    with torch.no_grad():
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = (src != src_vocab.word2idx["<pad>"]).unsqueeze(1).unsqueeze(2)
        enc_out = model.encoder(src, src_mask)
        ys = torch.tensor([[tgt_vocab.word2idx["<sos>"]]], dtype=torch.long, device=device)
        for i in range(max_len - 1):
            tgt_mask = (ys != tgt_vocab.word2idx["<pad>"]).unsqueeze(1).unsqueeze(2)
            seq_len = ys.size(1)
            subsequent_mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
            tgt_mask = tgt_mask & ~subsequent_mask
            out = model.decoder(ys, enc_out, src_mask, tgt_mask)
            prob = out[:, -1, :]
            next_word = torch.argmax(prob, dim=1).item()
            ys = torch.cat([ys, torch.tensor([[next_word]], device=device)], dim=1)
            if next_word == tgt_vocab.word2idx["<eos>"]:
                break
        return ys[0].cpu().numpy().tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, required=True)
    parser.add_argument('--src_vocab_file', type=str, required=True)
    parser.add_argument('--tgt_vocab_file', type=str, required=True)
    parser.add_argument('--input_sentence', type=str, required=True,
                        help="Arabic sentence to translate")
    parser.add_argument('--max_len', type=int, default=128)
    args = parser.parse_args()

    with open(args.src_vocab_file, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(args.tgt_vocab_file, 'rb') as f:
        tgt_vocab = pickle.load(f)

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
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    src_tokens = args.input_sentence.strip().lower().split()
    src_ids = [src_vocab.word2idx.get(token, src_vocab.word2idx["<unk>"]) for token in src_tokens]
    output_ids = greedy_decode(model, src_ids, src_vocab, tgt_vocab, max_len=args.max_len, device=device)
    output_tokens = [tgt_vocab.idx2word[i] for i in output_ids]
    filtered_tokens = [token for token in output_tokens if token not in SPECIAL_TOKENS]
    print("Translated English sentence:")
    print(" ".join(filtered_tokens))

if __name__ == "__main__":
    main()
