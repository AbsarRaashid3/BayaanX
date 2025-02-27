import argparse
from collections import Counter
import pickle

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.size = 0

    def build_vocab(self, token_lists, min_freq=1):
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        # Add special tokens first
        for token in SPECIAL_TOKENS:
            self.word2idx[token] = len(self.word2idx)
        # Add remaining tokens meeting min frequency
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.word2idx:
                self.word2idx[token] = len(self.word2idx)
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        self.size = len(self.word2idx)

    def __len__(self):
        return self.size

    def tokenize_to_ids(self, tokens):
        return [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]

    def ids_to_tokens(self, ids):
        return [self.idx2word.get(i, "<unk>") for i in ids]

def build_and_save_vocab(pairs_file, src_vocab_file, tgt_vocab_file):
    src_token_lists = []
    tgt_token_lists = []
    with open(pairs_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            src, tgt = line.split('\t')
            src_token_lists.append(src.split())
            tgt_token_lists.append(tgt.split())
    src_vocab = Vocabulary()
    src_vocab.build_vocab(src_token_lists)
    tgt_vocab = Vocabulary()
    tgt_vocab.build_vocab(tgt_token_lists)
    print("Arabic vocab size:", src_vocab.size)
    print("English vocab size:", tgt_vocab.size)
    with open(src_vocab_file, 'wb') as f:
        pickle.dump(src_vocab, f)
    with open(tgt_vocab_file, 'wb') as f:
        pickle.dump(tgt_vocab, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_file', type=str, required=True, help='Path to preprocessed pairs file')
    parser.add_argument('--src_vocab_file', type=str, required=True, help='Path to save source vocabulary (Arabic)')
    parser.add_argument('--tgt_vocab_file', type=str, required=True, help='Path to save target vocabulary (English)')
    args = parser.parse_args()
    build_and_save_vocab(args.pairs_file, args.src_vocab_file, args.tgt_vocab_file)

if __name__ == "__main__":
    main()
