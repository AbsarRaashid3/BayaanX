# src/app.py

import streamlit as st
import torch
import pickle
from model import TransformerSeq2Seq
from vocab import Vocabulary, SPECIAL_TOKENS
from infer import greedy_decode

@st.cache_resource
def load_model_and_vocab():
    with open("src/src_vocab.pkl", "rb") as f:
        src_vocab = pickle.load(f)
    with open("src/tgt_vocab.pkl", "rb") as f:
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
    model.load_state_dict(torch.load("transformer_nmt.pt", map_location=device))
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab, device

def main():
    st.title("Arabic to English Neural Machine Translation")
    st.write("Enter an Arabic sentence below to translate into English:")

    input_sentence = st.text_area("Arabic Input", height=150)

    if st.button("Translate"):
        if input_sentence.strip():
            model, src_vocab, tgt_vocab, device = load_model_and_vocab()
            tokens = input_sentence.strip().lower().split()
            src_ids = [src_vocab.word2idx.get(token, src_vocab.word2idx["<unk>"]) for token in tokens]
            output_ids = greedy_decode(model, src_ids, src_vocab, tgt_vocab, device=device)
            output_tokens = [tgt_vocab.idx2word[i] for i in output_ids]
            filtered_tokens = [token for token in output_tokens if token not in SPECIAL_TOKENS]
            st.write("Translated English Sentence:")
            st.code(" ".join(filtered_tokens))
        else:
            st.warning("Please enter an Arabic sentence!")

if __name__ == "__main__":
    main()
