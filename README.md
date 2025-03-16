# 🌍 BayaanX - Arabic to English Neural Machine Translation

BayaanX is a **Neural Machine Translation (NMT)** system that translates **Arabic** text into **English** using a **Transformer-based Sequence-to-Sequence model**. It is trained on the **Helsinki-NLP/tatoeba_mt dataset** and includes preprocessing, training, inference, and a web-based translation interface using **Streamlit**.

---

## 📌 Features

✅ **Preprocessing pipeline** to generate training and evaluation data  
✅ **Custom Vocabulary Builder** with tokenization  
✅ **Transformer-based Encoder-Decoder model**  
✅ **Training script** with batch processing and multi-head attention  
✅ **Inference script** for sentence translation  
✅ **Interactive Web App** for real-time translation  


---

## 🚀 Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/AbsarRaashid3/NMT-Arabic-English.git
cd NMT-Arabic-English

2️⃣ Install Dependencies
Make sure you have Python 3.8+ installed. Then, install the required packages:
pip install -r requirements.txt

3️⃣ Download & Preprocess the Dataset
Generate training and test pairs:
python src/preprocess.py --output_file data/train_pairs.txt --split validation
python src/preprocess.py --output_file data/test_pairs.txt --split test

4️⃣ Build Vocabulary
python src/vocab.py --pairs_file data/train_pairs.txt --src_vocab_file src/src_vocab.pkl --tgt_vocab_file src/tgt_vocab.pkl

5️⃣ Train the Model
python src/train.py --pairs_file data/train_pairs.txt --src_vocab_file src/src_vocab.pkl --tgt_vocab_file src/tgt_vocab.pkl --epochs 50 --batch_size 32

6️⃣ Run Inference
To translate an Arabic sentence:
python src/infer.py --model_checkpoint transformer_nmt.pt --src_vocab_file src/src_vocab.pkl --tgt_vocab_file src/tgt_vocab.pkl --input_sentence "يا له من مغامر !"

7️⃣ Run the Web App
Launch the Streamlit web interface for translation:
streamlit run src/app.py

```
## 🎯 Model Architecture
### The translation model is based on the Transformer architecture using Multi-Head Attention and Positional Encoding.

**It includes:**

**Encoder: Multi-layer Transformer Encoder**
**Decoder: Transformer Decoder with attention to the encoder’s outputs**
**Token Embeddings and Positional Encoding**
**Beam Search / Greedy Search for inference**

## 📊 Training Details
```
Dataset: Helsinki-NLP/tatoeba_mt (Arabic-English)
Model Size: 2-layer Transformer with 256 hidden units
Optimizer: Adam (lr=1e-4)
Loss Function: Cross-Entropy
Batch Size: 32
Epochs: 50
```

## ✨ Credits & Acknowledgments
**Hugging Face Datasets – for providing the Tatoeba Arabic-English dataset**
**PyTorch – for the deep learning framework**
**Streamlit – for the interactive UI**
## 📌 Developed by Absar Raashid 

## Some Results:
![NMT11](https://github.com/user-attachments/assets/42ee6e77-0332-437a-9977-44f5cbf44c90)
![NMT2](https://github.com/user-attachments/assets/976e2826-b7e9-4cb4-9ef7-bb954297b185)
![NMT3](https://github.com/user-attachments/assets/2388f474-6008-4561-9b20-5af48ab489ba)


