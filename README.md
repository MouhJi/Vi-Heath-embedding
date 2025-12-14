# 🧠 Vi-Heath-Embedding

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-yellow)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

**Vi-Heath-Embedding** is a **Vietnamese Medical Sentence Embedding** model built on top of **PhoBERT** using **Contrastive Learning**.  
Its goal is to map **medical questions, answers, and contexts** into the same semantic vector space to measure similarity.

---

## 🎯 Project Objectives

- Create a **domain-specific embedding** for Vietnamese medical texts  
- Enable **semantic retrieval** in Q&A systems  
- Support **context-aware question–answer matching**  
- Serve as a foundation for intelligent **medical chatbots** and **semantic search engines**

---

## 📂 Project Structure
```
ViMedEmbedding/

│

├── data.py # Dataset loading and preprocessing

├── model.py # PhoBERT + Linear projection model

├── train.py # Training with MultipleNegativesRankingLoss

├── inference.py # Inference and cosine similarity testing

└── checkpoints_with_anchor/
```

Dataset: tmnam20/ViMedAQA(hugging face)

| Column     | Description                                             |
| ---------- | ------------------------------------------------------- |
| `question` | Medical question                                        |
| `answer`   | Corresponding answer                                    |
| `context`  | Related context                                         |

---

🏗️ Model Training

🔹 Run training:
```
python train.py
```

---

🔎 Inference

Use inference.py to compute embeddings and semantic similarity between question and answer pairs.

<img width="972" height="182" alt="inference" src="https://github.com/user-attachments/assets/90754065-1498-4fd9-984e-3a09dc95f87c" />



🔹 Run Inference: python inference.py

Example output:

<img width="823" height="70" alt="Screenshot 2025-11-08 101009" src="https://github.com/user-attachments/assets/b4a26a37-c19f-468c-8ce1-d204bb60527b" />

---

🧠 Model Architecture

graph TD

    A[Question / Context] -->|Tokenizer| B[PhoBERT Encoder]
    
    C[Answer / Context] -->|Tokenizer| B2[PhoBERT Encoder]
    
    B --> D[CLS Embedding]
    
    B2 --> D2[CLS Embedding]
    
    D --> E[Projection Layer (Linear 768→768)]
    
    D2 --> E2[Projection Layer (Linear 768→768)]
    
    E --> F[L2 Normalize]
    
    E2 --> F2[L2 Normalize]
    
    F --> G[MultipleNegativesRankingLoss (Cosine Similarity)]
    
    F2 --> G

Model components:

Backbone: PhoBERT-base (vinai/phobert-base)

Projection Head: Linear layer (768 → 768)

Normalization: L2 normalization for stable cosine similarity

---

📈 Training Results

| Metric            | Value (Validation) |
| ----------------- | ------------------ |
| Cosine Similarity | 0.97               |
| Epochs            | 4                  |
| Batch size        | 16                 |
| Optimizer         | AdamW (lr = 2e-5)  |

---

💡 Potential Applications

🔍 Semantic search for medical Q&A

🤖 Intelligent healthcare chatbots

🧾 Information retrieval in clinical text

🧬 Biomedical NLP research and analysis

---

💾 Model Checkpoint

The best model is saved at:
checkpoints_with_anchor/best_model.pt


You can load it for inference or upload it to Hugging Face Model Hub.

👨‍💻 Author

Vi-Heath-embedding Project
Developed by: Mouth Ji
Powered by: PhoBERT (vinai/phobert-base)

📜 License

Released under the MIT License — free to use, modify, and distribute for research and non-commercial purposes.

