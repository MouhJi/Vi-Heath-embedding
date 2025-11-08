import torch
from transformers import AutoTokenizer
from model import ViMedEmbeddingModel
import torch.nn.functional as F

def get_embedding(text, model, tokenizer, device="cpu", max_len=256):
    inputs = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=max_len, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        emb = model(inputs["input_ids"], inputs["attention_mask"])
    return emb

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViMedEmbeddingModel()
    model.load_state_dict(torch.load("checkpoints_with_anchor/best_model.pt", map_location=device))
    model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=True)

    # test
    question = "Thuốc Buscopan có thể gây ra tác dụng phụ nào liên quan đến huyết áp?"
    answer = "Thuốc Buscopan có thể gây hạ huyết áp và chóng mặt."
    context = """- Bí tiểu - Khô miệng - Hạ huyết áp, chóng mặt - Nhịp tim nhanh"""

    text_q = f"{question}"
    text_a = f"{answer}"

    emb_q = get_embedding(text_q, model, tokenizer, device)
    emb_a = get_embedding(text_a, model, tokenizer, device)

    cos_sim = F.cosine_similarity(emb_q, emb_a).item()
    print(f"🔹 Cosine Similarity: {cos_sim:.4f}")
