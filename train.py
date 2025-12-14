import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn.functional as F
from model import ViMedEmbeddingModel
from data import ViMedAQADataset
import os

def mnrl_loss(a, b, temperature=0.05):
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    logits = torch.matmul(a, b.T) / temperature
    labels = torch.arange(len(a)).to(a.device)
    return F.cross_entropy(logits, labels)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    sims = []
    for batch in dataloader:
        a_emb = model(batch["anchor_input_ids"].to(device), batch["anchor_attention_mask"].to(device))
        p_emb = model(batch["positive_input_ids"].to(device), batch["positive_attention_mask"].to(device))
        cos_sim = F.cosine_similarity(a_emb, p_emb).mean().item()
        sims.append(cos_sim)
    return sum(sims) / len(sims)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset, val_dataset = ViMedAQADataset(split="train"), ViMedAQADataset(split="validation")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = ViMedEmbeddingModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    epochs = 4
    best_val = -1

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            optimizer.zero_grad()
            a_emb = model(batch["anchor_input_ids"].to(device), batch["anchor_attention_mask"].to(device))
            p_emb = model(batch["positive_input_ids"].to(device), batch["positive_attention_mask"].to(device))
            loss = mnrl_loss(a_emb, p_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss / (pbar.n+1):.4f}"})

        val_score = evaluate(model, val_loader, device)
        print(f"Validation cosine similarity: {val_score:.4f}")

        # Save best checkpoint
        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), "checkpoints_with_anchor/best_model.pt")
            print("Saved new best model.")

if __name__ == "__main__":
    train()
