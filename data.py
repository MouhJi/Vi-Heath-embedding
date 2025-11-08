from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ViMedAQADataset(Dataset):
    def __init__(
        self,
        split="train",
        tokenizer_name="vinai/phobert-base",
        max_len=256,
        use_context_in_anchor=False,
    ):
        """
        Dataset for embedding QA + context.

        Args:
            split (str): train / validation / test
            tokenizer_name (str): pretrained model
            max_len (int): max length of input
            use_context_in_anchor (bool): if true, add to context into anchor
        """
        self.HF_TOKEN = "******************************" #fill your's token in huging face
        self.data = load_dataset("tmnam20/ViMedAQA", split=split, token=self.HF_TOKEN)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_len = max_len
        self.sep_token = self.tokenizer.sep_token
        self.use_context_in_anchor = use_context_in_anchor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        # ========== BUILD INPUTS ==========
        # Anchor: just question(default), or question + context if flag is turn on
        if self.use_context_in_anchor:
            anchor_text = f"{row['question']} {self.sep_token} {row['context']}"
        else:
            anchor_text = f"{row['question']}"

        # Positive: answer + context
        positive_text = f"{row['answer']} {self.sep_token} {row['context']}"

        # ========== TOKENIZE ==========
        anchor = self.tokenizer(
            anchor_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        positive = self.tokenizer(
            positive_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # ========== RETURN ==========
        return {
            "anchor_input_ids": anchor["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
            "positive_input_ids": positive["input_ids"].squeeze(0),
            "positive_attention_mask": positive["attention_mask"].squeeze(0),
        }
