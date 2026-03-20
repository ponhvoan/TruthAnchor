import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CUECorrector:
    def __init__(self, model_name="microsoft/deberta-v3-base", max_length=128, batch_size=16, lr=2e-5, epochs=3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(self.device)

    def _prepare_dataloader(self, texts, labels=None, shuffle=False):
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        if labels is not None:
            labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"], labels_tensor)
        else:
            dataset = TensorDataset(encodings["input_ids"], encodings["attention_mask"])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, texts_train, y_train):
        train_loader = self._prepare_dataloader(texts_train, y_train, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"CUE Epoch {epoch + 1}/{self.epochs} | Loss: {total_loss / len(train_loader):.4f}")

    @torch.no_grad()
    def predict_proba(self, texts):
        self.model.eval()
        loader = self._prepare_dataloader(texts, shuffle=False)
        all_probs = []
        for batch in loader:
            input_ids, attention_mask = [b.to(self.device) for b in batch]
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            all_probs.extend(torch.sigmoid(outputs.logits).cpu().numpy().flatten())
        return np.array(all_probs)


def find_optimal_w(U_norm_val, C_val, y_val, num_steps=100):
    best_w = 0.5
    best_auroc = -np.inf
    for w in np.linspace(0, 1, num_steps):
        combined_scores = w * U_norm_val + (1 - w) * C_val
        try:
            auroc = roc_auc_score(y_val, combined_scores)
        except ValueError:
            continue
        if auroc > best_auroc:
            best_auroc = auroc
            best_w = w
    return best_w
