import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from truthanchor.utils.metrics import compute_ece, safe_auroc


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_2d(X):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X


class ScoreMapperNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class MLPScoreMapper:
    def __init__(
        self,
        hidden_dim=32,
        num_layers=2,
        dropout=0.1,
        lr=1e-3,
        epochs=200,
        batch_size=128,
        rank_weight=0.1,
        max_rank_pairs=5000,
        patience=30,
        print_every=10,
        val_size=0.2,
        num_bins_ece=10,
        normalize=True,
        seed=42,
        device=None,
        verbose=True,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.print_every = print_every
        self.val_size = val_size
        self.num_bins_ece = num_bins_ece
        self.normalize = normalize
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.rank_weight = rank_weight
        self.max_rank_pairs = max_rank_pairs
        self.model = None
        self.scaler = None
        self.input_dim_ = None
        self.tune_results_ = []

    def _build_model(self, input_dim):
        self.input_dim_ = input_dim
        self.model = ScoreMapperNet(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _reset_fitted_state(self):
        self.model = None
        self.scaler = None
        self.input_dim_ = None

    def _fit_scaler(self, X):
        if self.normalize:
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        else:
            self.scaler = None

    def _transform_X(self, X):
        X = ensure_2d(X)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X.astype(np.float32)

    def _prepare_tensor(self, X, y=None):
        X_tensor = torch.tensor(self._transform_X(X), dtype=torch.float32, device=self.device)
        if y is None:
            return X_tensor
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1), dtype=torch.float32, device=self.device)
        return X_tensor, y_tensor

    def _pairwise_rank_loss(self, logits, y):
        pos_logits = logits[y > 0.5]
        neg_logits = logits[y <= 0.5]
        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return torch.tensor(0.0, device=logits.device)
        diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)
        num_pairs = diff.numel()
        if self.max_rank_pairs is not None and num_pairs > self.max_rank_pairs:
            diff = diff.reshape(-1)
            idx = torch.randperm(num_pairs, device=logits.device)[: self.max_rank_pairs]
            diff = diff[idx]
        return torch.nn.functional.softplus(-diff).mean()

    def _loss_fn(self, logits, y, pos_weight):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        rank = self._pairwise_rank_loss(logits, y)
        total = bce + self.rank_weight * rank
        return total, bce.detach().item(), rank.detach().item()

    @torch.no_grad()
    def evaluate(self, X, y):
        self.model.eval()
        X_tensor, y_tensor = self._prepare_tensor(X, y)
        logits = self.model(X_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        preds = (probs >= 0.5).astype(int)
        total_loss, bce_loss, rank_loss = self._loss_fn(logits, y_tensor, pos_weight=None)
        y_np = np.asarray(y).reshape(-1).astype(int)
        return {
            "loss": float(total_loss.detach().cpu().item()),
            "bce": float(bce_loss),
            "rank": float(rank_loss),
            "acc": float(accuracy_score(y_np, preds)),
            "auroc": float(auroc) if not np.isnan(auroc := safe_auroc(y_np, probs)) else np.nan,
            "ece": float(compute_ece(probs, y_np, num_bins=self.num_bins_ece)),
            "probs": probs,
        }

    def _fit_core(self, X_train, y_train, X_val=None, y_val=None, verbose=None):
        if verbose is None:
            verbose = self.verbose
        set_seed(self.seed)
        self._reset_fitted_state()
        pos_weight = None
        X_train = ensure_2d(X_train)
        y_train = np.asarray(y_train).reshape(-1).astype(int)
        if X_val is None or y_val is None:
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.val_size,
                stratify=y_train,
                random_state=self.seed,
            )
        else:
            X_fit = X_train
            y_fit = y_train
            X_val = ensure_2d(X_val)
            y_val = np.asarray(y_val).reshape(-1).astype(int)
        self._fit_scaler(X_fit)
        self._build_model(input_dim=X_fit.shape[1])
        train_ds = TensorDataset(
            torch.tensor(self._transform_X(X_fit), dtype=torch.float32),
            torch.tensor(y_fit.astype(np.float32), dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_state = None
        best_val_auroc = -np.inf
        epochs_without_improvement = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss, _, _ = self._loss_fn(logits, yb, pos_weight)
                loss.backward()
                optimizer.step()
            val_metrics = self.evaluate(X_val, y_val)
            current_val_auroc = val_metrics["auroc"]
            if current_val_auroc > best_val_auroc + 1e-6:
                best_val_auroc = current_val_auroc
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}. Best val AUROC = {best_val_auroc:.4f}")
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def fit(self, X_train, y_train):
        return self._fit_core(X_train, y_train, verbose=self.verbose)

    def fit_with_validation(self, X_train, y_train, X_val, y_val):
        return self._fit_core(X_train, y_train, X_val=X_val, y_val=y_val, verbose=self.verbose)

    @torch.no_grad()
    def predict_proba(self, X):
        self.model.eval()
        return torch.sigmoid(self.model(self._prepare_tensor(X))).detach().cpu().numpy().reshape(-1)

    @torch.no_grad()
    def predict_logits(self, X):
        self.model.eval()
        return self.model(self._prepare_tensor(X)).detach().cpu().numpy().reshape(-1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path, extra_metadata=None):
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted mapper.")
        scaler_state = None
        if self.scaler is not None:
            scaler_state = {
                "mean_": self.scaler.mean_,
                "scale_": self.scaler.scale_,
                "var_": self.scaler.var_,
                "n_features_in_": self.scaler.n_features_in_,
                "n_samples_seen_": self.scaler.n_samples_seen_,
            }
        payload = {
            "config": {
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "rank_weight": self.rank_weight,
                "max_rank_pairs": self.max_rank_pairs,
                "patience": self.patience,
                "print_every": self.print_every,
                "val_size": self.val_size,
                "num_bins_ece": self.num_bins_ece,
                "normalize": self.normalize,
                "seed": self.seed,
                "input_dim": self.input_dim_,
            },
            "state_dict": self.model.state_dict(),
            "scaler_state": scaler_state,
            "extra_metadata": extra_metadata or {},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path, device=None):
        payload = torch.load(path, map_location=device or "cpu")
        config = payload["config"]
        mapper = cls(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            lr=config["lr"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            rank_weight=config["rank_weight"],
            max_rank_pairs=config["max_rank_pairs"],
            patience=config["patience"],
            print_every=config["print_every"],
            val_size=config["val_size"],
            num_bins_ece=config["num_bins_ece"],
            normalize=config["normalize"],
            seed=config["seed"],
            device=device,
            verbose=False,
        )
        mapper._build_model(config["input_dim"])
        mapper.model.load_state_dict(payload["state_dict"])
        scaler_state = payload["scaler_state"]
        if scaler_state is not None:
            mapper.scaler = StandardScaler()
            for key, value in scaler_state.items():
                setattr(mapper.scaler, key, value)
        mapper.model.eval()
        mapper.extra_metadata_ = payload.get("extra_metadata", {})
        return mapper
