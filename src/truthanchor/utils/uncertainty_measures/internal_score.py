import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import directional_stats


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class InternalScore:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states

    def eigenscore(self):
        embeddings = self.hidden_states.T
        centred = embeddings - embeddings.mean(axis=-1, keepdims=True)
        cov = centred.T @ centred
        reg_cov = cov + 1e-3 * np.eye(cov.shape[1])
        eigvals = np.linalg.eigvalsh(reg_cov)
        return np.log(np.clip(eigvals, 1e-8, None)).mean(-1)


class OutputScore:
    def __init__(self, logits, per_token=False):
        self.logits = torch.stack(
            [torch.as_tensor(logits[t][0], device=device, dtype=torch.float32) for t in range(len(logits))],
            dim=0,
        )
        self.probs = F.softmax(self.logits, dim=1)
        self.per_token = per_token

    def compute_maxprob(self):
        return torch.mean(torch.max(self.probs, dim=1)[0]).item()

    def compute_ppl(self):
        seq_ppl = torch.log(torch.clip(torch.max(self.probs, dim=1)[0], min=1e-8))
        return -torch.mean(seq_ppl).item()

    def compute_entropy(self):
        seq_entropy = torch.sum(-self.probs * torch.log(torch.clip(self.probs, min=1e-8)), dim=1)
        if self.per_token:
            return seq_entropy.detach().cpu().float().numpy()
        return torch.mean(seq_entropy).item()

    def compute_tempscale(self, temperature=0.7):
        probs = F.softmax(self.logits / temperature, dim=-1)
        return probs.max(dim=-1).values.mean().item()

    def compute_energy(self, temperature=0.7):
        energy_per_token = -temperature * torch.logsumexp(self.logits / temperature, dim=-1)
        return float(energy_per_token.mean().item())


class CoEScore:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.hs_layer = self._extract_hs()

    def _extract_hs(self):
        layer_means = (
            torch.stack(
                [
                    torch.stack(hs).squeeze(1).squeeze(1) if i >= 1 else torch.stack(hs)[:, :, -1, :].squeeze(1)
                    for i, hs in enumerate(self.hidden_states)
                ],
                dim=0,
            ).mean(dim=0)
        )
        hs_layer = layer_means.cpu().float().numpy().astype(np.float32)
        return hs_layer[None, ...]

    def coe_ang(self):
        eps = 1e-8
        hs = self.hs_layer
        num_last_first = (hs[:, -1] * hs[:, 0]).sum(axis=-1)
        denom_last_first = np.linalg.norm(hs[:, -1], axis=-1, ord=2) * np.linalg.norm(hs[:, 0], axis=-1, ord=2) + eps
        cos_beta = np.clip(num_last_first / denom_last_first, -1.0, 1.0)
        norm_denominator = np.arccos(cos_beta)
        a, b = hs[:, 1:], hs[:, :-1]
        num_ab = (a * b).sum(axis=-1)
        denom_ab = np.linalg.norm(a, axis=-1, ord=2) * np.linalg.norm(b, axis=-1, ord=2) + eps
        cos_alpha = np.clip(num_ab / denom_ab, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        al_semdiff_norm = alpha / norm_denominator[:, None]
        return al_semdiff_norm, al_semdiff_norm.mean(axis=-1)

    def coe_mag(self):
        hs = self.hs_layer
        eps = 1e-8
        norm_denominator = np.linalg.norm(hs[:, -1] - hs[:, 0], axis=-1, ord=2) + eps
        diff = hs[:, 1:] - hs[:, :-1]
        diff_norm = np.linalg.norm(diff, axis=-1, ord=2)
        al_repdiff_norm = diff_norm / norm_denominator[:, None]
        return al_repdiff_norm, al_repdiff_norm.mean(axis=-1)

    def compute_CoE_R(self):
        _, ang_ave = self.coe_ang()
        _, diff_ave = self.coe_mag()
        return diff_ave - ang_ave

    def compute_CoE_C(self):
        ang_norm, _ = self.coe_ang()
        diff_norm, _ = self.coe_mag()
        x = diff_norm * np.cos(ang_norm)
        y = diff_norm * np.sin(ang_norm)
        return np.sqrt(np.mean(x, axis=-1) ** 2 + np.mean(y, axis=-1) ** 2)


class VarianceScore(CoEScore):
    def circ_variance(self):
        dirstats = directional_stats(self.hs_layer, axis=1, normalize=True)
        return 1 - dirstats.mean_resultant_length
