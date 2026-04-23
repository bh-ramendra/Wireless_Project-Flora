"""
model.py — BERT-base with LoRA, FLoRA, and Fed-SB adapter variants.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


# ---------------------------------------------------------------------------
# Base BERT classifier (frozen backbone, only classifier head is full-rank)
# ---------------------------------------------------------------------------

class BertClassifier(nn.Module):
    """BERT-base with a linear classification head."""

    def __init__(self, num_labels: int = 2, pretrained: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        hidden = self.bert.config.hidden_size          # 768
        self.classifier = nn.Linear(hidden, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooled = out.pooler_output                     # [B, 768]
        logits = self.classifier(pooled)               # [B, num_labels]
        return logits


# ---------------------------------------------------------------------------
# LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Replaces a nn.Linear with W + B @ A  (standard LoRA).
    W is frozen; A and B are trainable.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.rank  = rank
        self.scale = alpha / rank

        # Frozen original weight
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = nn.Parameter(linear.bias.data.clone(),   requires_grad=False) \
                      if linear.bias is not None else None

        # Trainable LoRA matrices  A ∈ R^{rank×in},  B ∈ R^{out×rank}
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features)  * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        lora = nn.functional.linear(
            nn.functional.linear(x, self.lora_A), self.lora_B
        )
        return base + self.scale * lora

    def get_delta_W(self) -> torch.Tensor:
        """Return ΔW = scale * B @ A  (for aggregation)."""
        return self.scale * (self.lora_B @ self.lora_A)


# ---------------------------------------------------------------------------
# Fed-SB linear layer  (LoRA-SB style: only R is trainable)
# ---------------------------------------------------------------------------

class FedSBLinear(nn.Module):
    """
    LoRA-SB variant: ΔW = B @ R @ A
    B and A are frozen (initialized via SVD approximation).
    Only R ∈ R^{rank×rank} is trainable — tiny communication cost.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.rank  = rank
        self.scale = alpha / rank

        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        self.bias   = nn.Parameter(linear.bias.data.clone(),   requires_grad=False) \
                      if linear.bias is not None else None

        # Frozen B ∈ R^{out×rank},  A ∈ R^{rank×in}  (SVD init)
        B_init, A_init = self._svd_init(linear.weight.data, rank)
        self.lora_B = nn.Parameter(B_init, requires_grad=False)
        self.lora_A = nn.Parameter(A_init, requires_grad=False)

        # Trainable R ∈ R^{rank×rank}
        self.lora_R = nn.Parameter(torch.eye(rank) * 0.01)

    @staticmethod
    def _svd_init(W: torch.Tensor, rank: int):
        """Initialize B, A from top-r SVD of W."""
        U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
        U  = U[:, :rank]
        S  = S[:rank]
        Vh = Vh[:rank, :]
        B = (U * S.sqrt()).to(W.dtype)
        A = (Vh * S.sqrt().unsqueeze(1)).to(W.dtype)
        return B, A

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias)
        # ΔW x = B (R (A x))
        Ax  = nn.functional.linear(x,                 self.lora_A)
        RAx = nn.functional.linear(Ax,                self.lora_R)
        BRAx = nn.functional.linear(RAx,              self.lora_B)
        return base + self.scale * BRAx

    def get_R(self) -> torch.Tensor:
        return self.lora_R.data.clone()


# ---------------------------------------------------------------------------
# Helper: inject LoRA / Fed-SB layers into BERT self-attention Q & V
# ---------------------------------------------------------------------------

def inject_lora(model: BertClassifier, rank: int = 8,
                alpha: float = 16.0, mode: str = "lora") -> BertClassifier:
    """
    Replace Q and V projection layers in every BERT attention block
    with LoRALinear (mode='lora') or FedSBLinear (mode='fedsb').
    All other parameters are frozen.
    """
    assert mode in ("lora", "fedsb"), f"Unknown mode: {mode}"

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze classifier head
    for p in model.classifier.parameters():
        p.requires_grad = True

    for layer in model.bert.encoder.layer:
        attn = layer.attention.self
        if mode == "lora":
            attn.query = LoRALinear(attn.query, rank=rank, alpha=alpha)
            attn.value = LoRALinear(attn.value, rank=rank, alpha=alpha)
        else:
            attn.query = FedSBLinear(attn.query, rank=rank, alpha=alpha)
            attn.value = FedSBLinear(attn.value, rank=rank, alpha=alpha)

    return model


# ---------------------------------------------------------------------------
# Utility: count trainable parameters
# ---------------------------------------------------------------------------

def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_lora_params(model: nn.Module) -> int:
    """Count only LoRA adapter parameters (A, B, or R matrices)."""
    total = 0
    for name, p in model.named_parameters():
        if p.requires_grad and any(k in name for k in ("lora_A", "lora_B", "lora_R")):
            total += p.numel()
    return total
