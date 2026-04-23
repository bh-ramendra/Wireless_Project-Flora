"""
utils.py — Training loop, evaluation, metrics, and aggregation helpers.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Optional


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Training one local epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


def train_local(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    local_epochs: int = 5,
) -> tuple[float, float]:
    """Train for multiple local epochs."""
    for _ in range(local_epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, device)
    return loss, acc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model. Returns (avg_loss, accuracy)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        logits = model(input_ids, attention_mask, token_type_ids)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds       = logits.argmax(dim=-1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Communication cost tracking
# ---------------------------------------------------------------------------

def compute_comm_cost_mb(model: nn.Module, mode: str = "lora") -> float:
    """
    Compute MB of parameters transmitted per round per client.
    mode='fedavg' → all params; 'lora' → A+B matrices; 'fedsb' → R matrices only.
    """
    total_bytes = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if mode == "fedsb":
            if "lora_R" not in name:
                continue
        elif mode == "lora" or mode == "flora":
            if not any(k in name for k in ("lora_A", "lora_B", "classifier")):
                continue
        total_bytes += p.numel() * 4  # float32 = 4 bytes
    return total_bytes / (1024 ** 2)


# ---------------------------------------------------------------------------
# FedAvg aggregation helpers (get/set flat parameter vectors)
# ---------------------------------------------------------------------------

def get_trainable_params(model: nn.Module) -> OrderedDict:
    """Return only trainable parameters as an OrderedDict."""
    return OrderedDict(
        {k: v.clone().detach() for k, v in model.named_parameters() if v.requires_grad}
    )


def set_trainable_params(model: nn.Module, params: OrderedDict) -> None:
    """Load trainable parameters back into the model."""
    state = model.state_dict()
    state.update(params)
    model.load_state_dict(state, strict=False)


def fedavg_aggregate(client_params: list[OrderedDict],
                     weights: Optional[list[float]] = None) -> OrderedDict:
    """
    Standard FedAvg: weighted average of client parameter dicts.
    weights: list of floats (e.g. dataset sizes). Uniform if None.
    """
    if weights is None:
        weights = [1.0 / len(client_params)] * len(client_params)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    agg: OrderedDict = OrderedDict()
    for key in client_params[0]:
        agg[key] = sum(w * p[key] for w, p in zip(weights, client_params))
    return agg


# ---------------------------------------------------------------------------
# FLoRA stacking aggregation
# ---------------------------------------------------------------------------

def flora_aggregate(client_params: list[OrderedDict],
                    weights: Optional[list[float]] = None) -> OrderedDict:
    """
    FLoRA aggregation: stack A matrices vertically, B matrices horizontally.
    Scaling factor applied only to A (to avoid squaring pk in ΔW = BA).

    For non-LoRA params (e.g. classifier), use standard FedAvg.
    """
    if weights is None:
        weights = [1.0 / len(client_params)] * len(client_params)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    agg: OrderedDict = OrderedDict()
    keys = list(client_params[0].keys())

    for key in keys:
        if "lora_A" in key:
            # Stack A matrices vertically (dim=0), scaled by pk
            scaled = [w * p[key] for w, p in zip(weights, client_params)]
            agg[key] = torch.cat(scaled, dim=0)          # (sum_rk) × n
        elif "lora_B" in key:
            # Stack B matrices horizontally (dim=1), no scaling
            stacked = [p[key] for p in client_params]
            agg[key] = torch.cat(stacked, dim=1)          # m × (sum_rk)
        else:
            # FedAvg for classifier head etc.
            agg[key] = sum(w * p[key] for w, p in zip(weights, client_params))

    return agg


def flora_set_params(model: nn.Module, agg_params: OrderedDict) -> None:
    """
    Load FLoRA aggregated params. For LoRA layers the stacked A/B will have
    larger rank than each client's local A/B — we must resize them.
    We apply ΔW = B @ A directly to get the full update then store it.
    """
    # Build a new state dict: for LoRA layers, we update the frozen weight
    # by adding ΔW = B_stacked @ A_stacked, then reset A and B to zeros/randn.
    state = model.state_dict()

    # Collect layer prefixes that have lora_A / lora_B
    lora_prefixes = set()
    for key in agg_params:
        if "lora_A" in key:
            prefix = key.replace(".lora_A", "")
            lora_prefixes.add(prefix)

    for prefix in lora_prefixes:
        A_key = prefix + ".lora_A"
        B_key = prefix + ".lora_B"
        W_key = prefix + ".weight"

        if A_key in agg_params and B_key in agg_params:
            A = agg_params[A_key]   # (sum_rk) × n
            B = agg_params[B_key]   # m × (sum_rk)

            # Find scale from model
            scale = 1.0
            for name, module in model.named_modules():
                if name == prefix:
                    scale = module.scale
                    break

            delta_W = scale * (B @ A)           # m × n
            if W_key in state:
                state[W_key] = state[W_key] + delta_W

            # Reinitialize local A and B for next round
            rank = state[A_key].shape[0]
            state[A_key] = torch.randn_like(state[A_key]) * 0.01
            state[B_key] = torch.zeros_like(state[B_key])

    # Load non-LoRA updates (classifier etc.)
    for key, val in agg_params.items():
        if "lora_A" not in key and "lora_B" not in key:
            if key in state and state[key].shape == val.shape:
                state[key] = val

    model.load_state_dict(state, strict=False)


# ---------------------------------------------------------------------------
# Fed-SB aggregation (average R matrices only)
# ---------------------------------------------------------------------------

def fedsb_aggregate(client_params: list[OrderedDict],
                    weights: Optional[list[float]] = None) -> OrderedDict:
    """
    Fed-SB: only R matrices are trainable → simple average of R.
    Classifier head is averaged normally.
    """
    if weights is None:
        weights = [1.0 / len(client_params)] * len(client_params)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    agg: OrderedDict = OrderedDict()
    for key in client_params[0]:
        agg[key] = sum(w * p[key] for w, p in zip(weights, client_params))
    return agg


# ---------------------------------------------------------------------------
# Convergence round tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self.accuracies: list[float] = []
        self.losses:     list[float] = []
        self.comm_costs: list[float] = []
        self.convergence_round: Optional[int] = None

    def update(self, accuracy: float, loss: float, comm_mb: float, round_num: int):
        self.accuracies.append(accuracy)
        self.losses.append(loss)
        self.comm_costs.append(comm_mb)
        if self.convergence_round is None and accuracy >= self.threshold:
            self.convergence_round = round_num

    def summary(self) -> dict:
        return {
            "best_accuracy":      max(self.accuracies) if self.accuracies else 0.0,
            "final_accuracy":     self.accuracies[-1]  if self.accuracies else 0.0,
            "final_loss":         self.losses[-1]      if self.losses else 0.0,
            "convergence_round":  self.convergence_round,
            "total_comm_mb":      sum(self.comm_costs),
        }
