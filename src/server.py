"""
server.py — Flower-based FL server for FedAvg, FLoRA, and Fed-SB.

Usage:
    python src/server.py --config configs/fedavg.yaml
    python src/server.py --config configs/flora.yaml
    python src/server.py --config configs/fedsb.yaml
"""

import argparse
import copy
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# ── local imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data   import load_sst2, dirichlet_partition, get_client_loaders, get_test_loader
from model  import BertClassifier, inject_lora, count_trainable
from utils  import (
    get_device, train_local, evaluate,
    get_trainable_params, set_trainable_params,
    fedavg_aggregate, flora_aggregate, flora_set_params,
    fedsb_aggregate,
    compute_comm_cost_mb, MetricsTracker,
)


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Build model depending on method
# ---------------------------------------------------------------------------

def build_model(cfg: dict) -> torch.nn.Module:
    method = cfg["method"]
    model  = BertClassifier(num_labels=2)

    if method == "fedavg":
        # Full fine-tune (all params trainable)
        for p in model.parameters():
            p.requires_grad = True
    elif method == "flora":
        model = inject_lora(model, rank=cfg["lora_rank"],
                            alpha=cfg.get("lora_alpha", 16.0), mode="lora")
    elif method == "fedsb":
        model = inject_lora(model, rank=cfg["lora_rank"],
                            alpha=cfg.get("lora_alpha", 16.0), mode="fedsb")
    else:
        raise ValueError(f"Unknown method: {method}")

    return model


# ---------------------------------------------------------------------------
# FL training loop (simulated, in-process)
# ---------------------------------------------------------------------------

def run_fl(cfg: dict):
    set_seed(cfg.get("seed", 42))
    device = get_device()
    method = cfg["method"]
    alpha  = cfg["dirichlet_alpha"]

    print(f"\n{'='*60}")
    print(f"  Method : {method.upper()}")
    print(f"  Clients: {cfg['num_clients']}  |  Rounds: {cfg['num_rounds']}")
    print(f"  Alpha  : {alpha}  |  Rank: {cfg.get('lora_rank', 'N/A')}")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("Loading SST-2 dataset...")
    train_ds, val_ds = load_sst2()
    client_indices   = dirichlet_partition(train_ds, cfg["num_clients"],
                                           alpha=alpha, seed=cfg.get("seed", 42))
    client_loaders   = get_client_loaders(train_ds, client_indices,
                                          batch_size=cfg["batch_size"])
    test_loader      = get_test_loader(val_ds, batch_size=64)
    client_sizes     = [len(idx) for idx in client_indices]
    print(f"  Train samples per client: min={min(client_sizes)}, "
          f"max={max(client_sizes)}, total={sum(client_sizes)}")

    # ── Global model ───────────────────────────────────────────────────────
    global_model = build_model(cfg).to(device)
    trainable    = count_trainable(global_model)
    comm_mb_per_round = compute_comm_cost_mb(global_model, mode=method) * cfg["num_clients"]
    print(f"  Trainable params: {trainable:,}")
    print(f"  Est. comm cost/round: {comm_mb_per_round:.2f} MB\n")

    tracker = MetricsTracker(threshold=cfg.get("convergence_threshold", 0.80))

    # ── Results storage ────────────────────────────────────────────────────
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{method}_alpha{alpha}_clients{cfg['num_clients']}.csv"

    rows = []

    # ── FL rounds ─────────────────────────────────────────────────────────
    for rnd in range(1, cfg["num_rounds"] + 1):
        round_start = time.time()

        # Sample fraction of clients
        num_selected = max(1, int(cfg["client_fraction"] * cfg["num_clients"]))
        selected     = random.sample(range(cfg["num_clients"]), num_selected)

        client_param_list = []
        client_weights    = []

        for cid in selected:
            # Each client gets a fresh copy of the global model
            local_model = copy.deepcopy(global_model)
            local_model.to(device)

            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, local_model.parameters()),
                lr=cfg["learning_rate"],
                momentum=0.9,
            )

            # Local training
            loss, acc = train_local(
                local_model, client_loaders[cid], optimizer,
                device, local_epochs=cfg["local_epochs"],
            )

            client_param_list.append(get_trainable_params(local_model))
            client_weights.append(client_sizes[cid])

        # ── Aggregation ───────────────────────────────────────────────────
        if method == "fedavg":
            agg_params = fedavg_aggregate(client_param_list, client_weights)
            set_trainable_params(global_model, agg_params)

        elif method == "flora":
            agg_params = flora_aggregate(client_param_list, client_weights)
            flora_set_params(global_model, agg_params)

        elif method == "fedsb":
            agg_params = fedsb_aggregate(client_param_list, client_weights)
            set_trainable_params(global_model, agg_params)

        # ── Global evaluation ─────────────────────────────────────────────
        val_loss, val_acc = evaluate(global_model, test_loader, device)
        elapsed = time.time() - round_start

        cum_comm = comm_mb_per_round * rnd
        tracker.update(val_acc, val_loss, comm_mb_per_round, rnd)

        print(f"Round {rnd:3d}/{cfg['num_rounds']} | "
              f"Acc: {val_acc:.4f} | Loss: {val_loss:.4f} | "
              f"Comm: {cum_comm:.1f} MB | Time: {elapsed:.1f}s")

        rows.append({
            "method":          method,
            "round":           rnd,
            "num_clients":     cfg["num_clients"],
            "dirichlet_alpha": alpha,
            "lora_rank":       cfg.get("lora_rank", 0),
            "val_accuracy":    round(val_acc, 4),
            "val_loss":        round(val_loss, 4),
            "comm_cost_mb":    round(cum_comm, 2),
            "trainable_params":trainable,
        })

    # ── Save CSV ───────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved → {csv_path}")

    summary = tracker.summary()
    print(f"\n{'─'*50}")
    print(f"  Best Accuracy     : {summary['best_accuracy']:.4f}")
    print(f"  Final Accuracy    : {summary['final_accuracy']:.4f}")
    print(f"  Convergence Round : {summary['convergence_round']}")
    print(f"  Total Comm Cost   : {summary['total_comm_mb']:.1f} MB")
    print(f"{'─'*50}\n")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args   = parser.parse_args()
    cfg    = load_config(args.config)
    run_fl(cfg)
