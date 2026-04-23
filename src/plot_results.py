"""
plot_results.py — Generate all mandatory figures from saved CSV results.

Usage:
    python src/plot_results.py --results_dir results/
"""

import argparse
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "legend.fontsize":  11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "figure.dpi":       300,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

METHOD_COLORS = {
    "fedavg": "#E07B39",
    "flora":  "#4C72B0",
    "fedsb":  "#55A868",
}
METHOD_LABELS = {
    "fedavg": "FedAvg (Baseline)",
    "flora":  "FLoRA",
    "fedsb":  "Fed-SB",
}


# ---------------------------------------------------------------------------
# Load all CSVs into one DataFrame
# ---------------------------------------------------------------------------

def load_all(results_dir: str) -> pd.DataFrame:
    paths = glob.glob(os.path.join(results_dir, "*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    dfs = [pd.read_csv(p) for p in paths]
    df  = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows from {len(paths)} files.")
    return df


# ---------------------------------------------------------------------------
# Figure 1: Global Accuracy vs. Rounds (all methods on one plot)
# ---------------------------------------------------------------------------

def plot_accuracy_vs_rounds(df: pd.DataFrame, out_dir: Path, alpha: float = 0.1,
                             num_clients: int = 10):
    fig, ax = plt.subplots(figsize=(8, 5))
    subset = df[(df["dirichlet_alpha"] == alpha) & (df["num_clients"] == num_clients)]

    for method in ["fedavg", "flora", "fedsb"]:
        mdf = subset[subset["method"] == method].sort_values("round")
        if mdf.empty:
            continue
        ax.plot(mdf["round"], mdf["val_accuracy"],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_title(f"Global Accuracy vs. Rounds  (α={alpha}, {num_clients} clients)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.0)

    path = out_dir / f"fig1_accuracy_vs_rounds_alpha{alpha}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2: Global Loss vs. Rounds
# ---------------------------------------------------------------------------

def plot_loss_vs_rounds(df: pd.DataFrame, out_dir: Path, alpha: float = 0.1,
                        num_clients: int = 10):
    fig, ax = plt.subplots(figsize=(8, 5))
    subset = df[(df["dirichlet_alpha"] == alpha) & (df["num_clients"] == num_clients)]

    for method in ["fedavg", "flora", "fedsb"]:
        mdf = subset[subset["method"] == method].sort_values("round")
        if mdf.empty:
            continue
        ax.plot(mdf["round"], mdf["val_loss"],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=2, marker="s", markersize=4)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Global Test Loss")
    ax.set_title(f"Global Loss vs. Rounds  (α={alpha}, {num_clients} clients)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    path = out_dir / f"fig2_loss_vs_rounds_alpha{alpha}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3: Communication Cost vs. Accuracy (Category metric for Cat.9)
# ---------------------------------------------------------------------------

def plot_comm_vs_accuracy(df: pd.DataFrame, out_dir: Path, num_clients: int = 10):
    fig, ax = plt.subplots(figsize=(8, 5))

    for alpha_val in [0.01, 0.1, 0.5, "iid"]:
        for method in ["fedavg", "flora", "fedsb"]:
            mdf = df[(df["method"] == method) &
                     (df["num_clients"] == num_clients)].sort_values("round")
            # Try both float and string matching for alpha
            try:
                mdf = mdf[mdf["dirichlet_alpha"] == float(alpha_val)]
            except (ValueError, TypeError):
                mdf = mdf[mdf["dirichlet_alpha"] == alpha_val]
            if mdf.empty:
                continue
            ax.scatter(mdf["comm_cost_mb"], mdf["val_accuracy"],
                       color=METHOD_COLORS[method],
                       label=f"{METHOD_LABELS[method]}",
                       alpha=0.7, s=20)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.set_xlabel("Cumulative Communication Cost (MB)")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_title("Accuracy vs. Communication Cost (Cat. 9 metric)")
    ax.grid(True, linestyle="--", alpha=0.5)

    path = out_dir / "fig3_comm_vs_accuracy.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4: IID vs Non-IID comparison
# ---------------------------------------------------------------------------

def plot_iid_vs_noniid(df: pd.DataFrame, out_dir: Path, num_clients: int = 10):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, alpha_val in zip(axes, [1.0, 0.01]):
        label_str = "IID (α=1.0)" if alpha_val == 1.0 else "Non-IID (α=0.01)"
        ax.set_title(label_str)
        subset = df[(df["dirichlet_alpha"] == alpha_val) &
                    (df["num_clients"] == num_clients)]
        for method in ["fedavg", "flora", "fedsb"]:
            mdf = subset[subset["method"] == method].sort_values("round")
            if mdf.empty:
                continue
            ax.plot(mdf["round"], mdf["val_accuracy"],
                    color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method],
                    linewidth=2, marker="o", markersize=3)
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Global Test Accuracy")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(0, 1.0)

    fig.suptitle("IID vs. Non-IID Comparison", fontsize=14)
    path = out_dir / "fig4_iid_vs_noniid.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 5: Side-by-side bar chart — FedAvg vs methods (final accuracy)
# ---------------------------------------------------------------------------

def plot_baseline_vs_methods(df: pd.DataFrame, out_dir: Path, num_clients: int = 10):
    alphas  = sorted(df["dirichlet_alpha"].unique())
    methods = ["fedavg", "flora", "fedsb"]
    x       = np.arange(len(alphas))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, method in enumerate(methods):
        final_accs = []
        for alpha in alphas:
            mdf = df[(df["method"] == method) &
                     (df["dirichlet_alpha"] == alpha) &
                     (df["num_clients"] == num_clients)]
            if mdf.empty:
                final_accs.append(0.0)
            else:
                final_accs.append(mdf.sort_values("round").iloc[-1]["val_accuracy"])

        bars = ax.bar(x + i * width, final_accs,
                      width=width,
                      color=METHOD_COLORS[method],
                      label=METHOD_LABELS[method],
                      edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, final_accs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Dirichlet Alpha (Data Heterogeneity)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title(f"FedAvg (Baseline) vs. FL-LoRA Methods  ({num_clients} clients)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(a) for a in alphas])
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)

    path = out_dir / "fig5_baseline_vs_methods.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary Table CSV
# ---------------------------------------------------------------------------

def make_summary_table(df: pd.DataFrame, out_dir: Path):
    rows = []
    for (method, alpha, nclients), grp in df.groupby(
            ["method", "dirichlet_alpha", "num_clients"]):
        grp_sorted = grp.sort_values("round")
        final      = grp_sorted.iloc[-1]
        best_acc   = grp_sorted["val_accuracy"].max()
        conv_round = grp_sorted[grp_sorted["val_accuracy"] >= 0.80]
        conv_round = int(conv_round.iloc[0]["round"]) if not conv_round.empty else "N/A"
        total_comm = final["comm_cost_mb"]
        rows.append({
            "Method":          METHOD_LABELS.get(method, method),
            "Dataset":         "SST-2",
            "#Clients":        nclients,
            "Alpha":           alpha,
            "#Rounds":         int(final["round"]),
            "Test Accuracy (%)": f"{final['val_accuracy']*100:.2f}",
            "Best Acc (%)":    f"{best_acc*100:.2f}",
            "Convergence Round": conv_round,
            "Comm. Cost (MB)": f"{total_comm:.1f}",
            "Trainable Params": f"{int(final['trainable_params']):,}",
            "Category Metric (LoRA param eff.)": f"{int(final['trainable_params']):,}",
        })
    tbl = pd.DataFrame(rows)
    path = out_dir / "summary_table.csv"
    tbl.to_csv(path, index=False)
    print(f"\nSummary table saved → {path}")
    print(tbl.to_string(index=False))
    return tbl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--num_clients", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_all(args.results_dir)

    for alpha in [0.01, 0.1, 0.5, 1.0]:
        plot_accuracy_vs_rounds(df, out_dir, alpha=alpha, num_clients=args.num_clients)
        plot_loss_vs_rounds(df, out_dir, alpha=alpha, num_clients=args.num_clients)

    plot_comm_vs_accuracy(df, out_dir, num_clients=args.num_clients)
    plot_iid_vs_noniid(df, out_dir, num_clients=args.num_clients)
    plot_baseline_vs_methods(df, out_dir, num_clients=args.num_clients)
    make_summary_table(df, out_dir)

    print("\nAll figures generated successfully!")
