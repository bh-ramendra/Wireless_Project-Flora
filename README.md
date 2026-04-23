# Federated Fine-Tuning of LLMs with LoRA Variants
**Category 9 — FL with LLMs**

> **Group:** [Your Group Name]  
> **Papers:** FLoRA (Wang et al., 2024) | Fed-SB (Singhal et al., 2025)  
> **YouTube Demo:** [Add link here after recording]

---

## 📋 Overview

This repository reproduces and extends the core ideas from two papers on federated fine-tuning of large language models using Low-Rank Adaptation (LoRA):

| Paper | Method | Key Idea |
|-------|--------|----------|
| FLoRA (Wang et al., 2024) | `flora` | Stack LoRA matrices instead of averaging → noise-free aggregation |
| Fed-SB (Singhal et al., 2025) | `fedsb` | Train only r×r matrix R; exact aggregation with minimal communication |

**Baseline:** Standard FedAvg applied to LoRA fine-tuning (FedIT-style).

**Model:** BERT-base-uncased (110M params)  
**Dataset:** SST-2 (Stanford Sentiment Treebank, binary classification)  
**Framework:** [Flower (flwr)](https://flower.ai)

---

## 🗂️ Repository Structure

```
fl_project/
├── configs/               # YAML configs for each method
│   ├── fedavg.yaml
│   ├── flora.yaml
│   └── fedsb.yaml
├── src/
│   ├── model.py           # BERT + LoRA / FedSB layers
│   ├── data.py            # SST-2 loading + Dirichlet partitioning
│   ├── utils.py           # Training, evaluation, aggregation
│   ├── server.py          # Main FL simulation loop
│   └── plot_results.py    # All mandatory figures
├── results/               # Saved CSVs and plots (auto-generated)
├── report/                # Final report PDF
├── run_all_experiments.sh # One-click experiment runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone and enter directory
git clone <your-repo-url>
cd fl_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 Running Experiments

### Option A: Run everything at once
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### Option B: Run individual experiments
```bash
# FedAvg baseline
python src/server.py --config configs/fedavg.yaml

# FLoRA
python src/server.py --config configs/flora.yaml

# Fed-SB
python src/server.py --config configs/fedsb.yaml
```

### Option C: Run with custom alpha / clients
Edit the YAML config before running:
```yaml
dirichlet_alpha: 0.01   # 0.01 (extreme non-IID), 0.1, 0.5, 1.0 (near-IID)
num_clients: 10         # 10, 50, 100
num_rounds: 20
```

### Generate plots after experiments
```bash
python src/plot_results.py --results_dir results/ --num_clients 10
```

---

## 📊 Experimental Settings

| Setting | Value |
|---------|-------|
| Framework | Flower (flwr) 1.8.0 |
| Model | BERT-base-uncased |
| Dataset | SST-2 (GLUE) |
| Num Clients | 10, 50, 100 |
| Client Fraction | 0.5 |
| Local Epochs | 5 |
| Batch Size | 32 |
| Learning Rate | 0.01 (SGD + momentum=0.9) |
| LoRA Rank | 8 |
| LoRA Alpha | 16 |
| Dirichlet α | 0.01, 0.1, 0.5, 1.0, IID |
| Seed | 42 |

---

## 📈 Metrics Reported

- Global Test Accuracy (every round)
- Global Test Loss (every round)
- Convergence Round (first round ≥ 80% accuracy)
- Communication Cost (MB transmitted total)
- Trainable Parameters (LoRA parameter efficiency)

---

## 🔑 Key Concepts

### FLoRA Aggregation
Instead of averaging LoRA matrices A and B independently (which introduces noise), FLoRA **stacks** them:
```
A_global = p0*A0 ⊕ p1*A1 ⊕ ... ⊕ pK*AK   (vertical stack)
B_global = B0 ⊕ B1 ⊕ ... ⊕ BK             (horizontal stack)
ΔW = B_global @ A_global = Σ Bk @ Ak       (exact, noise-free)
```

### Fed-SB Aggregation
Only the tiny r×r matrix R is trainable. B and A are frozen after SVD initialization:
```
ΔWi = B @ Ri @ A        (client i)
R_agg = (1/c) Σ Ri      (simple average = exact aggregation)
ΔW_agg = B @ R_agg @ A  (global update, O(r²) communication)
```

---

## 📦 Dependencies

See `requirements.txt`. Key packages:
- `flwr==1.8.0`
- `torch>=2.0.0`
- `transformers==4.40.0`
- `datasets==2.19.0`

---

## 📝 Citation

```bibtex
@article{wang2024flora,
  title={FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations},
  author={Wang, Ziyao and Shen, Zheyu and He, Yexiao and Sun, Guoheng and Wang, Hongyi and Lyu, Lingjuan and Li, Ang},
  journal={arXiv preprint arXiv:2409.05976},
  year={2024}
}

@article{singhal2025fedsb,
  title={Fed-SB: A Silver Bullet for Extreme Communication Efficiency and Performance in (Private) Federated LoRA Fine-Tuning},
  author={Singhal, Raghav and Ponkshe, Kaustubh and Vartak, Rohit and Varshney, Lav R. and Vepakomma, Praneeth},
  journal={arXiv preprint arXiv:2502.15436},
  year={2025}
}
```
