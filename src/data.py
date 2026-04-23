"""
data.py — SST-2 loading & Dirichlet non-IID partitioning for FL clients.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset
from transformers import BertTokenizer


# ---------------------------------------------------------------------------
# Tokenized SST-2 dataset wrapper
# ---------------------------------------------------------------------------

class SST2Dataset(Dataset):
    def __init__(self, hf_split, tokenizer, max_length: int = 128):
        self.data      = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        text  = item["sentence"]
        label = item["label"]

        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "labels":         torch.tensor(label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Dirichlet non-IID partition
# ---------------------------------------------------------------------------

def dirichlet_partition(
    dataset: Dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[list[int]]:
    """
    Partition dataset indices among clients using Dirichlet(alpha).
    Lower alpha → more heterogeneous (non-IID).
    alpha='iid' → equal random split.
    """
    rng = np.random.default_rng(seed)
    labels = np.array([dataset[i]["labels"].item() for i in range(len(dataset))])
    num_classes = int(labels.max()) + 1

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        if alpha == "iid":
            # Equal split
            splits = np.array_split(class_idx, num_clients)
            for k, split in enumerate(splits):
                client_indices[k].extend(split.tolist())
        else:
            # Sample proportions from Dirichlet
            proportions = rng.dirichlet(np.ones(num_clients) * float(alpha))
            proportions = (proportions * len(class_idx)).astype(int)
            # Fix rounding
            proportions[-1] = len(class_idx) - proportions[:-1].sum()
            cum = 0
            for k, n in enumerate(proportions):
                client_indices[k].extend(class_idx[cum: cum + n].tolist())
                cum += n

    # Shuffle each client's indices
    for k in range(num_clients):
        random.Random(seed + k).shuffle(client_indices[k])

    return client_indices


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_sst2(tokenizer_name: str = "bert-base-uncased", max_length: int = 128):
    """Load SST-2 train / validation splits."""
    raw   = load_dataset("glue", "sst2")
    tok   = BertTokenizer.from_pretrained(tokenizer_name)
    train = SST2Dataset(raw["train"],      tok, max_length)
    val   = SST2Dataset(raw["validation"], tok, max_length)
    return train, val


def get_client_loaders(
    train_dataset: Dataset,
    client_indices: list[list[int]],
    batch_size: int = 32,
    num_workers: int = 2,
) -> list[DataLoader]:
    """Return one DataLoader per client."""
    loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        loaders.append(loader)
    return loaders


def get_test_loader(val_dataset: Dataset, batch_size: int = 64,
                    num_workers: int = 2) -> DataLoader:
    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
