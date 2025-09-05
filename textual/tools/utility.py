from itertools import combinations
import numpy as np
import pandas as pd
from sentence_transformers import InputExample


def build_same_motion_pairs(df: pd.DataFrame, seed: int = 42) -> list:
    """
    df: must have columns ["motion_id", "description"]
    Returns: list[InputExample] with label=1.0 for same-motion pairs
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for mid, sub in df.groupby("motion_id"):
        descs = sub["description"].dropna().tolist()
        if len(descs) < 2:
            continue
        comb = list(combinations(range(len(descs)), 2))  # all i<j pairs
        if len(comb) > 100:
            comb = list(rng.choice(len(comb), size=100, replace=False))
            comb = [list(combinations(range(len(descs)), 2))[k] for k in comb]
        for i, j in comb:
            a, b = descs[i], descs[j]
            pairs.append(InputExample(texts=[a, b], label=1.0))
    return pairs
