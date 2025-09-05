from itertools import combinations
import numpy as np
import pandas as pd
from sentence_transformers import InputExample
from collections import defaultdict


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


def parse_pos_tag_string(pos_str):
    """
    Input: 'a/DET man/NOUN kick/VERB ...'
    Output: list of dicts: [{'tok':'a','pos':'DET'},{'tok':'man','pos':'NOUN'},...]
    Robust to trailing spaces and stray tokens without '/'.
    """
    items = []
    for w in pos_str.strip().split():
        if "/" not in w:  # robustness
            continue
        tok, pos = w.rsplit("/", 1)
        tok = tok.strip()
        pos = pos.strip().upper()
        if not tok:
            continue
        items.append({"tok": tok, "pos": pos})
    return items


def tokens_by_pos(parsed):
    out = defaultdict(list)
    for d in parsed:
        out[d["pos"]].append(d["tok"])
    return out


def ngrams(tokens, n=2):
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def ensure_parsed(df):
    if "parsed_pos" not in df.columns:
        df["parsed_pos"] = df["pos_tags"].apply(parse_pos_tag_string)
    if "tokens" not in df.columns:
        df["tokens"] = df["parsed_pos"].apply(lambda lst: [d["tok"] for d in lst])
    if "pos_list" not in df.columns:
        df["pos_list"] = df["parsed_pos"].apply(lambda lst: [d["pos"] for d in lst])
    return df
