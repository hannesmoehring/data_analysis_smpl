import itertools
import torch
import math
import os
import re
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sk_normalize
from tools.IMS import IntraMotionSimilarity

from tools.utility import *


def read_data_texts(text_dir):
    data = []
    for fname in os.listdir(text_dir):
        if not fname.endswith(".txt"):
            continue
        motion_id = fname.replace(".txt", "")
        filepath = os.path.join(text_dir, fname)

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    desc, pos_tags, num1, num2 = line.split("#")
                except ValueError:
                    continue
                data.append(
                    {
                        "motion_id": motion_id,
                        "description": desc,
                        "pos_tags": pos_tags,
                        "num1": float(num1),
                        "num2": float(num2),
                    }
                )

    return pd.DataFrame(data)


_POS_KEEP = {"VERB", "NOUN", "ADJ", "ADV", "ADP", "PRON"}
_BODY_PARTS = {
    "hand",
    "hands",
    "arm",
    "arms",
    "leg",
    "legs",
    "foot",
    "feet",
    "knee",
    "knees",
    "hip",
    "hips",
    "head",
    "shoulder",
    "shoulders",
    "elbow",
    "elbows",
    "torso",
    "waist",
    "back",
}
_DIRECTION_WORDS = {"left", "right", "forward", "backward", "back", "up", "down", "clockwise", "counterclockwise"}
_SPEED_WORDS = {"slow", "slowly", "quick", "quickly", "fast", "rapid", "rapidly"}
_ACTOR_NOUNS = {
    # male
    "man": "male",
    "men": "male",
    "guy": "male",
    "boy": "male",
    "male": "male",
    # female
    "woman": "female",
    "women": "female",
    "girl": "female",
    "lady": "female",
    "female": "female",
    # neutral / role nouns
    "person": "neutral",
    "people": "neutral",
    "somebody": "neutral",
    "someone": "neutral",
    "human": "neutral",
    "child": "neutral",
    "kid": "neutral",
    "adult": "neutral",
    "dancer": "neutral",
    "walker": "neutral",
    "runner": "neutral",
    "athlete": "neutral",
    "subject": "neutral",
    "actor": "neutral",
    "individual": "neutral",
}
_PRONOUNS = {"male": {"he", "him", "his"}, "female": {"she", "her", "hers"}, "neutral": {"they", "them", "their", "theirs"}}


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


# =========================
# 1) Prepare parsed POS columns
# =========================
def ensure_parsed(df):
    if "parsed_pos" not in df.columns:
        df["parsed_pos"] = df["pos_tags"].apply(parse_pos_tag_string)
    if "tokens" not in df.columns:
        df["tokens"] = df["parsed_pos"].apply(lambda lst: [d["tok"] for d in lst])
    if "pos_list" not in df.columns:
        df["pos_list"] = df["parsed_pos"].apply(lambda lst: [d["pos"] for d in lst])
    return df


# =========================
# 2) Descriptive statistics
# =========================
def corpus_stats(df):
    df = ensure_parsed(df.copy())
    df["len_tokens"] = df["tokens"].apply(len)

    cs_vocab = Counter(tok.lower() for toks in df["tokens"] for tok in toks)
    pos_counts = Counter(pos for pos_list in df["pos_list"] for pos in pos_list)

    # POS counts per caption (means)
    def _pos_counts_row(pos_list):
        c = Counter(pos_list)
        return {f"mean_{p}": c.get(p, 0) for p in _POS_KEEP}

    pos_means = pd.DataFrame(df["pos_list"].apply(_pos_counts_row).tolist()).mean().to_dict()

    cs_stats = {
        "num_captions": len(df),
        "num_motions": df["motion_id"].nunique(),
        "avg_tokens": df["len_tokens"].mean(),
        "std_tokens": df["len_tokens"].std(ddof=1),
        "median_tokens": df["len_tokens"].median(),
        "vocab_size": len(cs_vocab),
        "top_20_words": cs_vocab.most_common(20),
        "pos_distribution": dict(pos_counts),
        "pos_means_per_caption": pos_means,
        "captions_per_motion_mean": df.groupby("motion_id").size().mean(),
        "captions_per_motion_median": df.groupby("motion_id").size().median(),
    }
    return cs_stats, cs_vocab


# =========================
# 3) Motion-specific linguistic cues
# =========================
def motion_specific_cues(df):
    df = ensure_parsed(df.copy())

    # Extract verbs/adjectives/adverbs
    def _extract(df_row):
        d = tokens_by_pos(df_row["parsed_pos"])
        verbs = [t.lower() for t in d.get("VERB", [])]
        adjs = [t.lower() for t in d.get("ADJ", [])]
        advs = [t.lower() for t in d.get("ADV", [])]
        nouns = [t.lower() for t in d.get("NOUN", [])]
        # modifiers
        dir_hits = sum(1 for t in (adjs + advs + nouns) if t.lower() in _DIRECTION_WORDS)
        speed_hits = sum(1 for t in (adjs + advs) if t.lower() in _SPEED_WORDS)
        body_mentions = sum(1 for t in nouns if t in _BODY_PARTS)
        return pd.Series(
            {
                "verbs": verbs,
                "adjs": adjs,
                "advs": advs,
                "nouns": nouns,
                "direction_hits": dir_hits,
                "speed_hits": speed_hits,
                "body_part_mentions": body_mentions,
            }
        )

    feats = df.apply(_extract, axis=1)
    df2 = pd.concat([df, feats], axis=1)

    top_verbs = Counter(v for vs in df2["verbs"] for v in vs).most_common(30)
    top_adjs = Counter(a for A in df2["adjs"] for a in A).most_common(30)
    top_advs = Counter(a for A in df2["advs"] for a in A).most_common(30)
    top_nouns = Counter(n for Ns in df2["nouns"] for n in Ns).most_common(30)

    rate_dir = (df2["direction_hits"] > 0).mean()
    rate_speed = (df2["speed_hits"] > 0).mean()
    rate_body = (df2["body_part_mentions"] > 0).mean()

    summary = {
        "top_30_verbs": top_verbs,
        "top_30_adjectives": top_adjs,
        "top_30_adverbs": top_advs,
        "top_30_nouns": top_nouns,
        "caption_rate_with_direction_cues": float(rate_dir),
        "caption_rate_with_speed_cues": float(rate_speed),
        "caption_rate_with_body_part_mentions": float(rate_body),
    }
    return summary, df2


# =========================
# 4) Redundancy: n-grams & duplicates
# =========================
def redundancy_analysis(df, n_values=(2, 3)):
    df = ensure_parsed(df.copy())
    # Normalize description text a bit
    norm = df["description"].str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df.assign(desc_norm=norm)

    dup_rate = 1 - df["desc_norm"].nunique() / len(df)

    ngram_freq = {}
    for n in n_values:
        counter = Counter()
        for toks in df["desc_norm"].str.split():
            counter.update(ngrams(toks, n=n))
        ngram_freq[n] = counter.most_common(30)

    return {"duplicate_rate": dup_rate, "top_ngrams": ngram_freq}


# ==========================
# 5) Intra-motion similarity (IMS)
# ==========================
ims = IntraMotionSimilarity(ensure_parsed_func=ensure_parsed)


def intramotion_similarity_nb(df):
    data = ims.intramotion_similarity_compare(df)
    print("\n=== Backend comparison ===")
    print(data["summary_table"])
    return data


# =========================
# 8) Actor addressing analysis
# =========================
def _classify_caption_actor(nouns, pronouns):
    """
    nouns, pronouns: lists of lowercase tokens from your parsed POS.
    Returns dict with counts + a single 'addressing_label' summarizing the caption.
    Priority rule:
      1) explicit gender noun wins (male_noun/female_noun)
      2) neutral noun -> neutral_noun
      3) otherwise fall back to pronouns (male_pron/female_pron/neutral_pron)
      4) none
    """
    # Count nouns by category
    noun_counts = {"male": 0, "female": 0, "neutral": 0}
    for n in nouns:
        cat = _ACTOR_NOUNS.get(n.lower())
        if cat:
            noun_counts[cat] += 1

    # Count pronouns by category
    pron_counts = {"male": 0, "female": 0, "neutral": 0}
    for p in pronouns:
        pl = p.lower()
        for cat, bag in _PRONOUNS.items():
            if pl in bag:
                pron_counts[cat] += 1
                break

    # Label with simple precedence
    label = "none"
    if noun_counts["male"] > 0:
        label = "male_noun"
    elif noun_counts["female"] > 0:
        label = "female_noun"
    elif noun_counts["neutral"] > 0:
        label = "neutral_noun"
    else:
        # fall back to pronouns
        if pron_counts["male"] > 0:
            label = "male_pron"
        elif pron_counts["female"] > 0:
            label = "female_pron"
        elif pron_counts["neutral"] > 0:
            label = "neutral_pron"

    return {
        "noun_male": noun_counts["male"],
        "noun_female": noun_counts["female"],
        "noun_neutral": noun_counts["neutral"],
        "pron_male": pron_counts["male"],
        "pron_female": pron_counts["female"],
        "pron_neutral": pron_counts["neutral"],
        "addressing_label": label,
    }


def actor_addressing_analysis(df: pd.DataFrame):
    df = ensure_parsed(df.copy())

    # extract nouns & pronouns from your parsed POS column
    def _extract_nouns_pron(row):
        posmap = tokens_by_pos(row["parsed_pos"])
        nouns = [t.lower() for t in posmap.get("NOUN", [])]
        prons = [t.lower() for t in posmap.get("PRON", [])]
        return nouns, prons

    res = []
    for _, r in df.iterrows():
        nouns, prons = _extract_nouns_pron(r)
        stats = _classify_caption_actor(nouns, prons)
        res.append(stats)

    A = pd.DataFrame(res)
    out = pd.concat([df.reset_index(drop=True), A], axis=1)

    # Dataset-level summary
    label_counts = out["addressing_label"].value_counts(dropna=False)
    totals = int(len(out))
    label_rates = (label_counts / totals).sort_values(ascending=False).to_dict()

    # Per-motion share of gendered addressing
    per_motion = (
        out.assign(
            any_gendered=lambda x: ((x["addressing_label"].isin(["male_noun", "female_noun", "male_pron", "female_pron"]))).astype(int),
            any_neutral=lambda x: ((x["addressing_label"].isin(["neutral_noun", "neutral_pron"]))).astype(int),
        )
        .groupby("motion_id")[["any_gendered", "any_neutral"]]
        .mean()
        .rename(columns={"any_gendered": "share_gendered_captions", "any_neutral": "share_neutral_captions"})
        .reset_index()
        .sort_values("share_gendered_captions", ascending=False)
    )

    # Top actor words (from nouns)
    all_actor_nouns = []
    for ns in out["noun_neutral"].index:  # iterate by row index
        pass  # placeholder to avoid confusion; we'll just compute directly below
    noun_counter = Counter()
    for ns in out["parsed_pos"]:
        for d in ns:
            if d["pos"] == "NOUN":
                lab = _ACTOR_NOUNS.get(d["tok"].lower())
                if lab:
                    noun_counter[d["tok"].lower()] += 1

    summary = {
        "label_rates": label_rates,  # proportion of captions per addressing label
        "top_actor_nouns": noun_counter.most_common(20),  # which actor nouns appear most
        "per_motion_summary": per_motion,  # dataframe with gendered/neutral shares
    }
    return out, summary


def lexical_richness_analysis(df):
    df = ensure_parsed(df.copy())

    # Flatten all tokens
    all_tokens = [tok.lower() for toks in df["tokens"] for tok in toks]
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))

    TTR = unique_tokens / total_tokens

    # Distinct n-gram ratios
    def distinct_n(df, n=2):
        ngrams = []
        for toks in df["tokens"]:
            ngrams.extend(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
        if not ngrams:
            return 0
        return len(set(ngrams)) / len(ngrams)

    distinct1 = distinct_n(df, 1)
    distinct2 = distinct_n(df, 2)
    distinct3 = distinct_n(df, 3)

    # Verbs per caption
    verbs_per_cap = df["pos_list"].apply(lambda x: sum(1 for p in x if p == "VERB"))
    mean_verbs = verbs_per_cap.mean()

    # Conjunctions / temporal markers
    conj_tokens = {"and", "then", "while", "before", "after"}
    conj_count = sum(tok.lower() in conj_tokens for tok in all_tokens)
    conj_rate = conj_count / total_tokens

    return {
        "TTR": TTR,
        "distinct-1": distinct1,
        "distinct-2": distinct2,
        "distinct-3": distinct3,
        "mean_verbs_per_caption": mean_verbs,
        "conj_rate": conj_rate,
    }


def analysis_routine(df: pd.DataFrame):
    if df is None:
        raise SystemExit("Please create a DataFrame `df` with columns: motion_id, description, pos_tags, num1, num2")

    # A) Corpus stats
    stats, vocab = corpus_stats(df)
    print("\n=== Corpus Stats ===")
    for k, v in stats.items():
        if k == "top_20_words":
            print(f"{k}: {v[:10]} ...")
        else:
            print(f"{k}: {v}")

    print("\n\n")
    # B) Motion-specific cues
    cues, df_cues = motion_specific_cues(df)
    print("\n=== Motion-specific Cues ===")
    print("Top verbs:", cues["top_30_verbs"][:10], "...")
    print("Rate with direction cues:", round(cues["caption_rate_with_direction_cues"], 3))
    print("Rate with speed cues:", round(cues["caption_rate_with_speed_cues"], 3))
    print("Rate with body-part mentions:", round(cues["caption_rate_with_body_part_mentions"], 3))

    print("\n\n")
    # B2) Actor addressing (NEW)
    actor_df, actor_summary = actor_addressing_analysis(df)
    print("\n=== Actor Addressing ===")
    print("Label rates:", {k: round(v, 3) for k, v in actor_summary["label_rates"].items()})
    print("Top actor nouns:", actor_summary["top_actor_nouns"][:10], "...")
    print("Per-motion (top 5 by gendered share):")
    print(actor_summary["per_motion_summary"].head(5))

    print("\n\n")
    # C) Redundancy
    red = redundancy_analysis(df, n_values=(2, 3))
    print("\n=== Redundancy ===")
    print("Duplicate caption rate:", round(red["duplicate_rate"], 4))
    print("Top bigrams:", red["top_ngrams"][2][:10], "...")
    print("Top trigrams:", red["top_ngrams"][3][:10], "...")

    print("\n\n")
    # D) Intra-motion similarity (both MiniLM & DistilBERT)
    both = ims.intramotion_similarity_compare(df)
    print("\n=== Backend comparison ===")
    print(both["summary_table"])

    print("\n\n")
    richness = lexical_richness_analysis(df)
    print("\n=== Linguistic Richness & Structural Variety ===")
    for k, v in richness.items():
        print(f"{k}: {v:.4f}")
