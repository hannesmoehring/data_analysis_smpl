import itertools
import math
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as sk_normalize

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_ST = True
    _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error ({e})\n\n sentence-transformers not available, falling back to TF-IDF for similarity.")
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer

_TFIDF = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2))


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


# =========================
# 5) Intra-motion similarity
# =========================
def _mean_pool(last_hidden, attention_mask):
    # last_hidden: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
    masked = last_hidden * mask
    summed = masked.sum(dim=1)  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
    return summed / counts


def _mean_pairwise_cosine_from_normed(V):
    """
    V: [N, D], assumed L2-normalized.
    Returns mean cosine similarity over all pairs (upper triangle, excl diag).
    """
    n = len(V)
    if n < 2:
        return np.nan
    S = np.clip(V @ V.T, -1.0, 1.0)
    iu = np.triu_indices(n, k=1)
    return float(S[iu].mean())


def intramotion_similarity(df, model="minilm"):
    df = ensure_parsed(df.copy())
    # Precompute embeddings once
    Z, label = sentence_embeddings(df["description"].tolist(), model=model)
    # Map row idx -> vector
    idx_to_vec = {i: Z[i] for i in range(len(Z))}

    rows = []
    for mid, sub in df.reset_index(drop=True).groupby("motion_id"):
        idxs = sub.index.tolist()
        V = np.stack([idx_to_vec[i] for i in idxs], axis=0)
        mpc = _mean_pairwise_cosine_from_normed(V)
        rows.append({"motion_id": mid, "mean_intra_caption_sim": mpc, "num_captions": len(idxs)})

    out = pd.DataFrame(rows).sort_values("mean_intra_caption_sim", ascending=False)
    summary = {
        "backend": label,
        "mean_of_means": float(out["mean_intra_caption_sim"].mean(skipna=True)),
        "median_of_means": float(out["mean_intra_caption_sim"].median(skipna=True)),
        "quantiles": out["mean_intra_caption_sim"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
    }
    return out, summary


def intramotion_similarity_compare(df):
    print("\n=== Intra-motion Similarity (MiniLM) ===")
    mini_df, mini_sum = intramotion_similarity(df, model="minilm")
    print(mini_sum)
    print("Top 3 most consistent motions (MiniLM):")
    print(mini_df.head(3)[["motion_id", "mean_intra_caption_sim", "num_captions"]])

    print("\n=== Intra-motion Similarity (DistilBERT-uncased) ===")
    dist_df, dist_sum = intramotion_similarity(df, model="distilbert")
    print(dist_sum)
    print("Top 3 most consistent motions (DistilBERT):")
    print(dist_df.head(3)[["motion_id", "mean_intra_caption_sim", "num_captions"]])

    # Side-by-side summary table
    cmp_tbl = pd.DataFrame(
        [
            {"backend": mini_sum["backend"], **{k: v for k, v in mini_sum.items() if k != "backend"}},
            {"backend": dist_sum["backend"], **{k: v for k, v in dist_sum.items() if k != "backend"}},
        ]
    )
    return {"minilm": (mini_df, mini_sum), "distilbert": (dist_df, dist_sum), "summary_table": cmp_tbl}


def sentence_embeddings(texts, model="minilm"):
    """
    backend: "minilm" | "distilbert" | "tfidf"
    Returns: np.ndarray [N, D] (L2-normalized), and a string label
    """
    texts = list(texts)
    label = model

    if model.lower() == "minilm" and _HAS_ST:
        emb = _ST_MODEL.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(emb), "minilm"

    if model.lower() == "distilbert":
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception:
            # fallback if transformers/torch not available
            X = _TFIDF.fit_transform(texts)
            Z = sk_normalize(X).toarray()
            return Z, "tfidf_fallback_distilbert_missing"

        tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        mdl = AutoModel.from_pretrained("distilbert-base-uncased")
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        print("Using device:", device)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl = mdl.to(device)
        mdl.eval()

        all_vecs = []
        bs = 64
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i : i + bs]
                inputs = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                out = mdl(**inputs)
                pooled = _mean_pool(out.last_hidden_state, inputs["attention_mask"])  # [B, H]
                all_vecs.append(pooled.cpu())
        Z = torch.cat(all_vecs, dim=0).numpy()
        # L2 normalize
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        return Z, "distilbert-meanpool"

    # TF-IDF (explicit request or fallback)
    X = _TFIDF.fit_transform(texts)
    Z = sk_normalize(X).toarray()
    return Z, "tfidf"


def mean_pairwise_cosine(emb):
    """
    emb: np.array [N,D] if ST; sparse if TF-IDF
    returns mean of upper-triangular pairwise cosine (excluding self-similarity)
    """
    if getattr(emb, "ndim", None) is None:  # sparse case (TF-IDF)
        sims = cosine_similarity(emb)
    else:
        sims = np.clip(np.dot(emb, emb.T), -1, 1)  # normalized ST embeddings
    n = sims.shape[0]
    if n < 2:
        return np.nan
    triu = sims[np.triu_indices(n, k=1)]
    return float(triu.mean())


# def intramotion_similarity(df):
#     df = ensure_parsed(df.copy())
#     results = []
#     for mid, sub in df.groupby("motion_id"):
#         emb = sentence_embeddings(sub["description"].tolist())
#         mpc = mean_pairwise_cosine(emb)
#         results.append({"motion_id": mid, "mean_intra_caption_sim": mpc, "num_captions": len(sub)})
#     out = pd.DataFrame(results)
#     summary = {
#         "mean_of_means": float(out["mean_intra_caption_sim"].mean(skipna=True)),
#         "median_of_means": float(out["mean_intra_caption_sim"].median(skipna=True)),
#         "quantiles": out["mean_intra_caption_sim"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
#     }
#     return out.sort_values("mean_intra_caption_sim", ascending=False), summary


# =========================
# 6) Simple clustering (optional)
# =========================


def cluster_captions(df, k=20, random_state=0):
    df = ensure_parsed(df.copy())
    emb = sentence_embeddings(df["description"].tolist())
    if _HAS_ST:
        X = emb
    else:
        X = emb  # sparse is fine for KMeans with scikit-learn's implementation (it will densify; watch RAM)
        if not hasattr(X, "toarray"):
            raise RuntimeError("Unexpected embedding type.")
        X = X.toarray()

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(X)
    df = df.assign(cluster=labels)
    top_terms = None
    if not _HAS_ST:
        # When using TF-IDF fallback, we can inspect cluster top terms by centroid weights
        centroids = kmeans.cluster_centers_
        terms = _TFIDF.get_feature_names_out()
        top_terms = {c: [terms[i] for i in centroids[c].argsort()[-10:][::-1]] for c in range(k)}
    return df, top_terms


# =========================
# 7) Tiny topic overview (optional, TF-IDF + top terms)
# =========================
def topic_overview(df, n_topics=10, top_k=12, random_state=0):
    """
    Not full LDA/BERTOPIC, just quick KMeans over TF-IDF to see coarse topics.
    Works even without sentence-transformers.
    """
    df = ensure_parsed(df.copy())
    X = _TFIDF.fit_transform(df["description"])
    km = KMeans(n_clusters=n_topics, n_init="auto", random_state=random_state).fit(X)
    terms = _TFIDF.get_feature_names_out()
    cent = km.cluster_centers_
    topics = {t: [terms[i] for i in cent[t].argsort()[-top_k:][::-1]] for t in range(n_topics)}
    assign = pd.DataFrame({"motion_id": df["motion_id"], "description": df["description"], "topic": km.labels_})
    return topics, assign


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


# =========================
# spaCy structural complexity
# =========================
def _dep_depth_for_token(tok):
    """Depth of a token in the dependency tree (root depth = 0)."""
    depth = 0
    cur = tok
    while cur.head is not cur:
        depth += 1
        cur = cur.head
    return depth


def spacy_structural_analysis(df, spacy_model="en_core_web_sm", max_docs=None, batch_size=256):
    """
    Computes:
      - avg_dep_depth: mean dependency depth per caption (avg over tokens)
      - avg_sentences_per_caption
      - avg_verbs_per_caption (from spaCy)
      - temporal/connective rate via tokens: then/while/before/after/and (spaCy tokens; case-insensitive)
    If spaCy/model is missing, returns a dict with 'available': False.
    """
    try:
        import spacy
    except Exception:
        return {"available": False, "reason": "spaCy not installed"}

    try:
        nlp = spacy.load(spacy_model, disable=["ner"])
    except Exception:
        # try to download on the fly if user permits; otherwise fail gracefully
        return {"available": False, "reason": f"spaCy model '{spacy_model}' not available"}

    texts = df["description"].tolist()
    if max_docs is not None:
        texts = texts[:max_docs]

    total_depth = 0.0
    total_tokens = 0
    total_sents = 0
    total_docs = 0
    verbs_per_cap = []
    conj_markers = {"and", "then", "while", "before", "after"}
    connective_hits = 0

    for doc in nlp.pipe(texts, batch_size=batch_size):
        if len(doc) == 0:
            continue
        depths = [_dep_depth_for_token(tok) for tok in doc]
        total_depth += sum(depths)
        total_tokens += len(doc)
        total_sents += len(list(doc.sents))
        total_docs += 1
        verbs_per_cap.append(sum(1 for t in doc if t.pos_ == "VERB"))
        connective_hits += sum(1 for t in doc if t.text.lower() in conj_markers)

    if total_docs == 0 or total_tokens == 0:
        return {"available": True, "note": "no tokens/docs", "avg_dep_depth": float("nan")}

    return {
        "available": True,
        "avg_dep_depth": total_depth / total_tokens,  # token-level mean
        "avg_sentences_per_caption": total_sents / total_docs,
        "avg_verbs_per_caption_spacy": float(np.mean(verbs_per_cap)),
        "connective_token_rate": connective_hits / total_tokens,  # fraction of tokens that are connectives
        "processed_docs": total_docs,
    }


# =========================
# Caption distance per motion
# =========================
import numpy as np


def _get_sentence_embeddings(texts):
    """
    Returns L2-normalized embeddings. Prefers sentence-transformers MiniLM;
    falls back to TF-IDF.
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        Z = model.encode(list(texts), batch_size=128, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(Z), "minilm"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        vec = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2))
        X = vec.fit_transform(texts)
        Z = normalize(X)  # L2 row-norm
        return Z.toarray(), "tfidf"


def _pairwise_stats_from_vectors(V, metric="cosine"):
    """
    Compute min/median/mean/max pairwise distance for a set of vectors V (n x d).
    metric='cosine' uses 1 - cosine_sim (for normalized vectors, cosine_sim = dot).
    """
    n = len(V)
    if n < 2:
        return dict(min=np.nan, median=np.nan, mean=np.nan, max=np.nan, n_pairs=0)
    # Cosine distance for normalized vectors
    S = np.clip(V @ V.T, -1.0, 1.0)  # similarity
    # take upper triangle distances
    iu = np.triu_indices(n, k=1)
    D = 1.0 - S[iu]
    return dict(
        min=float(np.min(D)),
        median=float(np.median(D)),
        mean=float(np.mean(D)),
        max=float(np.max(D)),
        n_pairs=len(D),
    )


def _train_vae_on_embeddings(E, latent_dim=16, epochs=10, batch_size=256, lr=1e-3, seed=0):
    """
    Tiny MLP VAE on embedding space E (n x d). Returns encoder that maps to mu (latent mean).
    If torch is unavailable, returns None.
    Designed to be quick; increase epochs for better fit.
    """
    try:
        import math

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        torch.manual_seed(seed)
    except Exception:
        return None, "torch_unavailable"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(E, dtype=torch.float32, device=device)
    n, d = X.shape

    h = 128

    class VAE(nn.Module):
        def __init__(self, d_in, d_lat=16):
            super().__init__()
            self.enc1 = nn.Linear(d_in, h)
            self.enc2 = nn.Linear(h, h)
            self.mu = nn.Linear(h, d_lat)
            self.logv = nn.Linear(h, d_lat)
            self.dec1 = nn.Linear(d_lat, h)
            self.dec2 = nn.Linear(h, d_in)

        def encode(self, x):
            z = F.relu(self.enc1(x))
            z = F.relu(self.enc2(z))
            return self.mu(z), self.logv(z)

        def reparam(self, mu, logv):
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            z = F.relu(self.dec1(z))
            return self.dec2(z)

        def forward(self, x):
            mu, logv = self.encode(x)
            z = self.reparam(mu, logv)
            xhat = self.decode(z)
            return xhat, mu, logv

    model = VAE(d, latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def loss_fn(x, xhat, mu, logv):
        # MSE recon + KL (standard normal prior)
        recon = F.mse_loss(xhat, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
        return recon + kl, recon, kl

    model.train()
    num_batches = int(np.ceil(n / batch_size))
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for bi in range(num_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            xb = X[idx]
            xhat, mu, logv = model(xb)
            loss, recon, kl = loss_fn(xb, xhat, mu, logv)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()

    @torch.no_grad()
    def encode_mu(x_np):
        xt = torch.tensor(x_np, dtype=torch.float32, device=device)
        mu, _ = model.encode(xt)
        return mu.cpu().numpy()

    return encode_mu, "ok"


def per_motion_distance_stats(df, use_vae=True, vae_latent_dim=16, vae_epochs=10):
    """
    For each motion_id:
      - embed all its captions
      - (optionally) map embeddings through a tiny VAE encoder (use mu as latent)
      - compute min/median/mean/max pairwise distances
    Returns:
      summary_df (motion_id, n_caps, min, median, mean, max),
      global_summary (min/median/mean/max across motions of 'mean' or 'median'—we report both)
    """
    df = ensure_parsed(df.copy())
    # Prepare embeddings for ALL captions once, then index by motion
    E, emb_kind = _get_sentence_embeddings(df["description"].tolist())

    # Optional tiny VAE on global embedding space (quick fit)
    encode_mu, vae_status = (None, "skipped")
    if use_vae:
        enc, vae_status = _train_vae_on_embeddings(E, latent_dim=vae_latent_dim, epochs=vae_epochs)
        if enc is not None:
            Z = enc(E)  # latent means
            z_kind = f"{emb_kind}+vae"
        else:
            Z = E  # fallback
            z_kind = f"{emb_kind}+no-vae"
    else:
        Z = E
        z_kind = emb_kind

    # map rows to vectors
    idx_map = {i: vec for i, vec in enumerate(Z)}
    rows = []
    for mid, sub in df.reset_index(drop=True).groupby("motion_id"):
        idxs = sub.index.tolist()
        V = np.stack([idx_map[i] for i in idxs], axis=0)
        stats = _pairwise_stats_from_vectors(V, metric="cosine")
        rows.append(
            {
                "motion_id": mid,
                "n_captions": len(idxs),
                "dist_min": stats["min"],
                "dist_median": stats["median"],
                "dist_mean": stats["mean"],
                "dist_max": stats["max"],
            }
        )
    out = pd.DataFrame(rows).sort_values("dist_mean", ascending=False)

    global_summary = {
        "embedding_space": z_kind,
        "vae_status": vae_status,
        "caption_distance_mean_stats": {
            "min": float(np.nanmin(out["dist_mean"])),
            "median": float(np.nanmedian(out["dist_mean"])),
            "mean": float(np.nanmean(out["dist_mean"])),
            "max": float(np.nanmax(out["dist_mean"])),
        },
        "caption_distance_median_stats": {
            "min": float(np.nanmin(out["dist_median"])),
            "median": float(np.nanmedian(out["dist_median"])),
            "mean": float(np.nanmean(out["dist_median"])),
            "max": float(np.nanmax(out["dist_median"])),
        },
    }
    return out, global_summary


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
    both = intramotion_similarity_compare(df)
    print("\n=== Backend comparison ===")
    print(both["summary_table"])

    print("\n\n")
    # E) (Optional) Quick clustering of captions
    clustered_df, top_terms = cluster_captions(df, k=20, random_state=0)
    print("\n=== Clustering ===")
    print(clustered_df["cluster"].value_counts().head())
    if top_terms is not None:
        for c, terms in list(top_terms.items())[:3]:
            print(f"Cluster {c} top terms: {terms}")

    print("\n\n")
    # F) (Optional) Topic overview
    topics, assign = topic_overview(df, n_topics=10, top_k=12)
    print("\n=== Topics (coarse) ===")
    for t, terms in list(topics.items())[:3]:
        print(f"Topic {t}: {terms}")

    print("\n\n")
    richness = lexical_richness_analysis(df)
    print("\n=== Linguistic Richness & Structural Variety ===")
    for k, v in richness.items():
        print(f"{k}: {v:.4f}")

    print("\n\n")
    # G) spaCy structural complexity
    spa = spacy_structural_analysis(df, spacy_model="en_core_web_sm", max_docs=None)
    print("\n=== spaCy Structural Complexity ===")
    if not spa.get("available", False):
        print("spaCy unavailable:", spa.get("reason"))
    else:
        for k, v in spa.items():
            if k == "available":
                continue
            print(f"{k}: {v}")

    print("\n\n")
    # H) Per-motion caption distances (MiniLM → VAE optional)
    dist_df, dist_summary = per_motion_distance_stats(df, use_vae=True, vae_latent_dim=16, vae_epochs=10)
    print("\n=== Caption Distance per Motion ===")
    print("Space:", dist_summary["embedding_space"], "| VAE:", dist_summary["vae_status"])
    print("Mean-distance stats:", dist_summary["caption_distance_mean_stats"])
    print("Median-distance stats:", dist_summary["caption_distance_median_stats"])
    print("Most diverse motions (top 5 by mean distance):")
    print(dist_df.head(5)[["motion_id", "n_captions", "dist_mean", "dist_median", "dist_max", "dist_min"]])
