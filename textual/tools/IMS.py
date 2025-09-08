import os
import tools.utility as th_utility
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses

os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_ST = True
    _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error ({e})\n\n sentence-transformers not available, falling back to TF-IDF for similarity.")
    exit(1)
    from sklearn.metrics.pairwise import cosine_similarity

    _HAS_ST = False

from sklearn.feature_extraction.text import TfidfVectorizer

_TFIDF = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1, 2))


class IntraMotionSimilarity:
    def __init__(self):
        self.ensure_parsed = th_utility.ensure_parsed

    def intramotion_similarity(self, df, model="minilm"):
        df = self.ensure_parsed(df.copy())
        Z, label = self.sentence_embeddings(df, model=model)
        # maping row idx -> vector
        idx_to_vec = {i: Z[i] for i in range(len(Z))}

        rows = []
        for mid, sub in df.reset_index(drop=True).groupby("motion_id"):
            idxs = sub.index.tolist()
            V = np.stack([idx_to_vec[i] for i in idxs], axis=0)
            mpc = self._mean_pairwise_cosine_from_normed(V)
            rows.append({"motion_id": mid, "mean_intra_caption_sim": mpc, "num_captions": len(idxs)})

        out = pd.DataFrame(rows).sort_values("mean_intra_caption_sim", ascending=False)
        summary = {
            "backend": label,
            "mean_of_means": float(out["mean_intra_caption_sim"].mean(skipna=True)),
            "median_of_means": float(out["mean_intra_caption_sim"].median(skipna=True)),
            "quantiles": out["mean_intra_caption_sim"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        }
        return out, summary

    def _mean_pool(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        return summed / counts

    def _mean_pairwise_cosine_from_normed(self, V):
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

    def intramotion_similarity_compare(self, df):
        print("\n=== Intra-motion Similarity (MiniLM) ===")
        mini_df, mini_sum = self.intramotion_similarity(df, model="minilm")
        print(mini_sum)
        print("Top 10 most consistent motions (MiniLM):")
        print(mini_df.head(10)[["motion_id", "mean_intra_caption_sim", "num_captions"]])
        mini_df.sort_values("mean_intra_caption_sim", ascending=True, inplace=True)
        print("Top 10 least consistent motions (MiniLM):")
        print(mini_df.head(10)[["motion_id", "mean_intra_caption_sim", "num_captions"]])

        print("\n=== Intra-motion Similarity (mpnet) ===")
        mpnet_df, mpnet_sum = self.intramotion_similarity(df, model="mpnet")
        print(mpnet_sum)
        print("Top 10 most consistent motions (mpnet):")
        print(mpnet_df.head(10)[["motion_id", "mean_intra_caption_sim", "num_captions"]])
        mpnet_df.sort_values("mean_intra_caption_sim", ascending=True, inplace=True)
        print("Top 10 least consistent motions (mpnet):")
        print(mpnet_df.head(10)[["motion_id", "mean_intra_caption_sim", "num_captions"]])

        print("\n=== Intra-motion Similarity (DistilBERT-uncased) ===")
        dist_df, dist_sum = self.intramotion_similarity(df, model="distilbert")
        print(dist_sum)
        print("Top 10 most consistent motions (DistilBERT):")
        print(dist_df.head(10)[["motion_id", "mean_intra_caption_sim", "num_captions"]])
        dist_df.sort_values("mean_intra_caption_sim", ascending=True, inplace=True)
        print("Top 10 least consistent motions (DistilBERT):")
        print(dist_df.head(10)[["motion_id", "mean_intra_caption_sim", "num_captions"]])

        # Side-by-side summary table
        cmp_tbl = pd.DataFrame(
            [
                {"backend": mini_sum["backend"], **{k: v for k, v in mini_sum.items() if k != "backend"}},
                {"backend": dist_sum["backend"], **{k: v for k, v in dist_sum.items() if k != "backend"}},
            ]
        )
        return {"minilm": (mini_df, mini_sum), "mpnet": (mpnet_df, mpnet_sum), "distilbert": (dist_df, dist_sum), "summary_table": cmp_tbl}

    def sentence_embeddings(self, df, model="minilm"):
        """
        backend: "minilm" | "distilbert" | "tfidf"
        Returns: np.ndarray [N, D] (L2-normalized), and a string label
        """

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        print("Using device:", device)
        texts = list(df["description"].tolist())
        label = model

        if model.lower() == "minilm" and _HAS_ST:
            emb = _ST_MODEL.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(emb), "minilm"

        if model.lower() == "mpnet" and _HAS_ST:
            model = SentenceTransformer("sentence-transformers/all-MPNet-base-v2")
            pairs = th_utility.build_same_motion_pairs(df)
            loader = DataLoader(pairs, batch_size=64, shuffle=True, drop_last=True)
            loss = losses.MultipleNegativesRankingLoss(model)
            model.fit([(loader, loss)], epochs=2, warmup_steps=int(0.1 * len(loader) * 2), optimizer_params={"lr": 2e-5})
            emb = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(emb), "mpnet"

        if model.lower() == "distilbert":
            try:

                from transformers import AutoModel, AutoTokenizer
            except Exception:
                # fallback if transformers/torch not available
                X = _TFIDF.fit_transform(texts)
                Z = sk_normalize(X).toarray()
                return Z, "tfidf_fallback_distilbert_missing"

            tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            mdl = AutoModel.from_pretrained("distilbert-base-uncased")

            mdl = mdl.to(device)
            mdl.eval()

            all_vecs = []
            bs = 128
            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    batch = texts[i : i + bs]
                    inputs = tok(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                    out = mdl(**inputs)
                    pooled = self._mean_pool(out.last_hidden_state, inputs["attention_mask"])
                    all_vecs.append(pooled.cpu())
            Z = torch.cat(all_vecs, dim=0).numpy()
            # L2 normalize
            Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
            return Z, "distilbert-meanpool"

        X = _TFIDF.fit_transform(texts)
        Z = sk_normalize(X).toarray()
        return Z, "tfidf"

    def mean_pairwise_cosine(self, emb):
        """
        emb: np.array [N,D] if ST; sparse if TF-IDF
        returns mean of upper-triangular pairwise cosine (excluding self-similarity)
        """
        if getattr(emb, "ndim", None) is None:
            sims = cosine_similarity(emb)
        else:
            sims = np.clip(np.dot(emb, emb.T), -1, 1)
        n = sims.shape[0]
        if n < 2:
            return np.nan
        triu = sims[np.triu_indices(n, k=1)]
        return float(triu.mean())
