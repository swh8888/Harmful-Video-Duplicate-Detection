"""Layer-2 text matching — v3 cross-medium 3x3 grid.

Replaces the v2 single `combined_text` embedding (which pooled all three
mediums into one vector and diluted a payload hiding in one medium). Each video
now keeps three medium embeddings (title / ocr / asr) in one shared space; a
query is compared against every seed across the full 3x3 medium grid and the
best cell is taken. That best cell yields the text_score AND which medium pair
matched, so a payload that moved mediums is still caught.

Implementation: all seed medium-vectors live in a single FAISS inner-product
index, tagged by (seed_id, medium). A query searches its three medium vectors
against that index; per seed we keep the best (query_medium, seed_medium) cell.
Empty medium texts are skipped so missing-vs-missing never scores as a match.
"""
import os
import numpy as np
import faiss

from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, MODELS_DIR

MEDIUMS = ["title", "ocr", "asr"]


def _default_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _clean(t):
    return t if isinstance(t, str) and t.strip() else None


class GridTextMatcher:
    def __init__(self, model=None):
        self.model = model if model is not None else _default_model()
        self.index = None
        self.row_seed = None      # parallel arrays: which seed / medium each index row is
        self.row_medium = None

    def _encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).astype(np.float32)

    def build_index(self, seed_df):
        """seed_df columns: video_id, title_text, ocr_text, asr_text."""
        texts, self.row_seed, self.row_medium = [], [], []
        for _, r in seed_df.iterrows():
            for m in MEDIUMS:
                t = _clean(r.get(f"{m}_text"))
                if t is None:
                    continue
                texts.append(t)
                self.row_seed.append(r["video_id"])
                self.row_medium.append(m)
        embs = self._encode(texts)
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embs)
        print(f"grid index: {self.index.ntotal} seed medium-vectors "
              f"({len(seed_df)} videos x up to 3 mediums)")
        return self

    def save_index(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        faiss.write_index(self.index, os.path.join(MODELS_DIR, "faiss_grid.index"))
        np.save(os.path.join(MODELS_DIR, "grid_row_seed.npy"), np.array(self.row_seed))
        np.save(os.path.join(MODELS_DIR, "grid_row_medium.npy"), np.array(self.row_medium))
        print("saved grid index + row maps")

    def load_index(self):
        self.index = faiss.read_index(os.path.join(MODELS_DIR, "faiss_grid.index"))
        self.row_seed = np.load(os.path.join(MODELS_DIR, "grid_row_seed.npy"), allow_pickle=True).tolist()
        self.row_medium = np.load(os.path.join(MODELS_DIR, "grid_row_medium.npy"), allow_pickle=True).tolist()
        return self

    def match(self, query_texts, top_n=None, top_k=None):
        """query_texts: dict {title, ocr, asr}. Returns a ranked list (best first)
        of candidate matches, each a dict:
            {seed_video_id, text_score, q_medium, c_medium,
             is_cross_medium, medium_transition}
        top_n=None returns ALL seeds ranked (gate-filtering is the caller's job);
        pass an int to cap. top_k is the FAISS search depth per medium and
        defaults to the full index so every seed's best cell is considered.
        """
        if not self.index or self.index.ntotal == 0:
            return []
        k = min(top_k or self.index.ntotal, self.index.ntotal)
        per_seed = {}  # seed_id -> best (score, q_medium, c_medium)
        for qm in MEDIUMS:
            qt = _clean(query_texts.get(f"{qm}_text"))
            if qt is None:
                continue
            qv = self._encode([qt])
            scores, idxs = self.index.search(qv, k)
            for score, ridx in zip(scores[0], idxs[0]):
                if ridx == -1:
                    continue
                sid = self.row_seed[ridx]
                cm = self.row_medium[ridx]
                s = float(max(0.0, min(1.0, score)))
                if sid not in per_seed or s > per_seed[sid][0]:
                    per_seed[sid] = (s, qm, cm)
        if not per_seed:
            return []
        ranked = sorted(per_seed.items(), key=lambda kv: kv[1][0], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return [
            {
                "seed_video_id": sid,
                "text_score": round(s, 4),
                "q_medium": qm,
                "c_medium": cm,
                "is_cross_medium": qm != cm,
                "medium_transition": f"{cm}->{qm}",
            }
            for sid, (s, qm, cm) in ranked
        ]
