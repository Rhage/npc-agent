"""
knowledge_manager.py
Owns the sentence-transformer embedding model, per-NPC belief caches,
and semantic retrieval logic.

The embedding model is loaded once at server startup and shared across
all NPC conversations. Per-NPC embedding caches are built lazily on first
conversation init — deserializing pre-computed blobs from the database,
which takes milliseconds.

Retrieval per message:
  1. Always-inject beliefs bypass scoring entirely.
  2. Semantic search against the player message text (tiered scoring).
  3. Presence pass — search visible actor names as secondary queries,
     surfacing beliefs about present entities even if not directly mentioned.
  4. Explicit ignorance injection for visible actors with no surfaced belief.
"""

import numpy as np
from typing import Optional

# ── Relevance tiers ──────────────────────────────────────────────────────────
# Topics are classified by their highest-scoring field into a tier.
# "irrelevant" topics are excluded from output entirely.

TIERS = [
    ("strong",     0.55),
    ("good",       0.45),
    ("weak",       0.30),
    ("very_weak",  0.15),
    ("irrelevant", 0.0),
]

# ── Upgrade rules ────────────────────────────────────────────────────────────
# A topic can be promoted one tier upward if enough fields meet a threshold.
# Format: tier_name -> (min_field_count, min_field_score)

UPGRADE_RULES = {
    "weak": (3, 0.40),  # 3+ fields >= 0.40 upgrades weak  -> good
    "good": (3, 0.45),  # 3+ fields >= 0.45 upgrades good  -> strong
}

# Tiers considered relevant enough to inject into context
#INJECT_TIERS = {"strong", "good", "weak"}
INJECT_TIERS = {"strong", "good"}

# Minimum tier for presence pass (stricter than general retrieval)
PRESENCE_MIN_TIER = "strong"

ALL_FIELDS = ("terms", "subject", "source", "associations", "knowledge")


class KnowledgeManager:
    def __init__(self, model_name: str):
        """
        Load the sentence-transformer model. Call once at server startup.
        model_name: e.g. "multi-qa-mpnet-base-dot-v1"
        """
        print(f"[KnowledgeManager] Loading embedding model '{model_name}'...")
        from sentence_transformers import SentenceTransformer
        self._model   = SentenceTransformer(model_name)
        self._cache: dict[str, list[dict]] = {}  # npc_name -> cached belief dicts
        print("[KnowledgeManager] Model ready.")

    # ── Cache management ─────────────────────────────────────────────────────

    def load_npc(self, npc_name: str, db) -> int:
        """
        Load and cache embeddings for an NPC from the database.
        Safe to call multiple times — skips if already cached.
        Returns number of beliefs loaded.
        """
        if npc_name in self._cache:
            return len(self._cache[npc_name])

        beliefs = db.get_beliefs(npc_name)
        cached  = []
        for b in beliefs:
            entry = {
                "belief_id":     b["id"],
                "label":         b["label"],
                "knowledge":     b["knowledge"],
                "always_inject": bool(b["always_inject"]),
                "emb_terms":        self._blob_to_vec(b["emb_terms"])        if b["emb_terms"]        else None,
                "emb_subject":      self._blob_to_vec(b["emb_subject"])      if b["emb_subject"]      else None,
                "emb_source":       self._blob_to_vec(b["emb_source"])       if b["emb_source"]       else None,
                "emb_associations": self._blob_to_vec(b["emb_associations"]) if b["emb_associations"] else None,
                "emb_knowledge":    self._blob_to_vec(b["emb_knowledge"]),
            }
            cached.append(entry)

        self._cache[npc_name] = cached
        print(f"[KnowledgeManager] Loaded {len(cached)} belief(s) for '{npc_name}'.")
        return len(cached)

    def invalidate(self, npc_name: str):
        """Remove an NPC's cache — call after beliefs are updated in the DB."""
        self._cache.pop(npc_name, None)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        npc_name:       str,
        query:          str,
        visible_actors: list[str],
        db,
    ) -> dict:
        """
        Retrieve relevant beliefs for context injection.

        Returns a dict with keys:
          always:    list of always-inject belief dicts
          semantic:  list of semantically relevant belief dicts (tiered)
          presence:  list of belief dicts surfaced by visible actor names
          no_knowledge: list of visible actor names with no surfaced belief
        """
        self.load_npc(npc_name, db)
        beliefs = self._cache.get(npc_name, [])

        always_ids  = set()
        semantic_ids = set()
        presence_ids = set()

        always   = []
        semantic = []
        presence = []

        # ── 1. Always-inject ──
        for b in beliefs:
            if b["always_inject"]:
                always.append(b)
                always_ids.add(b["belief_id"])

        # ── 2. Semantic search against player message ──
        if query.strip() and beliefs:
            query_vec = self._encode(query)
            for b in beliefs:
                if b["belief_id"] in always_ids:
                    continue
                result = self._classify(query_vec, b)
                if result["tier"] in INJECT_TIERS:
                    semantic.append({**b, **result})
                    semantic_ids.add(b["belief_id"])

            # Sort by tier then peak score
            tier_order = {t[0]: i for i, t in enumerate(TIERS)}
            semantic.sort(key=lambda x: (tier_order.get(x["tier"], 99), -x["peak"]))

        # ── 3. Presence pass — visible actor names ──
        surfaced_ids = always_ids | semantic_ids
        for actor_name in visible_actors:
            if not actor_name or actor_name == npc_name:
                continue
            actor_vec = self._encode(actor_name)
            best      = None
            best_peak = 0.0
            for b in beliefs:
                if b["belief_id"] in surfaced_ids:
                    continue
                result = self._classify(actor_vec, b)
                if result["tier"] == PRESENCE_MIN_TIER and result["peak"] > best_peak:
                    best      = {**b, **result}
                    best_peak = result["peak"]
            if best:
                presence.append(best)
                presence_ids.add(best["belief_id"])

        # ── 4. No-knowledge — visible actors with no surfaced belief ──
        all_surfaced = always_ids | semantic_ids | presence_ids
        covered_labels = {b["label"].lower() for b in always + semantic + presence}

        no_knowledge = []
        for actor_name in visible_actors:
            if not actor_name or actor_name == npc_name:
                continue
            if actor_name.lower() not in covered_labels:
                # Check if any surfaced belief mentions this actor in its label
                mentioned = any(
                    actor_name.lower() in b["label"].lower()
                    for b in always + semantic + presence
                )
                if not mentioned:
                    no_knowledge.append(actor_name)

        return {
            "always":       always,
            "semantic":     semantic,
            "presence":     presence,
            "no_knowledge": no_knowledge,
        }

    # ── Embedding helpers ────────────────────────────────────────────────────

    def encode_belief(self, belief: dict) -> dict:
        """
        Encode all text fields of a belief dict into embedding blobs.
        Returns a dict of field -> blob for storage in the database.
        """
        blobs = {}
        for field in ("terms", "subject", "source", "associations"):
            text = belief.get(field, "").strip()
            blobs[f"emb_{field}"] = self._vec_to_blob(self._encode(text)) if text else None
        blobs["emb_knowledge"] = self._vec_to_blob(self._encode(belief["knowledge"]))
        return blobs

    def _encode(self, text: str) -> np.ndarray:
        return self._model.encode(text)

    @staticmethod
    def _vec_to_blob(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def _blob_to_vec(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))

    def _classify(self, query_vec: np.ndarray, belief: dict) -> dict:
        """Classify a belief into a relevance tier against a query vector."""
        field_scores = {
            field: self._cosine(query_vec, belief[f"emb_{field}"])
            for field in ALL_FIELDS
            if belief.get(f"emb_{field}") is not None
        }

        peak      = max(field_scores.values()) if field_scores else 0.0
        base_tier = "irrelevant"
        for name, floor in TIERS:
            if peak >= floor:
                base_tier = name
                break

        # Apply upgrade rule
        upgraded   = False
        final_tier = base_tier
        if base_tier in UPGRADE_RULES:
            count_needed, floor_needed = UPGRADE_RULES[base_tier]
            qualifying = sum(1 for s in field_scores.values() if s >= floor_needed)
            if qualifying >= count_needed:
                tier_names  = [t[0] for t in TIERS]
                current_idx = tier_names.index(base_tier)
                if current_idx > 0:
                    final_tier = tier_names[current_idx - 1]
                    upgraded   = True

        return {
            "tier":         final_tier,
            "base_tier":    base_tier,
            "upgraded":     upgraded,
            "peak":         peak,
            "field_scores": field_scores,
        }
