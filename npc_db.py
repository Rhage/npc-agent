"""
npc_db.py
Database manager for the NPC Agent system.

Owns the SQLite connection and all queries. Nothing outside this file
should touch the database directly.

Usage:
    from npc_db import NPCDatabase
    db = NPCDatabase("npc_agent.db")
    npc = db.get_npc("Riko")
"""

import sqlite3
import os
from typing import Optional


# ── Schema ──────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS npcs (
    name              TEXT PRIMARY KEY,
    full_name         TEXT,
    aliases           TEXT,
    species           TEXT,
    gender            TEXT,
    age_description   TEXT,
    appearance        TEXT,
    role              TEXT,
    backstory         TEXT,
    personality       TEXT,
    speaking_style    TEXT,
    voice_language    TEXT,
    voice_description TEXT
);

CREATE TABLE IF NOT EXISTS npc_relationships (
    id          INTEGER PRIMARY KEY,
    npc_name    TEXT NOT NULL,
    target      TEXT NOT NULL,
    disposition TEXT,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(npc_name, target)
);

CREATE TABLE IF NOT EXISTS npc_beliefs (
    id              INTEGER PRIMARY KEY,
    npc_name        TEXT NOT NULL,
    label           TEXT NOT NULL,       -- human-readable name for DM (e.g. "The Merchants Guild")
    terms           TEXT NOT NULL DEFAULT '',
    subject         TEXT NOT NULL DEFAULT '',
    source          TEXT NOT NULL DEFAULT '',
    associations    TEXT NOT NULL DEFAULT '',
    knowledge       TEXT NOT NULL,       -- the belief content injected into context
    always_inject   BOOLEAN DEFAULT FALSE,
    emb_terms        BLOB,
    emb_subject      BLOB,
    emb_source       BLOB,
    emb_associations BLOB,
    emb_knowledge    BLOB NOT NULL,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


class NPCDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row  # rows accessible by column name
        self._conn.execute("PRAGMA journal_mode=WAL")  # safe for concurrent reads
        self._apply_schema()

    def _apply_schema(self):
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ── NPC queries ──────────────────────────────────────────────────────────

    def get_npc(self, name: str) -> Optional[dict]:
        """
        Fetch an NPC by canonical name or alias.
        Returns a dict of all fields, or None if not found.
        """
        row = self._conn.execute(
            "SELECT * FROM npcs WHERE name = ?", (name,)
        ).fetchone()

        if row:
            return dict(row)

        # Fall back to alias search
        all_npcs = self._conn.execute("SELECT * FROM npcs").fetchall()
        name_lower = name.lower()
        for npc in all_npcs:
            aliases = npc["aliases"] or ""
            alias_list = [a.strip().lower() for a in aliases.split(",") if a.strip()]
            if name_lower in alias_list:
                return dict(npc)

        return None

    def get_all_npcs(self) -> list[dict]:
        """Return all NPCs as a list of dicts."""
        rows = self._conn.execute("SELECT * FROM npcs ORDER BY name").fetchall()
        return [dict(r) for r in rows]

    def upsert_npc(self, npc: dict) -> None:
        """
        Insert or replace an NPC record.
        npc must contain at minimum a 'name' key.
        """
        fields = [
            "name", "full_name", "aliases", "species", "gender",
            "age_description", "appearance", "role", "backstory",
            "personality", "speaking_style", "voice_language", "voice_description"
        ]
        columns      = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        values       = tuple(npc.get(f) for f in fields)

        self._conn.execute(
            f"INSERT OR REPLACE INTO npcs ({columns}) VALUES ({placeholders})",
            values
        )
        self._conn.commit()

    def delete_npc(self, name: str) -> bool:
        """Delete an NPC by canonical name. Returns True if a row was deleted."""
        cursor = self._conn.execute("DELETE FROM npcs WHERE name = ?", (name,))
        self._conn.commit()
        return cursor.rowcount > 0

    def npc_exists(self, name: str) -> bool:
        """True if an NPC with this canonical name exists."""
        row = self._conn.execute(
            "SELECT 1 FROM npcs WHERE name = ?", (name,)
        ).fetchone()
        return row is not None

    # ── Relationship queries ─────────────────────────────────────────────────

    def get_relationships(self, npc_name: str) -> list[dict]:
        """Return all relationships for an NPC, ordered by target name."""
        rows = self._conn.execute(
            "SELECT * FROM npc_relationships WHERE npc_name = ? ORDER BY target",
            (npc_name,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_relationship(self, npc_name: str, target: str) -> Optional[dict]:
        """Return a single relationship, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM npc_relationships WHERE npc_name = ? AND target = ?",
            (npc_name, target)
        ).fetchone()
        return dict(row) if row else None

    def upsert_relationship(self, npc_name: str, target: str, disposition: str) -> None:
        """Insert or update a relationship disposition."""
        self._conn.execute(
            """
            INSERT INTO npc_relationships (npc_name, target, disposition, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(npc_name, target) DO UPDATE SET
                disposition = excluded.disposition,
                updated_at  = CURRENT_TIMESTAMP
            """,
            (npc_name, target, disposition)
        )
        self._conn.commit()

    def delete_relationship(self, npc_name: str, target: str) -> bool:
        """Delete a relationship. Returns True if a row was deleted."""
        cursor = self._conn.execute(
            "DELETE FROM npc_relationships WHERE npc_name = ? AND target = ?",
            (npc_name, target)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ── Belief queries ───────────────────────────────────────────────────────

    def get_beliefs(self, npc_name: str) -> list[dict]:
        """Return all beliefs for an NPC including embedding blobs."""
        rows = self._conn.execute(
            """SELECT id, npc_name, label, terms, subject, source, associations,
                      knowledge, always_inject,
                      emb_terms, emb_subject, emb_source, emb_associations, emb_knowledge
               FROM npc_beliefs WHERE npc_name = ?
               ORDER BY always_inject DESC, label""",
            (npc_name,)
        ).fetchall()
        return [dict(r) for r in rows]

    def upsert_belief(self, belief: dict) -> int:
        """
        Insert or update a belief record.
        belief must contain: npc_name, label, knowledge, emb_knowledge.
        Optional: terms, subject, source, associations, always_inject,
                  emb_terms, emb_subject, emb_source, emb_associations.
        Returns the row id.
        """
        fields = [
            "npc_name", "label", "terms", "subject", "source", "associations",
            "knowledge", "always_inject",
            "emb_terms", "emb_subject", "emb_source", "emb_associations", "emb_knowledge",
        ]
        columns      = ", ".join(fields)
        placeholders = ", ".join("?" for _ in fields)
        values       = tuple(belief.get(f) for f in fields)

        cursor = self._conn.execute(
            f"INSERT OR REPLACE INTO npc_beliefs ({columns}, updated_at) "
            f"VALUES ({placeholders}, CURRENT_TIMESTAMP)",
            values
        )
        self._conn.commit()
        return cursor.lastrowid

    def delete_belief(self, belief_id: int) -> bool:
        """Delete a belief by id. Returns True if a row was deleted."""
        cursor = self._conn.execute(
            "DELETE FROM npc_beliefs WHERE id = ?", (belief_id,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self):
        self._conn.close()
