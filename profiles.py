"""
profiles.py
NPC profile access — database-backed.

Public interface is unchanged from the file-based version so nothing
else in the system needs to change.
"""

from npc_db import NPCDatabase
from prompts import build_system_prompt
from config import DB_PATH

# Module-level database instance — shared across all callers
_db: NPCDatabase | None = None

def _get_db() -> NPCDatabase:
    global _db
    if _db is None:
        _db = NPCDatabase(DB_PATH)
    return _db


def load_profile(profile_name: str) -> dict:
    """
    Load an NPC profile by name or alias.
    Raises FileNotFoundError (for compatibility) if not found.
    """
    npc = _get_db().get_npc(profile_name)
    if npc is None:
        raise FileNotFoundError(
            f"No NPC found with name or alias '{profile_name}'. "
            f"Run migrate_profiles.py if you haven't already."
        )
    return npc


def build_system_prompt_for(profile_name: str, context: str = "") -> str:
    """
    Convenience function — load profile and build system prompt in one call.
    """
    npc = load_profile(profile_name)
    return build_system_prompt(npc, context)
