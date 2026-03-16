"""
prompts.py
Global prompt configuration and system prompt assembly.

Response format instructions live here as constants — they apply to all
NPCs regardless of character definition.

build_system_prompt() assembles the static system prompt from an NPC record.
build_context_block() assembles the dynamic per-message context block from
current visibility and relationship data. This is injected per-call but
never stored in conversation history.
"""

# ── Global response format instructions ─────────────────────────────────────

RESPONSE_FORMAT = (
    "Respond only with spoken dialog — no action descriptions, no stage directions, "
    "no asterisks. "
    "Keep your response brief: one to three sentences unless the situation demands more. "
    "Do not break character under any circumstances."
)


# ── Static system prompt ─────────────────────────────────────────────────────

def build_system_prompt(npc: dict) -> str:
    """
    Assemble the static system prompt for an NPC.
    Built once at conversation init and never changed.

    Args:
        npc: NPC record dict from NPCDatabase.get_npc()

    Returns:
        A complete system prompt string ready to pass to the LLM.
    """
    parts = []

    identity = _build_identity(npc)
    if identity:
        parts.append(identity)

    if npc.get("personality"):
        parts.append(f"Personality: {npc['personality']}")

    if npc.get("speaking_style"):
        parts.append(f"Speaking style: {npc['speaking_style']}")

    parts.append(RESPONSE_FORMAT)

    return "\n\n".join(parts)


# ── Dynamic context block ─────────────────────────────────────────────────────

def build_context_block(
    npc_name:        str,
    speaker:         str,
    visible_actors:  list[str],
    db
) -> str:
    """
    Assemble the dynamic context block for a single LLM call.
    Injected per-message, never stored in conversation history.

    Always includes the current speaker's relationship.
    Includes relationships for all visible actors.

    Args:
        npc_name:       Canonical NPC name.
        speaker:        Name of the player character sending this message.
        visible_actors: List of actor names currently visible to the NPC token.
        db:             NPCDatabase instance.

    Returns:
        Formatted context string, or empty string if nothing to inject.
    """
    parts = []

    # ── Current speaker relationship ──
    speaker_rel = db.get_relationship(npc_name, speaker)
    if speaker_rel and speaker_rel.get("disposition"):
        parts.append(f"Your relationship with {speaker} (current speaker): "
                     f"{speaker_rel['disposition']}")

    # ── Visible entities (excluding speaker, already handled above) ──
    visible_others = [a for a in visible_actors if a != speaker and a != npc_name]
    if visible_others:
        rel_lines = []
        for actor_name in sorted(visible_others):
            rel = db.get_relationship(npc_name, actor_name)
            if rel and rel.get("disposition"):
                rel_lines.append(f"- {actor_name}: {rel['disposition']}")
            else:
                rel_lines.append(f"- {actor_name}: Someone you have no strong feelings about.")

        if rel_lines:
            parts.append("Others currently present:\n" + "\n".join(rel_lines))

    if not parts:
        return ""

    return "[Current scene context]\n" + "\n\n".join(parts)


# ── Identity builder ─────────────────────────────────────────────────────────

def _build_identity(npc: dict) -> str:
    name            = npc.get("name", "this character")
    full_name       = npc.get("full_name")
    species         = npc.get("species")
    gender          = npc.get("gender")
    age_description = npc.get("age_description")
    role            = npc.get("role")
    appearance      = npc.get("appearance")
    backstory       = npc.get("backstory")

    lines = []

    descriptor_parts = []
    if age_description:
        descriptor_parts.append(age_description)
    if gender:
        descriptor_parts.append(gender)
    if species:
        descriptor_parts.append(species)

    descriptor  = " ".join(descriptor_parts) if descriptor_parts else "character"
    name_clause = f"{name}" + (f" ({full_name})" if full_name and full_name != name else "")
    role_clause = f" and {role}" if role else ""
    lines.append(f"You are {name_clause}, a {descriptor}{role_clause}.")

    if appearance:
        lines.append(f"Appearance: {appearance}")

    if backstory:
        lines.append(f"Background: {backstory}")

    return "\n".join(lines)
