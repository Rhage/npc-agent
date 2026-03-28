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

# ── Response format — the performance instruction ────────────────────────────
# Applied to the response call only.

RESPONSE_FORMAT = (
    "You are an actor performing this character. "
    "Deliver only your character's spoken dialog — no action descriptions, "
    "no stage directions, no asterisks. "
    "Keep your response brief: one to three sentences unless the situation demands more. "
    "Do not break character under any circumstances. "
    "You will receive a director's note in <direction> tags if scene guidance applies — "
    "incorporate it naturally without acknowledging it directly."
)

# ── Thinking prompt — the preparation step ───────────────────────────────────
# Applied to the thinking call only.
# The result is passed to the response call as a <thinking> block, never stored.

ORIGINAL_THINKING_PROMPT = (
    "You are an actor preparing to perform a scene. "
    "Before delivering your line, think through the following privately:\n\n"
    "- Who are you speaking with right now, and how do you feel about them?\n"
    "- Who else is physically present in this scene?\n"
    "- What does your character actually know that is relevant to this moment?\n"
    "- What does your character NOT know — and therefore should not mention or imply?\n"
    "- What is your character's emotional state and motivation in this moment?\n"
    "- What would your character naturally say, and what would they hold back?\n\n"
    "Be concise. This is your private preparation — it will not be seen by anyone. "
    "Do not write dialog here."
)

THINKING_PROMPT = (
    "You are an acting coach helping an actor prepare for their next line in an ongoing scene. "
    "Address your actor directly by name. Give them brief, specific guidance about:\n"
    "- Who they are speaking with and how their character feels about that person\n"
    "- Who else is physically present and how to factor that in\n"
    "- What their character knows and crucially what they do NOT know\n"
    "- Their character's emotional state and motivation right now\n"
    "Be concise and direct. You are giving verbal notes, not writing the performance."
)

THINKING_EXAMPLE_INPUT = (
    "Current speaker: Horgrim\nHorgrim says: Have you heard anything about the merchant guild?"
)

THINKING_EXAMPLE_OUTPUT = (
    "Your character dislikes Horgrim — keep your tone cool and guarded with him. "
    "Garren is present but not speaking; your character trusts him, so you can be slightly "
    "more relaxed in body language but focus on Horgrim since he's addressing you. "
    "Your character has heard a rumor that the guild is run by a catgirl — she believes this "
    "strongly but has no hard facts. She would be suspicious and dismissive about the guild "
    "without revealing how much she actually cares. She would not mention the catgirl rumor "
    "unless pressed — that feels too personal to volunteer unprompted. "
    "Hold back: don't speculate beyond what she actually knows."
)

# ── Static system prompt ─────────────────────────────────────────────────────

def build_system_prompt(npc: dict) -> str:
    """
    Assemble the static system prompt for an NPC.
    Built once at conversation init. Uses actor framing.

    Args:
        npc: NPC record dict from NPCDatabase.get_npc()
    """
    parts = []

    # ── Actor framing + identity ──
    identity = _build_identity(npc)
    if identity:
        parts.append(identity)

    if npc.get("personality"):
        parts.append(f"Character personality: {npc['personality']}")

    if npc.get("speaking_style"):
        parts.append(f"Character speaking style: {npc['speaking_style']}")

    # ── Context tag explanation (in system prompt so model knows the structure) ──
    parts.append(
        "During the scene you will receive context in tagged sections:\n"
        "<scene> — who is physically present right now\n"
        "<knowledge> — what your character knows about topics and people\n"
        "<relationships> — how your character feels about specific individuals\n"
        "<thinking> — your own private preparation notes from before this line\n"
        "Use all sections to inform your performance. "
        "Do not reference the tags directly in your dialog."
    )

    parts.append(RESPONSE_FORMAT)

    return "\n\n".join(parts)


# ── Dynamic context block ─────────────────────────────────────────────────────

def build_context_block(
    npc_name:        str,
    speaker:         str,
    visible_actors:  list[str],
    db,
    knowledge_mgr    = None,
    query:           str = "",
) -> str:
    """
    Assemble the dynamic context block for a single LLM call.
    Injected per-message, never stored in conversation history.

    Uses XML-style tags to give the model clear structural boundaries
    between physical reality, knowledge, and relationships — reducing
    context bleed between sections.
    """
    sections = []

    # ── <scene> — physical reality right now ────────────────────────────────
    visible_others = [
        a for a in visible_actors
        if a and a != speaker and a != npc_name
    ]

    scene_lines = [f"You are currently speaking with {speaker}."]
    if visible_others:
        others_list = ", ".join(visible_others)
        scene_lines.append(
            f"Also physically present in this scene: {others_list}. "
            f"These individuals are in the same room as you right now."
        )
    else:
        scene_lines.append("No one else is present in this scene.")

    sections.append("<scene>\n" + "\n".join(scene_lines) + "\n</scene>")

    # ── <knowledge> — what this NPC knows ───────────────────────────────────
    if knowledge_mgr is not None:
        retrieval = knowledge_mgr.retrieve(
            npc_name       = npc_name,
            query          = query,
            visible_actors = visible_actors,
            db             = db,
        )

        belief_lines = []
        for b in retrieval["always"] + retrieval["semantic"] + retrieval["presence"]:
            belief_lines.append(f"- {b['label']}: {b['knowledge']}")

        ignorance_lines = [
            f"- You have no knowledge of {name} — do not invent or speculate."
            for name in retrieval["no_knowledge"]
        ]

        if belief_lines or ignorance_lines:
            knowledge_content = []
            if belief_lines:
                knowledge_content.append("\n".join(belief_lines))
            if ignorance_lines:
                knowledge_content.append(
                    "Things you have no knowledge of:\n" + "\n".join(ignorance_lines)
                )
            sections.append(
                "<knowledge>\n" + "\n\n".join(knowledge_content) + "\n</knowledge>"
            )

    # ── <relationships> — how you feel about people ──────────────────────────
    rel_lines = []

    speaker_rel = db.get_relationship(npc_name, speaker)
    if speaker_rel and speaker_rel.get("disposition"):
        rel_lines.append(f"- {speaker} (speaking to you now): {speaker_rel['disposition']}")
    else:
        rel_lines.append(f"- {speaker} (speaking to you now): Someone you have no strong feelings about.")

    for actor_name in sorted(visible_others):
        rel = db.get_relationship(npc_name, actor_name)
        if rel and rel.get("disposition"):
            rel_lines.append(f"- {actor_name} (present): {rel['disposition']}")
        else:
            rel_lines.append(f"- {actor_name} (present): Someone you have no strong feelings about.")

    if rel_lines:
        sections.append(
            "<relationships>\n" + "\n".join(rel_lines) + "\n</relationships>"
        )

    if not sections:
        return ""

    return "\n\n".join(sections)


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

    # Actor framing — the model is playing a role, not being the character
    lines.append(
        f"You are an actor performing the role of {name_clause}, "
        f"a {descriptor}{role_clause}. "
        f"Stay true to this character in every response."
    )

    if appearance:
        lines.append(f"Character appearance: {appearance}")

    if backstory:
        lines.append(f"Character background: {backstory}")

    return "\n".join(lines)
