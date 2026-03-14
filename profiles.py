"""
profiles.py
Handles loading NPC profiles and assembling system prompts.

Profile directory structure:
    profiles/
        <ProfileName>/
            profile.json   — permanent identity, authored by GM, rarely changes
            memory.json    — current knowledge state, updated between sessions

profile.json schema:
    name            — display name, must match Foundry actor name exactly
    identity
        description — who this NPC is at their core
        personality — how they behave and make decisions
        speech_style — formatting and delivery instructions for the LLM
    knowledge
        self        — what they know about their own life and history
        location    — what they know about their immediate environment
        local       — what they know about the surrounding area
        world       — what they know about the wider world beyond their area
    people          — dict of known individuals keyed by name or description
    beliefs         — list of priors that shape how they interpret events
    secrets         — things they know but would never volunteer
    voice           — optional TTS configuration block

memory.json schema:
    people          — dict of individuals met through interaction
        <name>
            impression  — current overall impression
            trust       — neutral / friendly / wary / hostile
            notes       — list of specific remembered details
    facts           — list of specific things learned through interaction
        about       — what the fact concerns
        knows       — the fact itself, as the NPC understands it
        confidence  — certain / believes / suspects
    beliefs         — beliefs formed or updated through experience
                      (supplements profile beliefs, does not replace them)
"""

import json
import os


def load_profile(profile_name: str) -> dict:
    """
    Load a profile from profiles/<ProfileName>/profile.json.
    Raises FileNotFoundError if the directory or file does not exist.
    """
    path = os.path.join("profiles", profile_name, "profile.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No profile found for '{profile_name}' at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_memory(profile_name: str) -> dict:
    """
    Load the current memory state for a profile.
    Returns an empty memory structure if no memory file exists yet.
    """
    path = os.path.join("profiles", profile_name, "memory.json")
    if not os.path.exists(path):
        return {"people": {}, "facts": [], "beliefs": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[Profiles] Failed to load memory for '{profile_name}': {e}")
        return {"people": {}, "facts": [], "beliefs": []}


def build_system_prompt(profile: dict, memory: dict = None, context: dict = None) -> str:
    """
    Assemble the full system prompt for an NPC from three sources:

    1. profile.json  — permanent identity and knowledge (always present)
    2. memory.json   — current knowledge state from past interactions (if any)
    3. context       — dynamic Foundry context injected fresh each call (if provided)

    Context is intentionally never saved to history — it is rebuilt fresh on
    every LLM call so saved conversations never contain stale information.

    Clear section headers prevent smaller LLMs from conflating information
    across categories.
    """
    parts = []

    identity  = profile.get("identity",  {})
    knowledge = profile.get("knowledge", {})

    # ── Core identity ──
    name = profile.get("name", "")
    if identity.get("description"):
        parts.append(f"You are {name}. {identity['description']}")
    else:
        parts.append(f"You are {name}.")

    if identity.get("personality"):
        parts.append(identity["personality"])

    if identity.get("speech_style"):
        parts.append(identity["speech_style"])

    # ── Layered knowledge ──
    if knowledge.get("self"):
        parts.append(f"\nWHAT YOU KNOW ABOUT YOURSELF:\n{knowledge['self']}")

    if knowledge.get("location"):
        parts.append(f"\nWHAT YOU KNOW ABOUT YOUR IMMEDIATE SURROUNDINGS:\n{knowledge['location']}")

    if knowledge.get("local"):
        parts.append(f"\nWHAT YOU KNOW ABOUT THE LOCAL AREA:\n{knowledge['local']}")

    if knowledge.get("world"):
        parts.append(f"\nWHAT YOU KNOW ABOUT THE WIDER WORLD:\n{knowledge['world']}")

    # ── People the NPC already knows (from profile) ──
    profile_people = profile.get("people", {})
    if profile_people:
        lines = ["\nPEOPLE YOU KNOW:"]
        for person, detail in profile_people.items():
            relationship = detail.get("relationship", "")
            knows        = detail.get("knows", "")
            line         = f"- {person}"
            if relationship:
                line += f" ({relationship})"
            if knows:
                line += f": {knows}"
            lines.append(line)
        parts.append("\n".join(lines))

    # ── Beliefs from profile ──
    profile_beliefs = profile.get("beliefs", [])
    if profile_beliefs:
        lines = ["\nYOUR BELIEFS AND OPINIONS:"]
        for belief in profile_beliefs:
            lines.append(f"- {belief}")
        parts.append("\n".join(lines))

    # ── Secrets ──
    secrets = profile.get("secrets", [])
    if secrets:
        lines = ["\nTHINGS YOU KNOW BUT WOULD NOT VOLUNTEER:"]
        for secret in secrets:
            lines.append(f"- {secret}")
        parts.append("\n".join(lines))

    # ── Memory — current knowledge state from past interactions ──
    if memory:
        mem_people  = memory.get("people",  {})
        mem_facts   = memory.get("facts",   [])
        mem_beliefs = memory.get("beliefs", [])

        if mem_people:
            lines = ["\nPEOPLE YOU HAVE MET:"]
            for person, detail in mem_people.items():
                impression = detail.get("impression", "")
                trust      = detail.get("trust", "")
                notes      = detail.get("notes", [])
                line       = f"- {person}"
                if trust:
                    line += f" (trust: {trust})"
                if impression:
                    line += f": {impression}"
                for note in notes:
                    line += f" | {note}"
                lines.append(line)
            parts.append("\n".join(lines))

        if mem_facts:
            lines = ["\nTHINGS YOU HAVE LEARNED:"]
            for fact in mem_facts:
                about      = fact.get("about", "")
                knows      = fact.get("knows", "")
                confidence = fact.get("confidence", "certain")
                line       = f"- {about}: {knows}"
                if confidence != "certain":
                    line += f" (you {confidence} this)"
                lines.append(line)
            parts.append("\n".join(lines))

        if mem_beliefs:
            lines = ["\nOPINIONS YOU HAVE FORMED:"]
            for belief in mem_beliefs:
                lines.append(f"- {belief}")
            parts.append("\n".join(lines))

    # ── Dynamic context from Foundry — injected fresh, never saved ──
    if context:
        scene   = context.get("scene",   {})
        speaker = context.get("speaker", {})
        context_lines = []

        if scene.get("name"):
            context_lines.append(f"You are currently in: {scene['name']}.")

        if speaker.get("name"):
            desc    = f"The person now speaking to you is {speaker['name']}"
            details = []
            if speaker.get("race"):
                details.append(speaker["race"])
            if speaker.get("class"):
                details.append(speaker["class"])
            if details:
                desc += f", a {' '.join(details)}"
            desc += "."
            context_lines.append(desc)

        if context_lines:
            parts.append("\nCURRENT SITUATION:\n" + " ".join(context_lines))

    return "\n".join(parts)
