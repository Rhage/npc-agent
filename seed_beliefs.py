"""
seed_beliefs.py
Quick script to seed NPC beliefs into the database.
Computes embeddings and stores everything in one shot.

Usage:
    python seed_beliefs.py
"""

from npc_db import NPCDatabase
from knowledge_manager import KnowledgeManager
from config import DB_PATH, EMBEDDING_MODEL


def add(km, db, npc_name, label, knowledge,
        terms="", subject="", source="", associations="", always_inject=False):
    blobs = km.encode_belief({
        "terms": terms, "subject": subject,
        "source": source, "associations": associations,
        "knowledge": knowledge
    })
    db.upsert_belief({
        "npc_name":      npc_name,
        "label":         label,
        "terms":         terms,
        "subject":       subject,
        "source":        source,
        "associations":  associations,
        "knowledge":     knowledge,
        "always_inject": always_inject,
        **blobs
    })
    tag = " [always]" if always_inject else ""
    print(f"  ✓ {npc_name} | {label}{tag}")


def main():
    print("Loading embedding model...")
    km = KnowledgeManager(EMBEDDING_MODEL)
    db = NPCDatabase(DB_PATH)
    print()

    # =========================================================================
    # Add beliefs below. Copy and paste the add() block for each one.
    #
    # Required: npc_name, label, knowledge
    # Optional: terms, subject, source, associations, always_inject
    #
    # Field guidance:
    #   label        - Human-readable name shown in DB Browser
    #   terms        - Names/ways of referring to this topic
    #   subject      - How the NPC perceives or describes the thing
    #   source       - How they know it (rumor, experience, study, etc.)
    #   associations - Connected people, places, factions they're aware of
    #   knowledge    - What gets injected into the LLM context (required)
    #   always_inject - True = always in context regardless of query
    # =========================================================================

    # ── Riko ──────────────────────────────────────────────────────────────────

    add(km, db,
        npc_name      = "Riko",
        label         = "Catgirls",
        terms         = "catgirl, catgirls, cat-eared, nekomimi",
        subject       = "Natural rivals of foxgirls. Untrustworthy, scheming, and irritating.",
        source        = "Common knowledge among foxgirls. Also personal experience.",
        associations  = "Foxgirls, who are their natural rivals and superiors in every way.",
        knowledge     = (
            "Riko has a deep, instinctive rivalry with catgirls and considers them "
            "fundamentally untrustworthy. She will make snide remarks about them given "
            "any opportunity and becomes notably prickly if one is mentioned positively."
        ),
        always_inject = True,
    )

    add(km, db,
        npc_name      = "Riko",
        label         = "The Merchants Guild",
        terms         = "merchants guild, the guild, traders guild, merchant guild",
        subject       = (
            "A powerful trade organization that controls commerce in the region. "
            "Riko has heard the Guildmaster is secretly a catgirl."
        ),
        source        = "Street rumor from a dockworker she met at a tavern.",
        associations  = "The Guildmaster, whoever that actually is. Dockworkers. Trade routes.",
        knowledge     = (
            "Riko has heard a rumor — which she absolutely believes — that the Merchants "
            "Guild Guildmaster is secretly a catgirl operating under a disguise. This makes "
            "her distrust the entire organization on principle. She considers their trade "
            "monopoly suspicious and thinks something underhanded is going on."
        ),
        always_inject = False,
    )

    add(km, db,
        npc_name      = "Riko",
        label         = "The Northern Road",
        terms         = "northern road, north road, road north, road to the north",
        subject       = "The main road heading north out of the city. Currently closed.",
        source        = "She tried to use it last week and was turned back by guards.",
        associations  = "City guard, who are enforcing the closure. Whatever is causing it.",
        knowledge     = (
            "The northern road has been closed for at least a week. City guards are "
            "turning everyone back at the gate with no explanation given. Riko tried "
            "to head north herself and was refused. She has no idea why it's closed "
            "and finds the whole thing suspicious."
        ),
        always_inject = False,
    )

    print()
    print("Done.")
    db.close()


if __name__ == "__main__":
    main()
