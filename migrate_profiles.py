"""
migrate_profiles.py
One-time migration script — imports existing profile JSON files into the
NPC database.

Run once after setting up the database:
    python migrate_profiles.py

Safe to re-run — existing records are updated, not duplicated.
After migration, the profiles/ JSON files are no longer used by the server
but are kept as backup until you're satisfied with the migration.
"""

import os
import json
import argparse
from npc_db import NPCDatabase
from config import DB_PATH, PROFILES_DIR


def migrate_profile(db: NPCDatabase, path: str) -> str:
    """
    Migrate a single profile JSON file into the database.
    Returns the NPC name on success.
    """
    with open(path, "r") as f:
        data = json.load(f)

    name = data.get("name")
    if not name:
        raise ValueError(f"Profile at '{path}' has no 'name' field.")

    # Map old JSON fields to new schema.
    # personality and speech_style from the old format are merged into
    # the new fields with a best-effort split — the DM should review
    # and refine these after migration.
    voice = data.get("voice", {})

    npc = {
        "name":              name,
        "full_name":         data.get("full_name"),
        "aliases":           data.get("aliases"),
        "species":           data.get("species"),
        "gender":            data.get("gender"),
        "age_description":   data.get("age_description"),
        "appearance":        data.get("appearance"),
        "role":              data.get("role"),
        "backstory":         data.get("backstory"),
        # Old 'personality' maps to new 'personality' — will need DM review
        # as it previously contained a mix of personality + description
        "personality":       data.get("personality"),
        # Old 'speech_style' maps to 'speaking_style' — will need DM review
        # as it previously contained response format instructions mixed with
        # actual speaking style. Response format instructions should be removed
        # and placed in the global prompt config in prompts.py instead.
        "speaking_style":    data.get("speech_style") or data.get("speaking_style"),
        "voice_language":    voice.get("language"),
        "voice_description": voice.get("description"),
    }

    db.upsert_npc(npc)
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Migrate NPC profile JSON files into the database."
    )
    parser.add_argument(
        "--profiles-dir",
        default=PROFILES_DIR,
        help=f"Directory containing profile JSON files (default: {PROFILES_DIR})"
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help=f"Path to the database file (default: {DB_PATH})"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.profiles_dir):
        print(f"ERROR: Profiles directory not found: {args.profiles_dir}")
        return

    db = NPCDatabase(args.db)

    json_files = [
        f for f in os.listdir(args.profiles_dir)
        if f.endswith(".json")
    ]

    if not json_files:
        print(f"No JSON files found in {args.profiles_dir}")
        return

    print(f"Migrating {len(json_files)} profile(s) from '{args.profiles_dir}' to '{args.db}'...")
    print()

    succeeded = []
    failed    = []

    for filename in sorted(json_files):
        path = os.path.join(args.profiles_dir, filename)
        try:
            name = migrate_profile(db, path)
            print(f"  ✓ {filename} → '{name}'")
            succeeded.append(name)
        except Exception as e:
            print(f"  ✗ {filename} — FAILED: {e}")
            failed.append(filename)

    db.close()

    print()
    print(f"Migration complete: {len(succeeded)} succeeded, {len(failed)} failed.")

    if succeeded:
        print()
        print("Next steps:")
        print("  1. Review migrated NPCs — personality and speaking_style fields")
        print("     may need refinement as they were previously mixed together.")
        print("  2. Remove response format instructions from speaking_style —")
        print("     these now belong in prompts.py as global config.")
        print("  3. Fill in any new fields (species, appearance, backstory, etc.)")
        print("     using the DM console or direct database edits.")
        print("  4. Once satisfied, the profiles/ JSON files can be archived.")


if __name__ == "__main__":
    main()
