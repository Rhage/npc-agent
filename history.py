"""
history.py
Handles persistence of NPC conversation history to disk.

Each NPC's conversation is stored as a JSON file in the history/ directory,
keyed by profile name. History files are loaded on first conversation access
and saved automatically after each exchange.

File format:
    history/<profile_name>.json — a JSON array of message objects:
    [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]

The system message is always the first entry. On load, the saved system
message is replaced with the current one — this ensures profile edits
take effect without requiring a manual history clear.
"""

import os
import json

HISTORY_DIR = "history"


def load_history(profile_name: str) -> list | None:
    """
    Load saved conversation history for a profile.

    Returns the message list if a history file exists, None otherwise.
    The caller is responsible for replacing the system message with the
    current one if the profile has changed.
    """
    path = _history_path(profile_name)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            messages = json.load(f)
        if not isinstance(messages, list) or len(messages) == 0:
            return None
        print(f"[History] Loaded {len(messages)} messages for '{profile_name}'.")
        return messages
    except (json.JSONDecodeError, OSError) as e:
        print(f"[History] Failed to load history for '{profile_name}': {e}")
        return None


def save_history(profile_name: str, messages: list):
    """
    Save the current conversation history for a profile to disk.
    Creates the history/ directory if it doesn't exist.
    """
    os.makedirs(HISTORY_DIR, exist_ok=True)
    path = _history_path(profile_name)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[History] Failed to save history for '{profile_name}': {e}")


def delete_history(profile_name: str) -> bool:
    """
    Delete the saved history file for a profile.
    Returns True if deleted, False if no file existed.
    """
    path = _history_path(profile_name)
    if os.path.exists(path):
        try:
            os.remove(path)
            print(f"[History] Deleted history for '{profile_name}'.")
            return True
        except OSError as e:
            print(f"[History] Failed to delete history for '{profile_name}': {e}")
    return False


def list_histories() -> list[str]:
    """
    Return a list of profile names that have saved history files.
    """
    if not os.path.exists(HISTORY_DIR):
        return []
    return [
        f.replace(".json", "")
        for f in os.listdir(HISTORY_DIR)
        if f.endswith(".json")
    ]


def _history_path(profile_name: str) -> str:
    return os.path.join(HISTORY_DIR, f"{profile_name}.json")
