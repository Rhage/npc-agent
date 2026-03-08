import json
import os

def load_profile(profile_name: str) -> dict:
    path = os.path.join("profiles", f"{profile_name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No profile found: '{profile_name}'")
    with open(path, "r") as f:
        return json.load(f)
        
def build_system_prompt(profile: dict) -> str:
    return profile["speech_style"] + " " + profile["personality"]