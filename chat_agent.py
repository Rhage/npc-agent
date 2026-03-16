import re
import requests
from profiles import load_profile
from prompts import build_system_prompt, build_context_block
from npc_db import NPCDatabase
from config import MODEL_NAME, LM_STUDIO_URL, DB_PATH

CHAT_ENDPOINT = f"{LM_STUDIO_URL}/v1/chat/completions"


class ChatAgent:
    def __init__(self):
        self.conversations = {}
        self._db = NPCDatabase(DB_PATH)

    def _get_conversation(self, profile_name: str) -> dict:
        if profile_name not in self.conversations:
            profile       = load_profile(profile_name)
            system_prompt = build_system_prompt(profile)
            messages      = [{"role": "system", "content": system_prompt}]

            # TODO: load conversation history here

            self.conversations[profile_name] = {
                "profile":  profile,
                "messages": messages
            }

        return self.conversations[profile_name]

    def _call_llm(self, messages: list) -> str:
        payload = {
            "model":    MODEL_NAME,
            "messages": messages
        }

        # Log system prompt, context block, and latest user message for debugging
        system  = next((m["content"] for m in messages if m["role"] == "system"), None)
        context = next((m["content"] for m in messages if m["role"] == "system" and m["content"].startswith("[Current scene context]")), None)
        user    = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        if system and not context:
            print(f"\n[ChatAgent] System prompt:\n{system}\n")
        elif system:
            # Print static prompt once, context block every call
            static = next((m["content"] for m in messages if m["role"] == "system" and not m["content"].startswith("[Current scene context]")), None)
            if static:
                print(f"\n[ChatAgent] System prompt:\n{static}\n")
            print(f"[ChatAgent] Context block:\n{context}\n")
        if user:
            print(f"[ChatAgent] User message: {user}\n")

        try:
            response = requests.post(CHAT_ENDPOINT, json=payload)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to LM Studio. Is it running?")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"LM Studio returned an error: {e.response.status_code}")

        return response.json()["choices"][0]["message"]["content"]

    def _strip_name_prefix(self, text: str, profile_name: str) -> str:
        """
        Strip any leading name prefix the LLM may have added.
        Handles variants like "Riko: ", "riko:", "RIKO: " etc.
        """
        pattern = rf"^\s*{re.escape(profile_name)}\s*:\s*"
        return re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()

    def stage_message(self, player: str, profile_name: str, message: str) -> str:
        """
        Prepares the formatted input without sending to LLM yet.
        Initializes the conversation if this is the first message.
        """
        self._get_conversation(profile_name)
        return f"Current speaker: {player}\n{player} says: {message}"

    def complete(
        self,
        profile_name:   str,
        formatted_input: str,
        speaker:        str = "",
        visible_actors: list = None
    ) -> str:
        """
        Appends the staged message and calls the LLM.

        Builds a dynamic context block from current visibility and injects
        it into the call payload — but does NOT store it in history.
        Only the bare exchange is stored.

        Args:
            profile_name:    NPC profile name.
            formatted_input: Staged message from stage_message().
            speaker:         Player character name for this message.
            visible_actors:  Actor names currently visible to the NPC token.
        """
        conversation    = self._get_conversation(profile_name)
        stored_messages = conversation["messages"]  # clean history, never contains context

        # Build dynamic context block for this call
        context_block = ""
        if speaker:
            context_block = build_context_block(
                npc_name       = profile_name,
                speaker        = speaker,
                visible_actors = visible_actors or [],
                db             = self._db
            )

        # Assemble call payload — context injected but not stored
        call_messages = stored_messages.copy()
        if context_block:
            call_messages.append({"role": "system", "content": context_block})
        call_messages.append({"role": "user", "content": formatted_input})

        raw_response = self._call_llm(call_messages)

        # Trim to two paragraphs
        parts = raw_response.split("\n\n", 2)
        if len(parts) >= 3:
            raw_response = "\n\n".join(parts[:2])

        # Strip any name prefix the LLM added
        clean_response = self._strip_name_prefix(raw_response, profile_name)

        # Store clean exchange in history with name prefix for LLM context
        stored_messages.append({"role": "user",      "content": formatted_input})
        stored_messages.append({"role": "assistant",  "content": f"{profile_name}: {clean_response}"})

        # TODO: autosave history here

        return clean_response

    def respond_once(
        self,
        player:         str,
        profile_name:   str,
        message:        str,
        visible_actors: list = None
    ) -> str:
        """Stateless single call — no history maintained."""
        profile       = load_profile(profile_name)
        system_prompt = build_system_prompt(profile)
        formatted     = f"Current speaker: {player}\n{player} says: {message}"

        context_block = build_context_block(
            npc_name       = profile_name,
            speaker        = player,
            visible_actors = visible_actors or [],
            db             = self._db
        )

        messages = [{"role": "system", "content": system_prompt}]
        if context_block:
            messages.append({"role": "system", "content": context_block})
        messages.append({"role": "user", "content": formatted})

        raw_response   = self._call_llm(messages)
        return self._strip_name_prefix(raw_response, profile_name)
