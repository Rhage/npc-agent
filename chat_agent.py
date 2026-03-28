import re
import requests
from profiles import load_profile
from prompts import build_system_prompt, build_context_block, THINKING_PROMPT, THINKING_EXAMPLE_INPUT, THINKING_EXAMPLE_OUTPUT
from npc_db import NPCDatabase
from config import MODEL_NAME, LM_STUDIO_URL, DB_PATH

CHAT_ENDPOINT = f"{LM_STUDIO_URL}/v1/chat/completions"


class ChatAgent:
    def __init__(self, knowledge_mgr=None):
        self.conversations   = {}
        self._db             = NPCDatabase(DB_PATH)
        self._knowledge_mgr  = knowledge_mgr  # set after KnowledgeManager init

    def set_knowledge_manager(self, knowledge_mgr):
        """Called by server after KnowledgeManager is ready."""
        self._knowledge_mgr = knowledge_mgr

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

    def _call_llm(self, messages: list, label: str = "[LLM]") -> str:
        payload = {
            "model":    MODEL_NAME,
            "messages": messages
        }

        # Log the last system message and user message for debugging
        system_msgs = [m["content"] for m in messages if m["role"] == "system"]
        user_msg    = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        print(f"\n{label} System context ({len(system_msgs)} block(s)):")
        for i, s in enumerate(system_msgs):
            preview = s[:200] + "..." if len(s) > 200 else s
            print(f"  [{i}] {preview}")
        if user_msg:
            print(f"{label} User: {user_msg}\n")

        try:
            response = requests.post(CHAT_ENDPOINT, json=payload)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to LM Studio. Is it running?")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"LM Studio returned an error: {e.response.status_code}")

        result = response.json()["choices"][0]["message"]["content"]
        print(f"{label} Response: {result[:200]}{'...' if len(result) > 200 else ''}\n")
        return result

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
        profile_name:    str,
        formatted_input: str,
        speaker:         str = "",
        visible_actors:  list = None
    ) -> str:
        """
        Two-call flow:
          1. Thinking call — actor preparation, result discarded after use
          2. Response call — dialog generation, stored in history

        The context block is injected into both calls but never stored.
        The thinking output is passed to the response call as a <thinking>
        block and also never stored.
        """
        conversation    = self._get_conversation(profile_name)
        stored_messages = conversation["messages"]

        # Build dynamic context block for this message
        context_block = ""
        if speaker:
            context_block = build_context_block(
                npc_name       = profile_name,
                speaker        = speaker,
                visible_actors = visible_actors or [],
                db             = self._db,
                knowledge_mgr  = self._knowledge_mgr,
                query          = formatted_input,
            )

        # ── Call 1: Thinking ──────────────────────────────────────────────
        thinking_messages = [
            {"role": "system",    "content": THINKING_PROMPT},
            {"role": "user",      "content": THINKING_EXAMPLE_INPUT},
            {"role": "assistant", "content": THINKING_EXAMPLE_OUTPUT},
        ]
        if context_block:
            thinking_messages.append({"role": "system", "content": context_block})
        thinking_messages.append({"role": "user", "content": formatted_input})

        thinking = self._call_llm(thinking_messages, label="[Thinking]")

        # ── Call 2: Response ──────────────────────────────────────────────
        response_messages = stored_messages.copy()
        if context_block:
            response_messages.append({"role": "system", "content": context_block})
        if thinking:
            response_messages.append({
                "role":    "system",
                "content": f"<thinking>\n{thinking}\n</thinking>"
            })
        response_messages.append({"role": "user", "content": formatted_input})

        raw_response  = self._call_llm(response_messages, label="[Response]")

        # Trim to two paragraphs
        parts = raw_response.split("\n\n", 2)
        if len(parts) >= 3:
            raw_response = "\n\n".join(parts[:2])

        clean_response = self._strip_name_prefix(raw_response, profile_name)

        # Store only the clean exchange — no context, no thinking, no thinking block
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
        """Stateless single call — no history maintained. Still uses two-call flow."""
        profile       = load_profile(profile_name)
        system_prompt = build_system_prompt(profile)
        formatted     = f"Current speaker: {player}\n{player} says: {message}"

        context_block = build_context_block(
            npc_name       = profile_name,
            speaker        = player,
            visible_actors = visible_actors or [],
            db             = self._db,
            knowledge_mgr  = self._knowledge_mgr,
            query          = formatted,
        )

        # Thinking call
        thinking_messages = [
            {"role": "system",    "content": THINKING_PROMPT},
            {"role": "user",      "content": THINKING_EXAMPLE_INPUT},
            {"role": "assistant", "content": THINKING_EXAMPLE_OUTPUT},
        ]
        if context_block:
            thinking_messages.append({"role": "system", "content": context_block})
        thinking_messages.append({"role": "user", "content": formatted})
        thinking = self._call_llm(thinking_messages, label="[Thinking]")

        # Response call
        response_messages = [{"role": "system", "content": system_prompt}]
        if context_block:
            response_messages.append({"role": "system", "content": context_block})
        if thinking:
            response_messages.append({
                "role":    "system",
                "content": f"<thinking>\n{thinking}\n</thinking>"
            })
        response_messages.append({"role": "user", "content": formatted})

        raw_response = self._call_llm(response_messages, label="[Response]")
        return self._strip_name_prefix(raw_response, profile_name)
