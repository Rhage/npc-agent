import requests
from profiles import load_profile, load_memory, build_system_prompt
from history import load_history, save_history, delete_history
from config import MODEL_NAME, LM_STUDIO_URL

CHAT_ENDPOINT = f"{LM_STUDIO_URL}/v1/chat/completions"


class ChatAgent:
    def __init__(self):
        # Each active conversation keyed by profile name
        # { "Oswin": { "profile": {...}, "memory": {...}, "messages": [...] } }
        self.conversations = {}

    def _get_conversation(self, profile_name: str) -> dict:
        print(f"[ChatAgent] _get_conversation called for: '{profile_name}'")
        print(f"[ChatAgent] Active conversations: {list(self.conversations.keys())}")

        if profile_name not in self.conversations:
            profile = load_profile(profile_name)
            memory  = load_memory(profile_name)

            # Base system prompt — no context, used for history storage
            system_prompt = build_system_prompt(profile, memory)

            saved = load_history(profile_name)
            if saved:
                # Always replace saved system message with current profile + memory.
                # Ensures edits take effect without clearing history.
                messages = saved
                if messages and messages[0]["role"] == "system":
                    messages[0]["content"] = system_prompt
                else:
                    messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                messages = [{"role": "system", "content": system_prompt}]

            self.conversations[profile_name] = {
                "profile":  profile,
                "memory":   memory,
                "messages": messages
            }

        return self.conversations[profile_name]

    def _call_llm(self, messages: list) -> str:
        payload = {
            "model":    MODEL_NAME,
            "messages": messages
        }
        try:
            response = requests.post(CHAT_ENDPOINT, json=payload)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise Exception("Could not connect to LM Studio. Is it running?")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"LM Studio returned an error: {e.response.status_code}")

        return response.json()["choices"][0]["message"]["content"]

    def stage_message(self, player: str, profile_name: str, message: str) -> str:
        """Prepares the formatted input without sending to LLM yet."""
        self._get_conversation(profile_name)
        return f"{player} says: {message}"

    def complete(self, profile_name: str, formatted_input: str, context: dict = None) -> str:
        """
        Appends the staged message, calls the LLM, and saves history.

        Context is injected into the system prompt for this call only —
        it is never written to the history file so saved conversations
        never contain stale scene or character information.
        """
        conversation = self._get_conversation(profile_name)
        messages     = conversation["messages"]
        profile      = conversation["profile"]
        memory       = conversation["memory"]

        messages.append({"role": "user", "content": formatted_input})

        # Build a context-enriched prompt for this call only.
        # The stored messages list keeps the base prompt — history stays clean.
        if context:
            contextual_prompt = build_system_prompt(profile, memory, context)
            call_messages     = [{"role": "system", "content": contextual_prompt}] + messages[1:]
        else:
            call_messages = messages

        response_text = self._call_llm(call_messages)

        # Trim to two paragraphs
        parts = response_text.split("\n\n", 2)
        if len(parts) >= 3:
            response_text = "\n\n".join(parts[:2]) + "\n\n"

        messages.append({"role": "assistant", "content": response_text})

        # Save history with base prompt only — no context
        save_history(profile_name, messages)

        return response_text

    def respond_once(self, player: str, profile_name: str, message: str, context: dict = None) -> str:
        """Stateless single call — no history loaded or saved."""
        profile       = load_profile(profile_name)
        memory        = load_memory(profile_name)
        system_prompt = build_system_prompt(profile, memory, context)
        messages      = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"{player} says: {message}"}
        ]
        return self._call_llm(messages)

    def reset_conversation(self, profile_name: str):
        """
        Clear the in-memory conversation for a profile and delete its history file.
        The next message will start a fresh conversation.
        Memory is preserved — only the conversation log is cleared.
        """
        if profile_name in self.conversations:
            del self.conversations[profile_name]
        delete_history(profile_name)
        print(f"[ChatAgent] Conversation reset for '{profile_name}'.")
