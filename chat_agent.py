import requests
from profiles import load_profile, build_system_prompt
from config import MODEL_NAME, LM_STUDIO_URL

CHAT_ENDPOINT = f"{LM_STUDIO_URL}/v1/chat/completions"

class ChatAgent:
    def __init__(self):
        # Each active conversation keyed by profile name
        # { "riko": { "profile": {...}, "messages": [...] } }
        self.conversations = {}

    def _get_conversation(self, profile_name: str) -> dict:
        if profile_name not in self.conversations:
            profile = load_profile(profile_name)
            system_prompt = build_system_prompt(profile)

            messages = [{"role": "system", "content": system_prompt}]

            # TODO: load conversation history here
            # saved = load_history(profile_name)
            # if saved:
            #     messages = saved

            self.conversations[profile_name] = {
                "profile": profile,
                "messages": messages
            }

        return self.conversations[profile_name]

    def _call_llm(self, messages: list) -> str:
        payload = {
            "model": MODEL_NAME,
            "messages": messages
        }
        print(messages)
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
        conversation = self._get_conversation(profile_name)
        formatted_input = f"{player} says: {message}"
        return formatted_input

    def complete(self, profile_name: str, formatted_input: str) -> str:
        """Appends the staged message and calls the LLM."""
        conversation = self._get_conversation(profile_name)
        messages = conversation["messages"]
        messages.append({"role": "user", "content": formatted_input})
        response_text = self._call_llm(messages)

        # Trim to two paragraphs
        parts = response_text.split("\n\n", 2)
        if len(parts) >= 3:
            response_text = "\n\n".join(parts[:2]) + "\n\n"

        messages.append({"role": "assistant", "content": response_text})
        # TODO: autosave history here
        return response_text

    def respond_once(self, player: str, profile_name: str, message: str) -> str:
        profile = load_profile(profile_name)
        system_prompt = build_system_prompt(profile)
        formatted_input = f"{player} says: {message}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_input}
        ]
        return self._call_llm(messages)