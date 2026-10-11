"""Conversation identity and auditable prompts for a single attempt."""
from dataclasses import dataclass, field
import uuid
from .prompt import initial_prompt, continuation_prompt

@dataclass
class Context:
    description: str
    arm: str
    rounds: int
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: list = field(default_factory=list)

    def prompt(self, number):
        text = initial_prompt(self.description, self.arm) if number == 1 else continuation_prompt(number, self.rounds)
        self.messages.append({"round": number, "role": "user", "content": text})
        return text

    def session_flags(self, number):
        if self.rounds == 1:
            return ["--no-session-persistence"]
        return ["--session-id" if number == 1 else "--resume", self.session_id]
