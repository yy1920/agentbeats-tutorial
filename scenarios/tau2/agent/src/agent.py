import json
import os

from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


from groq import Groq


load_dotenv()


SYSTEM_PROMPT = (
    "You are a helpful home controlling agent to control smart appliances. "
    "Follow the policy and tool instructions provided in each message."
)


class Agent:
    def __init__(self):
        self.model = "llama-3.3-70b-versatile"
        self.client = Groq()
        self.messages: list[dict[str, object]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(TaskState.working, new_agent_text_message("Thinking..."))

        self.messages.append({"role": "user", "content": input_text})

        # try:
        #     completion = self.client.chat.completions.create(
        #         model=self.model,
        #         messages=self.messages,
        #         temperature=0.0,
        #         response_format={"type": "json_object"},
        #         max_completion_tokens=1024,
        #     )
        #     assistant_content = completion.choices[0].message.content or "{}"
        #     assistant_json = json.loads(assistant_content)
        # except Exception:
        assistant_json = {
            "name": "respond",
            "arguments": {"content": "['living_room.light.turn_on()']"},
        }
        assistant_content = json.dumps(assistant_json)

        self.messages.append({"role": "assistant", "content": assistant_content})

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=assistant_json))],
            name="Action",
        )

