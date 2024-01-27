from typing import List, Dict
from pybackpack.text.template import Template as TextTemplate
from pybackpack.ai.genai.schema import PromptMessage, Message


class PromptMessageFormatter:
    """Format prompt messages form/to different formats."""

    def _render_message(self, content: str, variables: Dict) -> str:
        if content and content.strip():
            return TextTemplate(content).render(variables)
        return None

    def from_messages(
        self,
        messages: List[Message],
        system_message: str = None,
        variables: Dict = None,
    ) -> PromptMessage:
        """Returns a PromptMessage object from the given messages."""
        system_message = self._render_message(system_message, variables)

        for message in messages:
            message.user = self._render_message(message.user, variables)
            message.assistant = self._render_message(
                message.assistant, variables
            )

        return PromptMessage(system=system_message, messages=messages)

    def from_text(self, text: str, variables: Dict = None) -> PromptMessage:
        """Returns a PromptMessage object from the given text.

        text will be the user message and system message will be empty.
        """
        message = Message(user=text, assistant=None)
        return self.from_messages(messages=[message], variables=variables)
