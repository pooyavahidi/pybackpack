from pybackpack.ai.genai.formatters import PromptMessageFormatter
from pybackpack.ai.genai.schema import Message

# pylint: disable=unsubscriptable-object


def test_prompt_message_formatter():
    formatter = PromptMessageFormatter()

    # Test from_messages
    messages = [
        Message(user="user message 1", assistant="assistant message 1"),
        Message(user="user message 2", assistant="assistant message {{var1}}"),
    ]
    prompt_message = formatter.from_messages(
        messages=messages,
        system_message="system message",
        variables={"var1": "A"},
    )
    assert prompt_message.system == "system message"
    assert len(prompt_message.messages) == 2
    assert prompt_message.messages[0].user == "user message 1"
    assert prompt_message.messages[0].assistant == "assistant message 1"
    assert prompt_message.messages[1].assistant == "assistant message A"

    # Test from_text and variables
    prompt_message = formatter.from_text(
        text="user message {{var1}}", variables={"var1": "A"}
    )
    assert prompt_message.system is None
    assert len(prompt_message.messages) == 1
    assert prompt_message.messages[0].user == "user message A"
    assert prompt_message.messages[0].assistant is None
