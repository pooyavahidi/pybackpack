from typing import List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A simple document which contains content and metadata."""

    content: str
    metadata: dict = Field(default_factory=dict)


class CompletionMetaData(BaseModel):
    """Metadata of the model completion."""

    model: str
    timestamp: str = None
    temperature: float
    max_tokens: int
    prompt_tokens: int = None
    completion_tokens: int = None
    total_tokens: int = None
    invocation_latency: int = None


class Message(BaseModel):
    """Human and AI Messages."""

    user: Optional[str] = None
    assistant: Optional[str] = None
    metadata: Optional[CompletionMetaData] = None


class PromptMessage(BaseModel):
    """Contains messages which make up the prompt."""

    system: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)


class PromptParameters(BaseModel):
    """Parameters for the prompt."""

    temperature: float = 0.0
    max_tokens: int = 500
    top_p: float = None
    top_k: int = None


class Prompt(BaseModel):
    """Prompt for the Model."""

    message: PromptMessage
    parameters: PromptParameters


class ModelResponse(BaseModel):
    """Response from the Model."""

    content: str
    metadata: Optional[CompletionMetaData] = None


class Conversation(BaseModel):
    """Conversation with the AI."""

    conversation_id: str = Field(alias="id")
    created: str
    system: str
    messages: List[Message] = Field(default_factory=list)
    metadata: Optional[dict] = Field(default_factory=dict)
