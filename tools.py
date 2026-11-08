"""Tools the agent can ask the model to call."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar
from zoneinfo import ZoneInfo

from google.genai import types


class Tool(ABC):
    """Base class for a tool the model can ask the agent to run.

    Subclasses must define the class attributes below and implement `run`.

    Attributes:
        name: Unique identifier the model uses to request the tool.
        description: What the tool does. The model reads it to decide when
            to request the tool, so it should be clear and specific.
        parameters: JSON Schema of the arguments. The model reads it to
            decide which arguments to send.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[dict[str, Any]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check that subclasses define the required class attributes."""
        super().__init_subclass__(**kwargs)
        for attr in ("name", "description", "parameters"):
            if not getattr(cls, attr, None):
                raise TypeError(f"{cls.__name__} must define a non-empty '{attr}'")

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Run the tool with the arguments requested by the model.

        Args:
            **kwargs: Arguments matching the `parameters` schema.

        Returns:
            The result sent back to the model. It must be JSON-serializable.
        """

    @property
    def declaration(self) -> types.FunctionDeclaration:
        """The tool's declaration, sent to the model so it knows the tool exists."""
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters_json_schema=self.parameters,
        )


class GetCurrentTimeTool(Tool):
    """Tool that returns the current time in ISO 8601 format."""

    name = "get_current_time"
    description = (
        "Get the current date and time in a given timezone. "
        "Use it whenever the user asks for the time or date somewhere, "
        "since you cannot know the current time otherwise."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name, e.g. 'Europe/Paris'",
            }
        },
        "required": ["timezone"],
    }

    def run(self, **kwargs: Any) -> str:
        """Return the current time in ISO 8601 format.

        Args:
            **kwargs: Must contain ``timezone``, an IANA timezone name
            such as "Europe/Paris".

        Returns:
            The current date and time in ISO 8601 format, with the UTC offset.
        """
        timezone = kwargs["timezone"]

        return datetime.now(ZoneInfo(timezone)).isoformat()
