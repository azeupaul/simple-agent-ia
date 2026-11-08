"""Tools the agent can ask the model to call."""

from abc import ABC, abstractmethod
from typing import Any

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

    name: str
    description: str
    parameters: dict[str, Any]

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
