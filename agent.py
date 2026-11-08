from dotenv import load_dotenv
from google import genai
from google.genai import types

import os

load_dotenv()
client = genai.Client()
model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

system_instruction = "Tu es un assistant qui aide à résoudre des problèmes."

config = types.GenerateContentConfig(
    system_instruction=system_instruction,
)

history: list[types.Content] = []

def call_agent(user_prompt: str) -> str:
    """
    Calls the LLM with the given user prompt and returns the response text.

    Args:
        user_prompt (str): The prompt to send to the LLM.

    Returns:
        str: The response text from the LLM.
    """
    user_content = types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])

    try:
        response = client.models.generate_content(
            model=model,
            contents=[*history, user_content],
            config=config
        )
    except genai.errors.APIError as error:
        raise RuntimeError(f"Gemini request failed: {error}") from error
    except Exception as error:  # network, timeout... the SDK does not wrap them all
        raise RuntimeError(f"Gemini request failed: {error!r}") from error

    if not response.candidates:
        raise RuntimeError("Gemini returned no candidates (response may have been blocked)")

    history.extend([user_content, response.candidates[0].content])

    return response.text
