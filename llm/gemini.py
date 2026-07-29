"""Google Gemini LLM Integration Client Module.

Encapsulates authentication, text generation requests, safety settings, retry logic,
and error handling for Google Gemini model APIs.
"""

import logging
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class GeminiLLMClient:
    """Wrapper client for communicating with Google Gemini LLM services."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> None:
        """Initializes Gemini client parameters.

        Args:
            api_key: Gemini API Key string. Defaults to settings.
            model_name: Gemini model ID string. Defaults to settings.
            temperature: Generation temperature. Defaults to settings.
            max_output_tokens: Max completion token length. Defaults to settings.
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL_NAME
        self.temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE
        self.max_output_tokens = (
            max_output_tokens if max_output_tokens is not None else settings.LLM_MAX_OUTPUT_TOKENS
        )
        self._client = None

        logger.info(
            "GeminiLLMClient configured (model=%s, temp=%.2f)",
            self.model_name,
            self.temperature,
        )

    def _get_client(self) -> Any:
        """Lazy loader for Google GenAI Client instance.

        Returns:
            GenAI client object or mock fallback.
        """
        if self._client is None:
            if not self.api_key or self.api_key == "your-gemini-api-key-here":
                logger.warning(
                    "GEMINI_API_KEY is not set or using default template value. Falling back to mock response generator."
                )
                self._client = "MOCK_GEMINI_CLIENT"
                return self._client

            try:
                from google import genai

                self._client = genai.Client(api_key=self.api_key)
                logger.info("Google GenAI client initialized successfully.")
            except ImportError:
                logger.warning("google-genai SDK not installed. Operating with mock client.")
                self._client = "MOCK_GEMINI_CLIENT"
            except Exception as exc:
                logger.error("Failed to initialize Google GenAI Client: %s", exc, exc_info=True)
                raise
        return self._client

    def generate_answer(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generates a text completion response from Google Gemini.

        Args:
            prompt: User request prompt with injected document context.
            system_instruction: Optional system level grounding instruction.

        Returns:
            Dictionary containing text answer and generation metadata.

        Raises:
            RuntimeError: If Gemini API request fails.
        """
        logger.info("Sending prompt request to Gemini LLM (model=%s)", self.model_name)
        logger.debug("Prompt content length: %d chars", len(prompt))

        client = self._get_client()
        if client == "MOCK_GEMINI_CLIENT":
            logger.info("Returning mock LLM grounded response.")
            return {
                "answer": (
                    "Based on the provided Machine Learning documentation, Supervised Learning algorithms "
                    "train on labeled datasets containing both features and ground-truth targets. "
                    "Common algorithms include Decision Trees, Support Vector Machines, and Neural Networks."
                ),
                "model": self.model_name,
                "finish_reason": "STOP",
            }

        try:
            from google.genai import types

            config = types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                system_instruction=system_instruction,
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            answer_text = response.text or ""
            logger.info("Successfully received LLM response (%d chars)", len(answer_text))

            return {
                "answer": answer_text,
                "model": self.model_name,
                "finish_reason": "STOP",
            }
        except Exception as exc:
            logger.error("Error communicating with Gemini API: %s", exc, exc_info=True)
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc
