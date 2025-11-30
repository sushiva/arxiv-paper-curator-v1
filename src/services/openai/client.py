"""OpenAI client for LLM generation."""

import json
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from src.config import Settings
from src.exceptions import OllamaException  # Reuse for now
from src.services.ollama.prompts import RAGPromptBuilder, ResponseParser

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for interacting with OpenAI API."""

    def __init__(self, settings: Settings):
        """Initialize OpenAI client with settings."""
        self.api_key = settings.openai_api_key
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.prompt_builder = RAGPromptBuilder()
        self.response_parser = ResponseParser()
        logger.info(f"OpenAI client initialized with model: {self.model}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if OpenAI API is accessible.

        Returns:
            Dictionary with health status information
        """
        try:
            # Try to list models as health check
            models = await self.client.models.list()
            return {
                "status": "healthy",
                "message": "OpenAI API is accessible",
                "model": self.model,
            }
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"OpenAI API error: {str(e)}",
            }

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models (stub for compatibility).

        Returns:
            List with configured model
        """
        return [{"name": self.model}]

    async def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Generate text using OpenAI API.

        Args:
            model: Model name to use (uses configured model if not specified)
            prompt: Input prompt for generation
            stream: Whether to stream response (not implemented for non-streaming)
            **kwargs: Additional generation parameters (temperature, top_p, etc.)

        Returns:
            Response dictionary compatible with Ollama format
        """
        try:
            model_to_use = model if model and not model.startswith("llama") else self.model

            # Extract OpenAI-compatible parameters
            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)

            logger.info(f"Sending request to OpenAI: model={model_to_use}, temperature={temperature}")

            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_tokens,
                stream=False,
            )

            # Convert to Ollama-compatible format
            return {
                "model": model_to_use,
                "response": response.choices[0].message.content,
                "done": True,
                "context": [],
                "total_duration": 0,
                "load_duration": 0,
                "prompt_eval_count": response.usage.prompt_tokens,
                "eval_count": response.usage.completion_tokens,
            }

        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise OllamaException(f"Error generating with OpenAI: {e}")

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """
        Generate text with streaming response.

        Args:
            model: Model name to use
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters

        Yields:
            JSON chunks in Ollama-compatible format
        """
        try:
            model_to_use = model if model and not model.startswith("llama") else self.model

            temperature = kwargs.get("temperature", 0.7)
            top_p = kwargs.get("top_p", 0.9)

            logger.info(f"Starting streaming generation with OpenAI: model={model_to_use}")

            stream = await self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=self.max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    # Convert to Ollama-compatible format
                    yield {
                        "model": model_to_use,
                        "response": chunk.choices[0].delta.content,
                        "done": False,
                    }

            # Final chunk
            yield {
                "model": model_to_use,
                "response": "",
                "done": True,
            }

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise OllamaException(f"Error in streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "gpt-4o-mini",
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation
            use_structured_output: Whether to use structured output (not implemented)

        Returns:
            Dictionary with answer, sources, confidence, and citations
        """
        try:
            # Use the same prompt builder as Ollama
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            model_to_use = model if model and not model.startswith("llama") else self.model

            logger.info(f"Generating RAG answer with OpenAI model: {model_to_use}")

            response = await self.client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                top_p=0.9,
                max_tokens=self.max_tokens,
            )

            answer_text = response.choices[0].message.content

            # Build response structure manually (same as Ollama non-structured mode)
            sources = []
            seen_urls = set()
            for chunk in chunks:
                arxiv_id = chunk.get("arxiv_id")
                if arxiv_id:
                    arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                    if pdf_url not in seen_urls:
                        sources.append(pdf_url)
                        seen_urls.add(pdf_url)

            citations = list(set(chunk.get("arxiv_id") for chunk in chunks if chunk.get("arxiv_id")))

            return {
                "answer": answer_text,
                "sources": sources,
                "confidence": "medium",
                "citations": citations[:5],
                "model_used": model_to_use,
                "tokens_used": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                },
            }

        except Exception as e:
            logger.error(f"Error generating RAG answer with OpenAI: {e}")
            raise OllamaException(f"RAG generation failed: {e}")

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "gpt-4o-mini",
    ):
        """
        Generate a streaming RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation

        Yields:
            Streaming response chunks with partial answers
        """
        try:
            # Create prompt for streaming
            prompt = self.prompt_builder.create_rag_prompt(query, chunks)

            # Stream the response
            async for chunk in self.generate_stream(
                model=model,
                prompt=prompt,
                temperature=0.7,
                top_p=0.9,
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error generating streaming RAG answer with OpenAI: {e}")
            raise OllamaException(f"Failed to generate streaming RAG answer: {e}")
