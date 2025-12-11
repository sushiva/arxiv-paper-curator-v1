"""Anthropic Claude client for LLM generation."""

import logging
from typing import Any, Dict, List

from anthropic import Anthropic
from src.config import Settings
from src.exceptions import OllamaException  # Reuse for consistency
from src.services.ollama.prompts import RAGPromptBuilder

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Client for interacting with Anthropic Claude API."""

    def __init__(self, settings: Settings):
        """Initialize Anthropic client with settings."""
        self.api_key = settings.anthropic_api_key
        self.model_name = settings.anthropic_model
        self.max_tokens = settings.anthropic_max_tokens

        # Configure Anthropic SDK
        self.client = Anthropic(api_key=self.api_key)
        self.prompt_builder = RAGPromptBuilder()

        logger.info(f"Anthropic client initialized with model: {self.model_name}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Anthropic API is accessible.

        Returns:
            Dictionary with health status information
        """
        try:
            # Simple API test
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            return {
                "status": "healthy",
                "message": "Anthropic API is accessible",
                "model": self.model_name,
            }
        except Exception as e:
            logger.error(f"Anthropic health check failed: {e}")
            return {
                "status": "unhealthy",
                "message": f"Anthropic API error: {str(e)}",
            }

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = None,
        use_structured_output: bool = False,
        document_type: str = "arxiv",
    ) -> Dict[str, Any]:
        """
        Generate a RAG answer using retrieved chunks.

        Args:
            query: User's question
            chunks: Retrieved document chunks with metadata
            model: Model to use for generation (uses configured if not specified)
            use_structured_output: Whether to use structured output
            document_type: Type of documents (arxiv or financial)

        Returns:
            Dictionary with answer, sources, confidence, and citations
        """
        try:
            # Use the same prompt builder as other LLMs
            prompt = self.prompt_builder.create_rag_prompt(query, chunks, document_type)

            model_to_use = model or self.model_name
            logger.info(f"Generating RAG answer with Claude model: {model_to_use}")

            # Call Claude API
            response = self.client.messages.create(
                model=model_to_use,
                max_tokens=self.max_tokens,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract answer from response
            answer_text = response.content[0].text

            # Build response structure based on document type
            if document_type == "financial":
                sources = []
                seen_urls = set()
                for chunk in chunks:
                    source_url = chunk.get("source_url")
                    if source_url and source_url not in seen_urls:
                        sources.append(source_url)
                        seen_urls.add(source_url)

                citations = []
                for chunk in chunks:
                    company = chunk.get("company_name", "")
                    filing = chunk.get("filing_type", "")
                    if company and filing:
                        citation = f"{company} {filing}"
                        if citation not in citations:
                            citations.append(citation)
            else:
                # Build response structure for arXiv papers
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
                "confidence": "high",  # Claude generally provides high-quality answers
                "citations": citations[:5],
                "model_used": model_to_use,
                "tokens_used": {
                    "prompt": response.usage.input_tokens,
                    "completion": response.usage.output_tokens,
                    "total": response.usage.input_tokens + response.usage.output_tokens,
                },
            }

        except Exception as e:
            logger.error(f"Error generating RAG answer with Claude: {e}")
            raise OllamaException(f"RAG generation failed: {e}")
