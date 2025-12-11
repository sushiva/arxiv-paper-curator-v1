import json
import logging
import time
from typing import Dict, List, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from src.dependencies import (
    CacheDep,
    EmbeddingsDep,
    FinancialOpenSearchDep,
    LangfuseDep,
    LLMDep,
    OpenSearchDep,
)
from src.schemas.api.ask import AskRequest, AskResponse
from src.services.langfuse.tracer import RAGTracer
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.financial_client import FinancialOpenSearchClient

logger = logging.getLogger(__name__)

# Two separate routers - one for regular ask, one for streaming
ask_router = APIRouter(tags=["ask"])
stream_router = APIRouter(tags=["stream"])


async def _prepare_chunks_and_sources_arxiv(
    request: AskRequest,
    opensearch_client: OpenSearchClient,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for arXiv papers."""

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        search_results = opensearch_client.search_unified(
            query=request.query,
            query_embedding=query_embedding,
            size=request.top_k,
            from_=0,
            categories=request.categories,
            use_hybrid=request.use_hybrid and query_embedding is not None,
            min_score=0.0,
        )

        # Extract essential data for LLM
        chunks = []
        arxiv_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            arxiv_id = hit.get("arxiv_id", "")

            # Minimal chunk data for LLM
            chunks.append(
                {
                    "arxiv_id": arxiv_id,
                    "chunk_text": hit.get("chunk_text", hit.get("abstract", "")),
                }
            )

            if arxiv_id:
                arxiv_ids.append(arxiv_id)
                arxiv_id_clean = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                sources_set.add(f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf")

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, arxiv_ids, search_results.get("total", 0))

    return chunks, list(sources_set), arxiv_ids


async def _prepare_chunks_and_sources_financial(
    request: AskRequest,
    financial_opensearch_client: FinancialOpenSearchClient,
    embeddings_service,
    rag_tracer: RAGTracer,
    trace=None,
) -> tuple[List[Dict], List[str], List[str]]:
    """Retrieve and prepare chunks for financial documents."""

    # Handle embeddings for hybrid search
    query_embedding = None
    if request.use_hybrid:
        with rag_tracer.trace_embedding(trace, request.query) as embedding_span:
            try:
                query_embedding = await embeddings_service.embed_query(request.query)
                logger.info("Generated query embedding for hybrid search")
            except Exception as e:
                logger.warning(f"Failed to generate embeddings, falling back to BM25: {e}")
                if embedding_span:
                    rag_tracer.tracer.update_span(embedding_span, output={"success": False, "error": str(e)})

    # Search with tracing
    with rag_tracer.trace_search(trace, request.query, request.top_k) as search_span:
        if request.use_hybrid and query_embedding is not None:
            search_results = financial_opensearch_client.search_chunks_hybrid(
                query=request.query,
                query_embedding=query_embedding,
                size=request.top_k,
                ticker=request.ticker,
                document_types=request.filing_types,
                min_score=0.0,
            )
        else:
            search_results = financial_opensearch_client.search_chunks_bm25(
                query=request.query,
                size=request.top_k,
                ticker=request.ticker,
                document_types=request.filing_types,
            )

        # Extract essential data for LLM
        chunks = []
        document_ids = []
        sources_set = set()

        for hit in search_results.get("hits", []):
            document_id = hit.get("document_id", "")
            ticker = hit.get("ticker_symbol", "")
            company_name = hit.get("company_name", "")
            doc_type = hit.get("document_type", "")
            filing_date = hit.get("filing_date", "")
            accession = hit.get("accession_number", "")

            # Minimal chunk data for LLM (financial-specific)
            chunks.append(
                {
                    "document_id": document_id,
                    "ticker": ticker,
                    "company_name": company_name,
                    "document_type": doc_type,
                    "filing_date": filing_date,
                    "chunk_text": hit.get("chunk_text", ""),
                }
            )

            if document_id:
                document_ids.append(document_id)

            # Build SEC EDGAR source URL
            if accession:
                # SEC EDGAR URL format
                accession_clean = accession.replace("-", "")
                sources_set.add(
                    f"https://www.sec.gov/cgi-bin/viewer?action=view&cik=&accession_number={accession}&xbrl_type=v"
                )

        # End search span with essential metadata
        rag_tracer.end_search(search_span, chunks, document_ids, search_results.get("total", 0))

    return chunks, list(sources_set), document_ids


@ask_router.post("/ask", response_model=AskResponse)
async def ask_question(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    financial_opensearch_client: FinancialOpenSearchDep,
    embeddings_service: EmbeddingsDep,
    llm_client: LLMDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> AskResponse:
    """Clean RAG endpoint with support for both arXiv and financial documents."""

    rag_tracer = RAGTracer(langfuse_tracer)
    start_time = time.time()

    with rag_tracer.trace_request("api_user", request.query) as trace:
        try:
            # Check exact cache first
            cached_response = None
            if cache_client:
                try:
                    cached_response = await cache_client.find_cached_response(request)
                    if cached_response:
                        logger.info("Returning cached response for exact query match")
                        return cached_response
                except Exception as e:
                    logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

            # Route to appropriate search based on document_type
            if request.document_type == "financial":
                chunks, sources, _ = await _prepare_chunks_and_sources_financial(
                    request, financial_opensearch_client, embeddings_service, rag_tracer, trace
                )
            else:  # "arxiv"
                chunks, sources, _ = await _prepare_chunks_and_sources_arxiv(
                    request, opensearch_client, embeddings_service, rag_tracer, trace
                )

            if not chunks:
                no_results_message = (
                    "I couldn't find any relevant financial documents to answer your question."
                    if request.document_type == "financial"
                    else "I couldn't find any relevant papers to answer your question."
                )
                response = AskResponse(
                    query=request.query,
                    answer=no_results_message,
                    sources=[],
                    chunks_used=0,
                    search_mode="bm25" if not request.use_hybrid else "hybrid",
                )
                rag_tracer.end_request(trace, response.answer, time.time() - start_time)
                return response

            # Build prompt
            with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                from src.services.ollama.prompts import RAGPromptBuilder

                prompt_builder = RAGPromptBuilder()

                try:
                    prompt_data = prompt_builder.create_structured_prompt(
                        request.query,
                        chunks,
                        document_type=request.document_type
                    )
                    final_prompt = prompt_data["prompt"]
                except Exception:
                    final_prompt = prompt_builder.create_rag_prompt(
                        request.query,
                        chunks,
                        document_type=request.document_type
                    )

                rag_tracer.end_prompt(prompt_span, final_prompt)

            # Generate answer with fallback logic
            answer = None
            provider_used = "primary"

            with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                try:
                    # Try primary LLM (Gemini or configured provider)
                    rag_response = await llm_client.generate_rag_answer(
                        query=request.query,
                        chunks=chunks,
                        model=request.model,
                        document_type=request.document_type
                    )
                    answer = rag_response.get("answer", "Unable to generate answer")
                    logger.info(f"Successfully generated answer using primary LLM provider")

                except Exception as primary_error:
                    # Primary LLM failed - try 3-tier fallback strategy
                    logger.warning(f"Tier 1 (Primary) failed: {primary_error}")

                    from src.config import get_settings
                    from src.services.gemini.client import GeminiClient
                    from src.services.anthropic.client import AnthropicClient
                    from src.services.openai.client import OpenAIClient

                    settings = get_settings()

                    # TIER 2: Try Gemini Pro (upgraded model)
                    try:
                        logger.info("Tier 2: Trying Gemini Pro fallback...")
                        if settings.gemini_api_key:
                            gemini_pro_client = GeminiClient(settings)
                            rag_response = await gemini_pro_client.generate_rag_answer(
                                query=request.query,
                                chunks=chunks,
                                model="gemini-2.0-flash-exp",  # Upgraded Gemini model
                                document_type=request.document_type
                            )
                            answer = rag_response.get("answer", "Unable to generate answer")
                            provider_used = "gemini_pro_fallback"
                            logger.info("✅ Tier 2 SUCCESS: Gemini Pro answered")
                        else:
                            raise Exception("Gemini API key not configured")

                    except Exception as tier2_error:
                        logger.warning(f"Tier 2 (Gemini Pro) failed: {tier2_error}")

                        # TIER 3: Try Claude Haiku
                        try:
                            logger.info("Tier 3: Trying Claude Haiku fallback...")
                            if settings.anthropic_api_key:
                                claude_client = AnthropicClient(settings)
                                rag_response = await claude_client.generate_rag_answer(
                                    query=request.query,
                                    chunks=chunks,
                                    model=settings.anthropic_model,
                                    document_type=request.document_type
                                )
                                answer = rag_response.get("answer", "Unable to generate answer")
                                provider_used = "claude_fallback"
                                logger.info("✅ Tier 3 SUCCESS: Claude Haiku answered")
                            else:
                                raise Exception("Anthropic API key not configured")

                        except Exception as tier3_error:
                            logger.warning(f"Tier 3 (Claude) failed: {tier3_error}")

                            # TIER 4: Try OpenAI (last resort)
                            try:
                                logger.info("Tier 4: Trying OpenAI fallback (last resort)...")
                                if settings.openai_api_key:
                                    openai_client = OpenAIClient(settings)
                                    rag_response = await openai_client.generate_rag_answer(
                                        query=request.query,
                                        chunks=chunks,
                                        model=settings.openai_model,
                                        document_type=request.document_type
                                    )
                                    answer = rag_response.get("answer", "Unable to generate answer")
                                    provider_used = "openai_fallback"
                                    logger.info("✅ Tier 4 SUCCESS: OpenAI answered")
                                else:
                                    raise Exception("OpenAI API key not configured")

                            except Exception as tier4_error:
                                # All 4 tiers failed
                                logger.error(f"❌ ALL TIERS FAILED - Tier 4 (OpenAI): {tier4_error}")
                                raise HTTPException(
                                    status_code=503,
                                    detail="LLM service temporarily unavailable. Please try again in a few moments."
                                )

                rag_tracer.end_generation(gen_span, answer, request.model)

            # Prepare response
            response = AskResponse(
                query=request.query,
                answer=answer,
                sources=sources,
                chunks_used=len(chunks),
                search_mode="bm25" if not request.use_hybrid else "hybrid",
            )

            rag_tracer.end_request(trace, answer, time.time() - start_time)

            # Store response in exact match cache
            if cache_client:
                try:
                    await cache_client.store_response(request, response)
                except Exception as e:
                    logger.warning(f"Failed to store response in cache: {e}")

            return response

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@stream_router.post("/stream")
async def ask_question_stream(
    request: AskRequest,
    opensearch_client: OpenSearchDep,
    financial_opensearch_client: FinancialOpenSearchDep,
    embeddings_service: EmbeddingsDep,
    llm_client: LLMDep,
    langfuse_tracer: LangfuseDep,
    cache_client: CacheDep,
) -> StreamingResponse:
    """Clean streaming RAG endpoint with support for both document types."""

    async def generate_stream():
        rag_tracer = RAGTracer(langfuse_tracer)
        start_time = time.time()

        with rag_tracer.trace_request("api_user", request.query) as trace:
            try:
                # Check exact cache first
                if cache_client:
                    try:
                        cached_response = await cache_client.find_cached_response(request)
                        if cached_response:
                            logger.info("Returning cached response for exact streaming query match")

                            # Send metadata first (same format as non-cached)
                            metadata_response = {
                                "sources": cached_response.sources,
                                "chunks_used": cached_response.chunks_used,
                                "search_mode": cached_response.search_mode,
                            }
                            yield f"data: {json.dumps(metadata_response)}\n\n"

                            # Stream the cached response in chunks
                            for chunk in cached_response.answer.split():
                                yield f"data: {json.dumps({'chunk': chunk + ' '})}\n\n"

                            # Send completion signal with just the final answer
                            yield f"data: {json.dumps({'answer': cached_response.answer, 'done': True})}\n\n"
                            return
                    except Exception as e:
                        logger.warning(f"Cache check failed, proceeding with normal flow: {e}")

                # Route to appropriate search based on document_type
                if request.document_type == "financial":
                    chunks, sources, _ = await _prepare_chunks_and_sources_financial(
                        request, financial_opensearch_client, embeddings_service, rag_tracer, trace
                    )
                else:  # "arxiv"
                    chunks, sources, _ = await _prepare_chunks_and_sources_arxiv(
                        request, opensearch_client, embeddings_service, rag_tracer, trace
                    )

                if not chunks:
                    no_results_message = (
                        "No relevant financial documents found."
                        if request.document_type == "financial"
                        else "No relevant papers found."
                    )
                    yield f"data: {json.dumps({'answer': no_results_message, 'sources': [], 'done': True})}\n\n"
                    return

                # Send metadata first
                search_mode = "bm25" if not request.use_hybrid else "hybrid"
                metadata_response = {"sources": sources, "chunks_used": len(chunks), "search_mode": search_mode}
                yield f"data: {json.dumps(metadata_response)}\n\n"

                # Build prompt
                with rag_tracer.trace_prompt_construction(trace, chunks) as prompt_span:
                    from src.services.ollama.prompts import RAGPromptBuilder

                    prompt_builder = RAGPromptBuilder()
                    final_prompt = prompt_builder.create_rag_prompt(
                        request.query,
                        chunks,
                        document_type=request.document_type
                    )
                    rag_tracer.end_prompt(prompt_span, final_prompt)

                # Stream generation
                with rag_tracer.trace_generation(trace, request.model, final_prompt) as gen_span:
                    full_response = ""
                    async for chunk in llm_client.generate_rag_answer_stream(
                        query=request.query,
                        chunks=chunks,
                        model=request.model,
                        document_type=request.document_type
                    ):
                        if chunk.get("response"):
                            text_chunk = chunk["response"]
                            full_response += text_chunk
                            yield f"data: {json.dumps({'chunk': text_chunk})}\n\n"

                        if chunk.get("done", False):
                            rag_tracer.end_generation(gen_span, full_response, request.model)
                            yield f"data: {json.dumps({'answer': full_response, 'done': True})}\n\n"
                            break

                rag_tracer.end_request(trace, full_response, time.time() - start_time)

                # Store response in exact match cache
                if cache_client and full_response:
                    try:
                        search_mode = "bm25" if not request.use_hybrid else "hybrid"
                        response_to_cache = AskResponse(
                            query=request.query,
                            answer=full_response,
                            sources=sources,
                            chunks_used=len(chunks),
                            search_mode=search_mode,
                        )
                        await cache_client.store_response(request, response_to_cache)
                    except Exception as e:
                        logger.warning(f"Failed to store streaming response in cache: {e}")

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(), media_type="text/plain", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
