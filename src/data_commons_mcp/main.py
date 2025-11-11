"""HTTP API to deploy the EOSC Data Commons search agent."""

import json
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from ag_ui.core import (
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageChunkEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from langchain.chat_models import BaseChatModel
from langchain.messages import AnyMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp.types import TextContent
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

from data_commons_mcp.config import settings
from data_commons_mcp.logging import BLUE, BOLD, RESET, YELLOW
from data_commons_mcp.mcp_server import mcp
from data_commons_mcp.models import (
    AgentInput,
    LangChainRerankingOutputMsg,
    LangChainResponseMetadata,
    OpenSearchResults,
    RankedSearchResponse,
    RerankingOutput,
    TokenUsageMetadata,
)
from data_commons_mcp.prompts import RERANK_PROMPT, SUMMARIZE_PROMPT, TOOL_CALL_PROMPT
from data_commons_mcp.utils import (
    file_logger,
    get_langchain_msgs,
    get_system_prompt,
    load_chat_model,
    logger,
    sse_event,
)

# Get the MCP server Starlette app, and mount our routes to it
app = mcp.streamable_http_app()

if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

mcp_client = MultiServerMCPClient(
    {
        "data-commons-mcp": {
            "url": f"{settings.server_url}/mcp",
            "transport": "streamable_http",
        }
    }
)

logger.info(f"""💬 {BOLD}{BLUE}Search UI{RESET} started on {BOLD}{YELLOW}{settings.server_url}{RESET}
⚡️ Streamable HTTP MCP server started on {BOLD}{settings.server_url}/mcp{RESET}
🔎 Using OpenSearch service on {BOLD}{settings.opensearch_url}{RESET}""")


async def chat_handler(request: Request) -> StreamingResponse:
    """Chat with the assistant main endpoint."""
    auth_header = request.headers.get("Authorization", "")
    if settings.chat_api_key and (not auth_header or not auth_header.startswith("Bearer ")):
        raise ValueError("Missing or invalid Authorization header")
    if settings.chat_api_key and auth_header.split(" ")[1] != settings.chat_api_key:
        raise ValueError("Invalid API key")

    return StreamingResponse(
        stream_chat_response(AgentInput.model_validate(await request.json())),
        media_type="text/event-stream",
    )


async def stream_chat_response(request: AgentInput) -> AsyncGenerator[str, None]:
    """Stream the chat response with tool calls, reranking, and results."""
    msg_id = str(uuid.uuid4())
    token_usage = TokenUsageMetadata()
    yield sse_event(RunStartedEvent(thread_id=request.thread_id, run_id=request.run_id))
    yield sse_event(TextMessageStartEvent(message_id=msg_id, role="assistant"))

    # Get tools from the MCP client
    tools = await mcp_client.get_tools()

    # Get model with tools for the initial query
    llm = load_chat_model(request.model)
    llm_with_tools = llm.bind_tools(tools)

    # Step 1: Call LLM to get tool calls
    msgs = get_langchain_msgs(request.messages)
    tc_llm_resp = llm_with_tools.invoke([get_system_prompt(TOOL_CALL_PROMPT), *msgs])
    token_usage += LangChainResponseMetadata.model_validate(tc_llm_resp.response_metadata).token_usage

    if tc_llm_resp.content and isinstance(tc_llm_resp.content, str):
        # If tc_llm_resp has text send it as a TextMessage content alongside tool calls
        yield sse_event(
            TextMessageChunkEvent(
                delta=tc_llm_resp.content,
            )
        )

    # Step 2: Execute each tool and collect search results and textual outputs
    search_results = OpenSearchResults(total_found=0, hits=[])
    tool_text_outputs: list[str] = []
    async with mcp_client.session("data-commons-mcp") as session:
        for tool_call in tc_llm_resp.tool_calls:
            tool_call_id = tool_call["name"]
            yield sse_event(
                ToolCallStartEvent(
                    tool_call_id=tool_call_id, tool_call_name=tool_call["name"], parent_message_id=msg_id
                )
            )
            yield sse_event(ToolCallArgsEvent(tool_call_id=tool_call_id, delta=json.dumps(tool_call["args"])))
            tc_exec_res = await session.call_tool(tool_call["name"], tool_call["args"])

            if tc_exec_res.structuredContent:
                # Handle structured content, try to parse as `OpenSearchResults`
                try:
                    tool_results = OpenSearchResults(**tc_exec_res.structuredContent)
                    search_results.hits.extend(tool_results.hits)
                    search_results.total_found += tool_results.total_found
                finally:
                    tool_results_str = json.dumps(tc_exec_res.structuredContent)
                yield sse_event(
                    ToolCallResultEvent(
                        message_id=msg_id, tool_call_id=tool_call_id, content=tool_results_str, role="tool"
                    )
                )
            elif tc_exec_res.content:
                # Handle if text content is sent back
                for resp_content in tc_exec_res.content:
                    if isinstance(resp_content, TextContent):
                        # Stream the raw tool text back to the UI, and record it for fallback summarization
                        yield sse_event(
                            ToolCallResultEvent(
                                message_id=msg_id, tool_call_id=tool_call_id, content=resp_content.text, role="tool"
                            )
                        )
                        try:
                            if resp_content.text:
                                tool_text_outputs.append(resp_content.text)
                        except Exception as exc:
                            logger.exception("Failed to record tool text output: %s", exc)

            yield sse_event(ToolCallEndEvent(tool_call_id=tool_call_id))

    # Handle if there were tool calls output, but no search results: ask the LLM to summarize tools outputs
    if tc_llm_resp.tool_calls and search_results.total_found == 0 and tool_text_outputs:
        summary_msgs: list[AnyMessage] = [
            get_system_prompt(SUMMARIZE_PROMPT),
            *msgs,
            HumanMessage(
                content=(
                    "The following tool outputs were produced when handling the user's query:\n\n"
                    + "\n\n---\n\n".join(tool_text_outputs)
                    + "\n\nPlease provide a concise summary for the user explaining what the tools returned and any recommendation or next steps."
                )
            ),
        ]
        try:
            fallback_tool_id = "search_summary"
            yield sse_event(
                ToolCallStartEvent(
                    tool_call_id=fallback_tool_id, tool_call_name=fallback_tool_id, parent_message_id=msg_id
                )
            )
            summary_resp = llm.invoke(summary_msgs)
            token_usage += LangChainResponseMetadata.model_validate(summary_resp.response_metadata).token_usage
            # Send the summary back as a ToolCallResult-like event so the UI can display it
            # NOTE: use TextMessageChunkEvent?
            yield sse_event(
                ToolCallResultEvent(
                    message_id=msg_id, tool_call_id=fallback_tool_id, content=str(summary_resp.content), role="tool"
                )
            )
            yield sse_event(ToolCallEndEvent(tool_call_id=fallback_tool_id))
            return
        except Exception as e:
            logger.error(f"Fallback summarization failed: {e}")

    # Step 3: If no results found or no tool calls, handle early exit
    if not tc_llm_resp.tool_calls or search_results.total_found == 0:
        yield sse_event(TextMessageEndEvent(message_id=msg_id))
        yield sse_event(RunFinishedEvent(thread_id=request.thread_id, run_id=request.run_id))
        return

    # print(json.dumps(search_results.model_dump(), indent=2))

    # Step 4: Rerank search results using LLM with structured output
    rerank_tc_id = "rerank_results"
    yield sse_event(
        ToolCallStartEvent(
            tool_call_id=rerank_tc_id,
            tool_call_name="rerank_results",
            parent_message_id=msg_id,
        )
    )
    final_response = await rerank_search_results(
        llm,
        msgs,
        search_results,
        token_usage,
    )
    yield sse_event(
        ToolCallResultEvent(
            message_id=msg_id,
            tool_call_id=rerank_tc_id,
            content=final_response.model_dump_json(by_alias=True),
            role="tool",
        )
    )
    yield sse_event(ToolCallEndEvent(tool_call_id=rerank_tc_id))
    yield sse_event(TextMessageEndEvent(message_id=msg_id))
    yield sse_event(RunFinishedEvent(thread_id=request.thread_id, run_id=request.run_id))
    file_logger.info(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "token_usage": token_usage.model_dump(),
                "input": request.model_dump(),
                "response": final_response.model_dump(),
            }
        )
    )
    logger.info(f'/chat "{request.messages[-1].content}" | {token_usage.model_dump()}')


app.router.add_route("/chat", chat_handler, methods=["POST"])


async def rerank_search_results(
    llm: BaseChatModel,
    chat_messages: list[AnyMessage],
    search_results: OpenSearchResults,
    token_usage: TokenUsageMetadata,
) -> RankedSearchResponse:
    """Rerank search results using LLM with structured output.

    Args:
        model: The LLM model to use for reranking
        chat_messages: Original chat messages for context
        search_results: Search results to rerank

    Returns:
        RankedSearchResponse with reranked hits and summary
    """
    # Format the context for the LLM
    last_msg = chat_messages[-1] if chat_messages else None
    last_msg_content = last_msg.content if last_msg and isinstance(last_msg.content, str) else ""
    formatted_context = f"Found {search_results.total_found} datasets relevant to the query '{last_msg_content}':\n\n"
    for i, hit in enumerate(search_results.hits[: settings.reranking_results_count]):
        formatted_context += f"{i + 1}. **{hit.id}**\n"
        formatted_context += f"   {' | '.join([title.title for title in hit.source.titles])}\n"
        if hit.source.dates:
            formatted_context += (
                f"   Dates: {' | '.join([f'{date.date_type}: {date.date}' for date in hit.source.dates])}\n"
            )
        if hit.source.creators:
            formatted_context += f"   Authors: {', '.join([creator.creator_name for creator in hit.source.creators])}\n"
        if hit.source.subjects:
            formatted_context += f"   Keywords: {', '.join([subj.subject for subj in hit.source.subjects])}\n"
        formatted_context += f"   Description: {hit.description}\n\n"

    rerank_msgs: list[AnyMessage] = [
        get_system_prompt(RERANK_PROMPT),
        *chat_messages,
        HumanMessage(content=formatted_context),
    ]
    try:
        # Call LLM with structured output for reranking
        llm_structured_rerank = llm.with_structured_output(RerankingOutput, method="function_calling", include_raw=True)
        rerank_resp = LangChainRerankingOutputMsg.model_validate(llm_structured_rerank.invoke(rerank_msgs))
        token_usage += LangChainResponseMetadata.model_validate(rerank_resp.raw.response_metadata).token_usage

        # Add scores to all datasets from search results
        score_lookup = {hit.url: hit.score for hit in rerank_resp.parsed.hits}
        # print(f"Rerank response: {score_lookup}")
        for hit in search_results.hits:
            hit.score = score_lookup.get(hit.id, 0.0)

        # Sort hits by score in descending order
        search_results.hits.sort(key=lambda h: h.score or 0.0, reverse=True)
        return RankedSearchResponse(summary=rerank_resp.parsed.summary, hits=search_results.hits)
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Fallback: return results as-is without reranking
        return RankedSearchResponse(
            summary=f"Found {search_results.total_found} relevant datasets.",
            hits=search_results.hits,
        )


# Serve website built using vite
app.mount(
    "/assets",
    StaticFiles(directory="src/data_commons_mcp/webapp/assets"),
    name="static",
)


async def ui_handler(request: Request) -> FileResponse:
    """Serve the chat UI HTML file directly."""
    return FileResponse("src/data_commons_mcp/webapp/index.html")


# Serve index.html for root and any other unmatched GET paths, so a SPA can handle routing
app.router.add_route("/", ui_handler, methods=["GET"])
app.router.add_route("/{path:path}", ui_handler, methods=["GET"])
