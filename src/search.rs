use axum::extract::State;
use axum::response::sse;
use axum::{
    extract::Json,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use utoipa::ToSchema;

use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder},
    chat::{ChatMessage, StructuredOutputFormat},
};

use rmcp::{
    ServiceExt,
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    transport::StreamableHttpClientTransport,
};

use crate::AppState;
use crate::error::{AppError, AppResult};
use crate::mcp::{McpSearchResult, SearchHit};
use crate::utils::{SearchLog, get_llm_config};

const SYSTEM_PROMPT_TOOLS: &str = r#"You are an assistant that help users find datasets and tools for scientific research.
Define if you need to use one of the tool provided to get more context to answer the user request, or directly answer the user question.
If the user provides a simple question (just a word or concept), you should prioritize searching for relevant datasets.
"#;
const SYSTEM_PROMPT_RESOLUTION: &str = r#"You are an assistant that help users find datasets and tools for scientific research.
Given the user question and datasets retrieved from the search API, summarize the findings in 1 sentence,
extract which datasets might be the most interesting to answer the user question, and give them a relevance score between 0 and 1
"#;
// TODO: your goal is to answer questions about data, nothing else

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct SearchInput {
    pub messages: Vec<ApiChatMessage>,
    // #[schema(example = "groq/moonshotai/kimi-k2-instruct")]
    // #[schema(example = "openai/gpt-4.1-nano")]
    #[schema(example = "einfracz/qwen3-coder")]
    // #[schema(example = "mistralai/mistral-small-latest")]
    pub model: String,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize, Serialize, ToSchema, Clone)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    pub summary: String,
    // pub usage: Option<UsageInfo>,
}

// /// Token usage and cost information for LLM calls
// #[derive(Debug, Deserialize, Serialize, ToSchema, Clone)]
// pub struct UsageInfo {
//     pub prompt_tokens: u32,
//     pub completion_tokens: u32,
//     pub total_tokens: u32,
//     pub total_cost_usd: f64,
// }

/// Wrapper around llm::ChatMessage that implements ToSchema for API documentation
#[derive(Debug, Deserialize, Serialize, ToSchema, Clone)]
pub struct ApiChatMessage {
    #[schema(example = "user")]
    pub role: String,
    #[schema(
        example = "insulin"
        // example = "I am looking for data about glucose level evolution in liver on people with type 1 diabetes in Europe between 1980 and 2020"
    )]
    pub content: String,
}

impl ApiChatMessage {
    /// Convert to llm::ChatMessage for use with the LLM client
    pub fn to_chat_message(&self) -> ChatMessage {
        match self.role.as_str() {
            "user" => ChatMessage::user().content(&self.content).build(),
            "assistant" => ChatMessage::assistant().content(&self.content).build(),
            _ => ChatMessage::assistant().content(&self.content).build(), // Default to assistant
        }
    }
    // /// Create from role and content
    // pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
    //     Self {
    //         role: role.into(),
    //         content: content.into(),
    //     }
    // }
    // /// Create an assistant message
    // pub fn assistant(content: impl Into<String>) -> Self {
    //     Self::new("assistant", content)
    // }
}

/// OpenAI-compatible streaming response chunk
#[derive(Debug, Serialize, ToSchema)]
struct StreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize, ToSchema)]
struct StreamChoice {
    index: u32,
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
struct StreamDelta {
    role: Option<String>,
    content: Option<String>,
    function_call: Option<serde_json::Value>,
}

/// LLM response format
#[derive(Debug, Deserialize, Serialize, ToSchema)]
struct LLMStructuredOutput {
    summary: String,
    hits: Vec<ScoredHits>,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
struct ScoredHits {
    url: String,
    score: f64,
}

/// Workflow manager for handling search operations with fragmented steps
pub struct SearchWorkflow {
    pub mcp_client:
        rmcp::service::RunningService<rmcp::RoleClient, rmcp::model::InitializeRequestParam>,
    pub llm_backend: LLMBackend,
    pub llm_api_key: String,
    pub llm_model: String,
    pub llm_url: Option<String>,
    pub msg_id: String,
    pub created: u64,
}

impl SearchWorkflow {
    /// Initialize a new search workflow (query LLM with MCP tools)
    pub async fn new(model: String, bind_address: String) -> AppResult<Self> {
        let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let msg_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        // Use bind_address for MCP connection
        let transport =
            StreamableHttpClientTransport::from_uri(format!("http://{bind_address}/mcp"));
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: "MCP streamable HTTP client".to_string(),
                version: "0.0.1".to_string(),
            },
        };
        let client = match client_info.serve(transport).await {
            Ok(client) => client,
            Err(e) => {
                tracing::error!("client error: {:?}", e);
                return Err(AppError::Llm(format!(
                    "MCP client initialization failed: {e}"
                )));
            }
        };
        let (llm_backend, llm_api_key, llm_model, llm_url) =
            get_llm_config(&model).map_err(AppError::Llm)?;
        Ok(Self {
            mcp_client: client,
            llm_backend,
            llm_api_key,
            llm_model,
            llm_url,
            msg_id,
            created,
        })
    }

    /// Step 1: Check if tool calls are needed and execute them
    pub async fn execute_tool_calls(
        &self,
        messages: &[ApiChatMessage],
    ) -> AppResult<(String, Option<Vec<llm::ToolCall>>, McpSearchResult)> {
        // Convert messages to LLM ChatMessage format
        let chat_messages: Vec<ChatMessage> =
            messages.iter().map(|msg| msg.to_chat_message()).collect();

        // Configure LLM client with dynamic tools from MCP
        let mut llm_builder = LLMBuilder::new()
            .backend(self.llm_backend.clone())
            .api_key(&self.llm_api_key)
            .model(&self.llm_model)
            .max_tokens(1024)
            .temperature(0.1)
            .system(SYSTEM_PROMPT_TOOLS);
        if let Some(url) = &self.llm_url {
            llm_builder = llm_builder.base_url(url);
        }

        // Convert MCP tools to LLM functions and add them to the llm builder
        let tools = self.mcp_client.list_tools(Default::default()).await?;
        for tool in &tools.tools {
            let schema_value = serde_json::Value::Object(tool.input_schema.as_ref().clone());
            let function = FunctionBuilder::new(tool.name.to_string())
                .description(tool.description.as_deref().unwrap_or(""))
                .json_schema(schema_value);
            llm_builder = llm_builder.function(function);
        }
        let llm = llm_builder.build().expect("Failed to build LLM client");

        // Query the LLM to check if tool call necessary
        let (response_text, tool_calls) =
            match llm.chat_with_tools(&chat_messages, llm.tools()).await {
                Ok(response) => {
                    // tracing::debug!("LLM tool call response: {response:#?}");
                    (
                        response.text().unwrap_or_default().to_string(),
                        response.tool_calls(),
                    )
                }
                Err(e) => {
                    // Stream error as message
                    (
                        e.to_string(),
                        None,
                    )
                }
            };

        let mut search_results = McpSearchResult {
            total_found: 0,
            hits: vec![],
        };
        // Execute each tool call if any
        if let Some(tc) = &tool_calls {
            for call in tc {
                tracing::debug!("Calling tool {}", call.function.name);
                let arguments =
                    match serde_json::from_str::<serde_json::Value>(&call.function.arguments) {
                        Ok(value) => value.as_object().cloned(),
                        Err(_) => None,
                    };

                // Call MCP tools
                let tool_results = self
                    .mcp_client
                    .call_tool(CallToolRequestParam {
                        name: call.function.name.clone().into(),
                        arguments,
                    })
                    .await?;

                // Handle structured content if present
                if let Some(structured) = &tool_results.structured_content {
                    // serde_json::from_value::<McpSearchResult>(structured.clone())?
                    match serde_json::from_value::<McpSearchResult>(structured.clone()) {
                        Ok(new_search_results) => {
                            // Accumulate results from multiple tool calls
                            search_results.hits.extend(new_search_results.hits);
                            search_results.total_found += new_search_results.total_found;
                        }
                        Err(e) => {
                            return Err(AppError::Serde(e));
                        }
                    }
                } else {
                    // Fallback: plain content
                    tracing::warn!(
                        "Tool {} returned plain text content: {:?}",
                        call.function.name,
                        tool_results.content
                    );
                    // let plain_content = tool_results
                    //     .content
                    //     .iter()
                    //     .flat_map(|annotated_vec| annotated_vec.iter())
                    //     .filter_map(|annotated| match &annotated.raw {
                    //         rmcp::model::RawContent::Text(text_content) => {
                    //             Some(text_content.text.as_str())
                    //         }
                    //         _ => None,
                    //     })
                    //     .collect::<Vec<_>>()
                    //     .join(" ");
                }
            }
        }
        Ok((response_text, tool_calls, search_results))
    }

    /// Step 2: Generate summary and scores using LLM with structured output, then create final response
    pub async fn generate_summary_and_scores(
        &self,
        messages: &[ApiChatMessage],
        search_results: McpSearchResult,
    ) -> AppResult<SearchResponse> {
        if search_results.total_found == 0 || search_results.hits.is_empty() {
            return Err(AppError::NoDataFound(
                "No datasets found for your query.".to_string(),
            ));
        }
        let last_message_content = messages
            .last()
            .map(|msg| msg.content.as_str())
            .unwrap_or("");

        // Format the context for the LLM based on tool call results
        let mut formatted_context = format!(
            "Found {} datasets relevant to the query '{}':\n\n",
            search_results.total_found, last_message_content
        );
        for (i, dataset) in search_results.hits.iter().enumerate() {
            formatted_context.push_str(&format!("{}. **{}**\n", i + 1, dataset.title));
            formatted_context.push_str(&format!("   Zenodo: {}\n", dataset.url));
            formatted_context.push_str(&format!("   Published: {}\n", dataset.publication_date));
            if let Some(creators) = &dataset.creators {
                if !creators.is_empty() {
                    formatted_context.push_str(&format!("   Authors: {}\n", creators.join(", ")));
                }
            }
            if let Some(keywords) = &dataset.keywords {
                if !keywords.is_empty() {
                    formatted_context.push_str(&format!("   Keywords: {}\n", keywords.join(", ")));
                }
            }
            formatted_context.push_str(&format!("   Description: {}\n\n", dataset.description));
        }

        // Convert messages and add formatted context
        let mut chat_messages: Vec<ChatMessage> =
            messages.iter().map(|msg| msg.to_chat_message()).collect();
        chat_messages.push(ChatMessage::user().content(&formatted_context).build());

        // Create LLM client with structured output schema
        let schema: StructuredOutputFormat = serde_json::from_str(SEARCH_OUTPUT_SCHEMA)?;
        let mut llm_builder = LLMBuilder::new()
            .backend(self.llm_backend.clone())
            .api_key(&self.llm_api_key)
            .model(&self.llm_model)
            .max_tokens(512)
            .temperature(0.1)
            .system(SYSTEM_PROMPT_RESOLUTION)
            .schema(schema);
        if let Some(url) = &self.llm_url {
            llm_builder = llm_builder.base_url(url);
        }

        let llm_resolution = llm_builder.build()
            .expect("Failed to build LLM client");

        // Send chat request using additional infos retrieved by the tool call
        let response_text = match llm_resolution.chat(&chat_messages).await {
            Ok(response) => response.text().unwrap_or_default().to_string(),
            Err(e) => {
                tracing::error!("Chat error!!!: {e}");
                // return Err(AppError::Llm(e.to_string()));
                e.to_string()
            }
        };

        // Parse the response
        let llm_response = match serde_json::from_str::<LLMStructuredOutput>(&response_text) {
            Ok(llm_response) => llm_response,
            Err(e) => {
                tracing::error!("Failed to parse LLM response as JSON: {}", e);
                return Err(AppError::Serde(e));
            }
        };

        // Create a lookup map for LLM scores by URL
        let score_lookup: std::collections::HashMap<String, f64> = llm_response
            .hits
            .into_iter()
            .map(|llm_dataset| (llm_dataset.url, llm_dataset.score))
            .collect();

        let mut search_results = search_results;
        // Add scores to all datasets from search results
        for hit in &mut search_results.hits {
            hit.score = Some(score_lookup.get(&hit.url).copied().unwrap_or(0.0));
        }
        // Sort hits by score in descending order (highest score first)
        search_results.hits.sort_by(|a, b| {
            let score_a = a.score.unwrap_or(0.0);
            let score_b = b.score.unwrap_or(0.0);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(SearchResponse {
            hits: search_results.hits,
            summary: llm_response.summary,
        })
    }

    /// Create SSE event for streaming responses
    pub fn create_sse_event(
        &self,
        event_type: &str,
        data: impl Serialize,
    ) -> AppResult<sse::Event> {
        Ok(sse::Event::default()
            .event(event_type)
            .data(serde_json::to_string(&data)?))
    }

    /// Create streaming chunk for OpenAI compatibility
    pub fn create_stream_chunk(
        &self,
        content: Option<String>,
        finish_reason: Option<String>,
    ) -> AppResult<sse::Event> {
        let chunk = StreamChunk {
            id: self.msg_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.llm_model.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: if content.is_some() {
                        Some("assistant".to_string())
                    } else {
                        None
                    },
                    content,
                    function_call: None,
                },
                finish_reason,
            }],
        };
        Ok(sse::Event::default()
            .event("message")
            .data(serde_json::to_string(&chunk)?))
    }

    /// Log search operation response with execution time
    pub fn log_response(
        &self,
        stream: bool,
        conversation: Vec<ApiChatMessage>,
        response: SearchResponse,
        execution_time_ms: u64,
    ) {
        let search_log = SearchLog::new(
            self.msg_id.clone(),
            self.llm_model.clone(),
            stream,
            conversation,
            response,
            execution_time_ms,
        );
        if let Err(e) = search_log.write_to_file() {
            tracing::error!("Failed to write search log: {:?}", e);
        }
    }
}

// Define a simple JSON schema for structured output
const SEARCH_OUTPUT_SCHEMA: &str = r#"
    {
        "name": "search_results",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of the findings in 1 sentence"
                },
                "hits": {
                    "type": "array",
                    "description": "List of most relevant hits",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL of the item"
                            },
                            "score": {
                                "type": "number",
                                "description": "Relevance score of the hit based on the search query (between 0 and 1)"
                            }
                        },
                        "additionalProperties": false,
                        "required": ["url", "score"]
                    }
                }
            },
            "additionalProperties": false,
            "required": ["summary", "hits"]
        },
        "strict": true
    }
"#;

/// Search data relevant to a user question in a conversation
#[utoipa::path(
    post,
    path = "/search",
    request_body(content = SearchInput, description = "List of messages in the chat"),
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 200, description = "Search results (SSE stream)", content_type = "text/event-stream", body = StreamChunk),
        (status = 401, description = "Unauthorized"),
        (status = 500, description = "Internal server error")
    ),
)]
pub async fn search_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(resp): Json<SearchInput>,
) -> Response {
    if resp.stream {
        streaming_search_handler(headers, resp, state)
            .await
            .into_response()
    } else {
        regular_search_handler(headers, resp, state)
            .await
            .into_response()
    }
}

/// Streaming search handler for SSE responses
async fn streaming_search_handler(
    headers: HeaderMap,
    resp: SearchInput,
    state: AppState,
) -> impl IntoResponse {
    // Validate API key only if SEARCH_API_KEY is set
    if let Ok(secret_key) = std::env::var("SEARCH_API_KEY") {
        let auth_header = headers.get("authorization");
        let auth_value = auth_header.and_then(|h| h.to_str().ok()).unwrap_or("");
        if auth_value != secret_key {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "unauthorized"})),
            )
                .into_response();
        }
    }
    let stream = create_search_stream(resp, state);
    Sse::new(stream)
        // .keep_alive(
        //     sse::KeepAlive::new()
        //         .interval(std::time::Duration::from_secs(10)),
        //         // .text("keep-alive-text"),
        // )
        .into_response()
}

/// Helper function to create error SSE events
fn send_error_event(error_message: &str) -> AppResult<sse::Event> {
    Ok(sse::Event::default()
        .event("error")
        .data(serde_json::to_string(&serde_json::json!({
            "error": error_message.to_string(),
        }))?))
}

/// Create a streaming search response
fn create_search_stream(
    resp: SearchInput,
    state: AppState,
) -> impl Stream<Item = AppResult<sse::Event>> {
    async_stream::stream! {
        let start_time = SystemTime::now();
        // Initialize the workflow
        let workflow = match SearchWorkflow::new(resp.model.clone(), state.bind_address.clone()).await {
            Ok(workflow) => workflow,
            Err(e) => {
                tracing::error!("Failed to initialize workflow: {:?}", e);
                yield Ok(send_error_event("Error connecting to search service")?);
                return;
            }
        };
        // Step 1: Execute tool calls if needed
        let (response_txt, tool_calls, search_results) = match workflow.execute_tool_calls(&resp.messages).await {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("Tool call execution failed: {:?}", e);
                yield Ok(send_error_event(&format!("Error searching for datasets: {e}"))?);
                return;
            }
        };
        // Stream tool call requested event
        yield Ok(workflow.create_sse_event("tool_call_requested", &tool_calls)?);
        // If no tool calls were made, handle as direct response
        if tool_calls.is_none() || search_results.total_found == 0 {
            let execution_time = start_time.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
            if tool_calls.is_none() {
                // Direct response without tool calls
                let response = SearchResponse {
                    hits: vec![],
                    summary: response_txt,
                };
                yield Ok(workflow.create_stream_chunk(
                    Some(response.summary.clone()),
                    Some("stop".to_string())
                )?);
                workflow.log_response(
                    resp.stream,
                    resp.messages.clone(),
                    response,
                    execution_time,
                );
            } else {
                // No results found
                let response = SearchResponse {
                    hits: vec![],
                    summary: "Nothing found for your query.".to_string(),
                };
                workflow.log_response(
                    resp.stream,
                    resp.messages.clone(),
                    response,
                    execution_time,
                );
                yield Ok(send_error_event("Nothing found for your query.")?);
            }
            return;
        }
        // Stream the tool results (without scores or summary)
        yield Ok(workflow.create_sse_event("tool_call_result", &search_results)?);

        // Step 2: Generate and stream summary and scores using LLM
        let final_response = match workflow.generate_summary_and_scores(&resp.messages, search_results).await {
            Ok(response) => response,
            Err(e) => {
                // Fallback if LLM processing fails
                yield Ok(send_error_event(&format!("{e}"))?);
                return;
            }
        };
        yield Ok(workflow.create_sse_event("search_response", &final_response)?);

        // Calculate execution time and log the search resp
        let execution_time = start_time.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
        workflow.log_response(
            resp.stream,
            resp.messages.clone(),
            final_response,
            execution_time,
        );
        yield Ok(workflow.create_stream_chunk(None, Some("stop".to_string()))?);
    }
}

/// Search handler for non-streaming responses
async fn regular_search_handler(
    headers: HeaderMap,
    resp: SearchInput,
    state: AppState,
) -> impl IntoResponse {
    let start_time = SystemTime::now();
    // Validate API key only if SEARCH_API_KEY is set
    if let Ok(secret_key) = std::env::var("SEARCH_API_KEY") {
        let auth_header = headers.get("authorization");
        let auth_value = auth_header.and_then(|h| h.to_str().ok()).unwrap_or("");
        if auth_value != secret_key {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "unauthorized"})),
            );
        }
    }
    // Initialize the workflow
    let workflow = match SearchWorkflow::new(resp.model.clone(), state.bind_address.clone()).await {
        Ok(workflow) => workflow,
        Err(e) => {
            tracing::error!("Failed to initialize workflow: {:?}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "Error connecting to search service"})),
            );
        }
    };
    // Step 1: Execute tool calls if needed
    let (_response_txt, _tool_calls, search_results) =
        match workflow.execute_tool_calls(&resp.messages).await {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("Tool call execution failed: {:?}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": "Error searching for datasets"})),
                );
            }
        };
    // If no datasets were found, return early
    if search_results.total_found == 0 || search_results.hits.is_empty() {
        let execution_time = start_time
            .elapsed()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let final_response = SearchResponse {
            hits: vec![],
            summary: "No datasets found for your query.".to_string(),
        };
        workflow.log_response(
            resp.stream,
            resp.messages.clone(),
            final_response.clone(),
            execution_time,
        );
        return (StatusCode::OK, Json(serde_json::json!(final_response)));
    }

    // Step 2: Generate summary and scores using LLM
    let final_response = match workflow
        .generate_summary_and_scores(&resp.messages, search_results)
        .await
    {
        Ok(response) => response,
        Err(e) => {
            tracing::error!("LLM processing failed: {e:?}");
            // Fallback response without scoring
            SearchResponse {
                hits: vec![],
                summary: format!(
                    "Found datasets for your query, but could not process relevance scores. {e:?}"
                ),
            }
        }
    };
    // Calculate execution time and log the search
    let execution_time = start_time
        .elapsed()
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    workflow.log_response(
        resp.stream,
        resp.messages.clone(),
        final_response.clone(),
        execution_time,
    );
    (StatusCode::OK, Json(serde_json::json!(final_response)))
}
