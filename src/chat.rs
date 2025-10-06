use axum::extract::State;
use axum::response::sse;
use axum::{
    extract::Json,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
};
use futures_util::stream::Stream;
use llm::ToolCall;
use rmcp::model::CallToolResult;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::SystemTime;
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
pub struct ChatInput {
    pub messages: Vec<ApiChatMessage>,
    // #[schema(example = "groq/moonshotai/kimi-k2-instruct")]
    // #[schema(example = "openai/gpt-4.1-nano")]
    // #[schema(example = "einfracz/gpt-oss-120b")]
    #[schema(example = "einfracz/qwen3-coder")]
    // #[schema(example = "mistralai/mistral-small-latest")]
    pub model: Option<String>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize, Serialize, ToSchema, Clone)]
pub struct ChatSearchResponse {
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
    pub fn to_chat_msg(&self) -> ChatMessage {
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
}

impl SearchWorkflow {
    /// Initialize a new search workflow (query LLM with MCP tools)
    pub async fn new(model: Option<String>, bind_address: String) -> AppResult<Self> {
        // let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        let msg_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
        // Use bind_address for MCP connection
        let transport =
            StreamableHttpClientTransport::from_uri(format!("http://{bind_address}/mcp"));
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                version: "0.0.1".to_string(),
                name: "MCP streamable HTTP client".to_string(),
                title: Some("Data Commons MCP Client".to_string()),
                website_url: Some("https://github.com/EOSC-Data-Commons/data-commons-mcp".to_string()),
                icons: None,
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
            get_llm_config(&model.unwrap_or("einfracz/qwen3-coder".to_string()))
                .map_err(AppError::Llm)?;
        Ok(Self {
            mcp_client: client,
            llm_backend,
            llm_api_key,
            llm_model,
            llm_url,
            msg_id,
        })
    }

    /// Step 1: Check if tool calls are needed and execute them
    pub async fn request_tool_calls(
        &self,
        messages: &[ApiChatMessage],
    ) -> AppResult<(String, Option<Vec<llm::ToolCall>>)> {
        // Convert messages to LLM ChatMessage format
        let chat_messages: Vec<ChatMessage> =
            messages.iter().map(|msg| msg.to_chat_msg()).collect();

        // Configure LLM client with dynamic tools from MCP
        let mut llm_builder = LLMBuilder::new()
            .backend(self.llm_backend.clone())
            .api_key(&self.llm_api_key)
            .model(&self.llm_model)
            .max_tokens(1024)
            .temperature(0.1)
            // .tool_choice(ToolChoice::Any)  // NOTE: required for gpt-oss-120b to properly trigger too calls
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
                    (e.to_string(), None)
                }
            };
        Ok((response_text, tool_calls))
    }

    /// Execute a single tool call and return the results
    pub async fn execute_tool_call(&self, call: &ToolCall) -> AppResult<CallToolResult> {
        // tracing::debug!("Calling tool {}", call.function.name);
        let arguments = match serde_json::from_str::<serde_json::Value>(&call.function.arguments) {
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
        Ok(tool_results)
    }

    // /// Step 2: Execute tool calls
    // pub async fn execute_tool_calls(
    //     &self,
    //     tool_calls: &Option<Vec<ToolCall>>,
    // ) -> AppResult<McpSearchResult> {
    //     let mut search_results = McpSearchResult {
    //         total_found: 0,
    //         hits: vec![],
    //     };
    //     // Dict with tool_call_id as key, and results (search_results, or structured content JSON, or plain text), remove total_found
    //     // Execute each tool call if any
    //     if let Some(tc) = &tool_calls {
    //         for call in tc {
    //             tracing::debug!("Calling tool {}", call.function.name);
    //             let arguments =
    //                 match serde_json::from_str::<serde_json::Value>(&call.function.arguments) {
    //                     Ok(value) => value.as_object().cloned(),
    //                     Err(_) => None,
    //                 };

    //             // Call MCP tools
    //             let tool_results = self
    //                 .mcp_client
    //                 .call_tool(CallToolRequestParam {
    //                     name: call.function.name.clone().into(),
    //                     arguments,
    //                 })
    //                 .await?;

    //             // Handle structured content if present
    //             if let Some(structured) = &tool_results.structured_content {
    //                 // serde_json::from_value::<McpSearchResult>(structured.clone())?
    //                 match serde_json::from_value::<McpSearchResult>(structured.clone()) {
    //                     Ok(new_search_results) => {
    //                         // Accumulate results from multiple tool calls
    //                         search_results.hits.extend(new_search_results.hits);
    //                         search_results.total_found += new_search_results.total_found;
    //                     }
    //                     Err(e) => {
    //                         return Err(AppError::Serde(e));
    //                     }
    //                 }
    //             } else {
    //                 // TODO: fallback for plain text
    //                 tracing::warn!(
    //                     "Tool {} returned plain text content: {:?}",
    //                     call.function.name,
    //                     tool_results.content
    //                 );
    //                 // let plain_content = tool_results
    //                 //     .content
    //                 //     .iter()
    //                 //     .flat_map(|annotated_vec| annotated_vec.iter())
    //                 //     .filter_map(|annotated| match &annotated.raw {
    //                 //         rmcp::model::RawContent::Text(text_content) => {
    //                 //             Some(text_content.text.as_str())
    //                 //         }
    //                 //         _ => None,
    //                 //     })
    //                 //     .collect::<Vec<_>>()
    //                 //     .join(" ");
    //             }
    //         }
    //     }
    //     Ok(search_results)
    // }

    /// Step 2: Generate summary and scores using LLM with structured output, then create final response
    pub async fn generate_summary_and_scores(
        &self,
        messages: &[ApiChatMessage],
        search_results: McpSearchResult,
    ) -> AppResult<ChatSearchResponse> {
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
            messages.iter().map(|msg| msg.to_chat_msg()).collect();
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

        let llm_resolution = llm_builder.build().expect("Failed to build LLM client");

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

        Ok(ChatSearchResponse {
            hits: search_results.hits,
            summary: llm_response.summary,
        })
    }

    /// Log search operation response with execution time
    pub fn log_response(
        &self,
        stream: bool,
        conversation: Vec<ApiChatMessage>,
        response: ChatSearchResponse,
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
    path = "/chat",
    request_body(content = ChatInput, description = "Chat input object"),
    responses(
        (status = 200, description = "Chat results", body = ChatSearchResponse),
        (status = 200, description = "Chat results (SSE stream)", content_type = "text/event-stream", body = StreamChunk),
        (status = 401, description = "Unauthorized"),
        (status = 500, description = "Internal server error")
    ),
)]
pub async fn chat_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(resp): Json<ChatInput>,
) -> Response {
    if resp.stream {
        streaming_chat_handler(headers, resp, state)
            .await
            .into_response()
    } else {
        regular_chat_handler(headers, resp, state)
            .await
            .into_response()
    }
}

/// Streaming search handler for SSE responses
async fn streaming_chat_handler(
    headers: HeaderMap,
    resp: ChatInput,
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
    let stream = create_chat_stream(resp, state);
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

// TODO: support AG-UI streaming format https://docs.ag-ui.com/concepts/messages#streaming-messages
// Streaming events docs: https://docs.ag-ui.com/concepts/events#toolcallresult
// Events types code reference: https://github.com/ag-ui-protocol/ag-ui/blob/be23d7e6b86bdfc1252e2d2948ef5743ba2e613c/python-sdk/ag_ui/core/events.py#L20
// data: {"type":"RUN_STARTED","threadId":"t1","runId":"r1"}
// data: {"type":"TEXT_MESSAGE_START","messageId":"m1","role":"assistant"}
// data: {"type":"TOOL_CALL_START","toolCallId":"c1","toolCallName":"search_data","parentMessageId":"m1"}
// data: {"type":"TOOL_CALL_ARGS","toolCallId":"c1","delta":"{\"question\": \"insulin\"}"}
// data: {"type":"TOOL_CALL_END","toolCallId":"c1"}
// data: {"type":"TOOL_CALL_RESULT","id":"toolmsg1","role":"tool","toolCallId":"c1","content":"{\"total_found\":5,…}"}
// data: {"type":"TEXT_MESSAGE_CONTENT","messageId":"m1","delta":"Here are the results I found:"}
// data: {"type":"TEXT_MESSAGE_END","messageId":"m1"}
// data: {"type":"RUN_FINISHED","threadId":"t1","runId":"r1"}
// NOTE: using `sse_event` for an AG-UI compliant streaming format

/// Create a streaming search response
fn create_chat_stream(
    resp: ChatInput,
    state: AppState,
) -> impl Stream<Item = AppResult<sse::Event>> {
    async_stream::stream! {
        let start_time = SystemTime::now();
        // Start run and assistant message
        let run_id = uuid::Uuid::new_v4().to_string();
        let thread_id = format!("t{run_id}");
        let msg_id = format!("chatcmpl-{run_id}");
        yield Ok(sse_event(json!({"type":"RUN_STARTED","threadId":thread_id,"runId":run_id}))?);
        yield Ok(sse_event(json!({"type":"TEXT_MESSAGE_START","messageId":msg_id,"role":"assistant"}))?);

        // Initialize the workflow
        let workflow = match SearchWorkflow::new(resp.model.clone(), state.bind_address.clone()).await {
            Ok(workflow) => workflow,
            Err(e) => {
                tracing::error!("Failed to initialize workflow: {:?}", e);
                yield Ok(send_error_event("Error connecting to search service")?);
                return;
            }
        };

        // Step 1: Check if tool calls are neededß
        let (response_txt, tool_calls) = match workflow.request_tool_calls(&resp.messages).await {
            Ok(result) => result,
            Err(e) => {
                tracing::error!("Tool call execution failed: {:?}", e);
                yield Ok(send_error_event(&format!("Error searching for datasets: {e}"))?);
                return;
            }
        };

        // Stream tool call request, and execute each tool call if any
        let mut search_results = McpSearchResult {
            total_found: 0,
            hits: vec![],
        };
        if let Some(ref tc) = tool_calls {
            for (i, call) in tc.iter().enumerate() {
                let tc_id = format!("c{}", i + 1);
                yield Ok(sse_event(json!({"type":"TOOL_CALL_START","toolCallId":tc_id,"toolCallName":call.function.name,"parentMessageId":msg_id}))?);
                yield Ok(sse_event(json!({"type":"TOOL_CALL_ARGS","toolCallId":tc_id,"delta":call.function.arguments}))?);

                // Step 2: Execute tool calls and get aggregate search results
                let tool_results = workflow.execute_tool_call(call).await?;

                // Determine tool results message content
                let tool_msg_content = if let Some(structured) = &tool_results.structured_content {
                    // Priority 1: Try to parse as `McpSearchResult` (structured search results)
                    match serde_json::from_value::<McpSearchResult>(structured.clone()) {
                        Ok(new_search_results) => {
                            let serialized = serde_json::to_string(&new_search_results).unwrap_or_default();
                            // Accumulate results from multiple tool calls
                            search_results.hits.extend(new_search_results.hits);
                            search_results.total_found += new_search_results.total_found;
                            serialized
                        }
                        Err(_) => {
                            // Priority 2: Fallback to any structured JSON content
                            serde_json::to_string(structured).unwrap_or_default()
                        }
                    }
                } else {
                    // Priority 3: Fallback to plain text content
                    tracing::debug!(
                        "Tool {} returned plain text content: {:?}",
                        call.function.name,
                        tool_results.content
                    );
                    tool_results
                        .content
                        .iter()
                        .filter_map(|annotated| match &annotated.raw {
                            rmcp::model::RawContent::Text(text_content) => {
                                Some(text_content.text.as_str())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                };
                yield Ok(sse_event(json!({
                    "type":"TOOL_CALL_RESULT",
                    "messageId": msg_id,
                    "role": "tool",
                    "content": tool_msg_content,
                    "toolCallId": tc_id.clone(),
                }))?);
                yield Ok(sse_event(json!({"type":"TOOL_CALL_END","toolCallId":tc_id}))?);
            }
        }
        // LEGACY: Stream tool call requested event
        // yield Ok(workflow.create_sse_event("tool_call_requested", &tool_calls)?);


        // If no tool calls were made, handle as direct response
        if tool_calls.is_none() || search_results.total_found == 0 {
            let execution_time = start_time.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
            let response = if tool_calls.is_none() {
                // Direct response without tool calls
                let response = ChatSearchResponse {
                    hits: vec![],
                    summary: response_txt,
                };
                yield Ok(sse_event(json!({"type":"TEXT_MESSAGE_CONTENT","messageId":msg_id,"delta":response.summary.clone()}))?);
                response
            } else {
                // No results found
                let no_res_msg = "No datasets found for your query.";
                let response = ChatSearchResponse {
                    hits: vec![],
                    summary: no_res_msg.to_string(),
                };
                yield Ok(sse_event(json!({"type":"TEXT_MESSAGE_CONTENT","messageId":msg_id,"delta":no_res_msg}))?);
                yield Ok(send_error_event(no_res_msg)?);
                response
            };
            yield Ok(sse_event(json!({"type":"TEXT_MESSAGE_END","messageId":msg_id}))?);
            yield Ok(sse_event(json!({"type":"RUN_FINISHED","threadId":thread_id,"runId":run_id}))?);
            workflow.log_response(
                resp.stream,
                resp.messages.clone(),
                response,
                execution_time,
            );
            return;
        }

        // LEGACY: Stream the tool results (without scores or summary)
        // yield Ok(workflow.create_sse_event("tool_call_result", &search_results)?);

        // Step 3: Generate and stream summary and scores using LLM
        let final_response = match workflow.generate_summary_and_scores(&resp.messages, search_results).await {
            Ok(response) => response,
            Err(e) => {
                // Fallback if LLM processing fails
                yield Ok(send_error_event(&format!("{e}"))?);
                return;
            }
        };
        // LEGACY: Stream final response
        // yield Ok(workflow.create_sse_event("search_response", &final_response)?);

        // Stream the final assistant message content with summary
        // yield Ok(sse_event(json!({"type":"TEXT_MESSAGE_CONTENT","messageId":msg_id,"delta":final_response.summary.clone()}))?);
        let final_tc_id = "summarize_results";
        yield Ok(sse_event(json!({"type":"TOOL_CALL_START","toolCallId":final_tc_id,"toolCallName":"summarize_results","parentMessageId":msg_id}))?);
        // yield Ok(sse_event(json!({"type":"TOOL_CALL_ARGS","toolCallId":final_tc_id,"delta":call.function.arguments}))?);
        yield Ok(sse_event(json!({
            "type":"TOOL_CALL_RESULT",
            "messageId": msg_id,
            "role": "tool",
            "content": serde_json::to_string(&final_response).unwrap_or_default(),
            "toolCallId": final_tc_id,
        }))?);
        yield Ok(sse_event(json!({"type":"TOOL_CALL_END","toolCallId":final_tc_id}))?);

        yield Ok(sse_event(json!({"type":"TEXT_MESSAGE_END","messageId":msg_id}))?);
        // Finish the run
        yield Ok(sse_event(json!({"type":"RUN_FINISHED","threadId":thread_id,"runId":run_id}))?);

        // LEGACY: stop streaming chunk
        // yield Ok(workflow.create_stream_chunk(None, Some("stop".to_string()))?);

        // Calculate execution time and log the search resp
        let execution_time = start_time.elapsed().map(|d| d.as_millis() as u64).unwrap_or(0);
        workflow.log_response(
            resp.stream,
            resp.messages.clone(),
            final_response,
            execution_time,
        );
    }
}

/// Search handler for non-streaming responses
async fn regular_chat_handler(
    headers: HeaderMap,
    resp: ChatInput,
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
    let (_response_txt, tool_calls) = match workflow.request_tool_calls(&resp.messages).await {
        Ok(result) => result,
        Err(e) => {
            tracing::error!("Tool call execution failed: {:?}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": &format!("Error searching for datasets: {e}")})),
            );
        }
    };
    // let search_results = match workflow.execute_tool_calls(&tool_calls).await {
    //     Ok(results) => results,
    //     Err(e) => {
    //         tracing::error!("Tool call execution failed: {:?}", e);
    //         return (
    //             StatusCode::INTERNAL_SERVER_ERROR,
    //             Json(serde_json::json!({"error": &format!("Error executing tool calls: {e}")})),
    //         );
    //     }
    // };
    let mut search_results = McpSearchResult {
        total_found: 0,
        hits: vec![],
    };
    if let Some(ref tc) = tool_calls {
        for call in tc.iter() {
            // Step 2: Execute tool calls and get aggregate search results
            let tool_results = workflow.execute_tool_call(call).await.unwrap();

            // Determine tool results message content
            if let Some(structured) = &tool_results.structured_content {
                // Priority 1: Try to parse as `McpSearchResult` (structured search results)
                match serde_json::from_value::<McpSearchResult>(structured.clone()) {
                    Ok(new_search_results) => {
                        let serialized = serde_json::to_string(&new_search_results).unwrap_or_default();
                        // Accumulate results from multiple tool calls
                        search_results.hits.extend(new_search_results.hits);
                        search_results.total_found += new_search_results.total_found;
                        serialized
                    }
                    Err(_) => {
                        // Priority 2: Fallback to any structured JSON content
                        serde_json::to_string(structured).unwrap_or_default()
                    }
                }
            } else {
                // Priority 3: Fallback to plain text content
                tracing::debug!(
                    "Tool {} returned plain text content: {:?}",
                    call.function.name,
                    tool_results.content
                );
                tool_results
                    .content
                    .iter()
                    .filter_map(|annotated| match &annotated.raw {
                        rmcp::model::RawContent::Text(text_content) => {
                            Some(text_content.text.as_str())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            };
        }
    }

    // If no datasets were found, return early
    if search_results.total_found == 0 || search_results.hits.is_empty() {
        let execution_time = start_time
            .elapsed()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let final_response = ChatSearchResponse {
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
            ChatSearchResponse {
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

/// Create SSE event for streaming responses
pub fn sse_event(data: impl Serialize) -> AppResult<sse::Event> {
    Ok(sse::Event::default().data(serde_json::to_string(&data)?))
}
