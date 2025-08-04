use axum::response::sse;
use axum::{
    extract::Json,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
};
use futures_util::stream::Stream;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::{SystemTime, UNIX_EPOCH};
use utoipa::ToSchema;

use llm::{
    // FunctionCall, ToolCall,
    builder::{FunctionBuilder, LLMBackend, LLMBuilder},
    chat::{ChatMessage, StructuredOutputFormat},
};

use rmcp::{
    ServiceExt,
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    transport::StreamableHttpClientTransport,
};

use crate::mcp::{SearchHit, SearchResult};

pub const ADDRESS: &str = "0.0.0.0:8000";
const SYSTEM_PROMPT: &str =
    "You are an assistant that help users find datasets and tools for scientific research.";
const SYSTEM_PROMPT_TOOLS: &str = "You are an assistant that help users find datasets and tools for scientific research. Define if you need to use one of the tool provided to get more context to answer the user request.";
const SYSTEM_PROMPT_RESOLUTION: &str = "Given the user question and datasets retrieved from the search API, summarize the findings in 1 sentence, extract which datasets might be the most interesting to answer the user question, and give them a relevance score between 0 and 1";
// const DEFAULT_MODEL: &str = "mistral-small-latest";
// const DEFAULT_MODEL: &str = "mistral-medium-latest";

const LLM_BACKEND: LLMBackend = LLMBackend::OpenAI;
const LLM_API_KEY: &str = "OPENAI_API_KEY";
// const LLM_BACKEND: LLMBackend = LLMBackend::Mistral;
// const LLM_API_KEY: &str = "MISTRAL_API_KEY";

/// Wrapper around llm::ChatMessage that implements ToSchema for API documentation
#[derive(Debug, Deserialize, Serialize, ToSchema, Clone)]
pub struct ApiChatMessage {
    #[schema(example = "user")]
    pub role: String,
    #[schema(
        example = "I am looking for data about glucose level evolution in liver on people with type 1 diabetes in Europe between 1980 and 2020"
    )]
    pub content: String,
}

impl ApiChatMessage {
    /// Convert to llm::ChatMessage for use with the LLM client
    pub fn to_chat_message(&self) -> ChatMessage {
        match self.role.as_str() {
            "user" => ChatMessage::user().content(&self.content).build(),
            "assistant" => ChatMessage::assistant().content(&self.content).build(),
            _ => ChatMessage::user().content(&self.content).build(), // Default to user if unknown role
        }
    }

    /// Create from role and content
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Self::new("user", content)
    }

    /// Create an assistant message
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new("assistant", content)
    }
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct SearchInput {
    pub messages: Vec<ApiChatMessage>,
    // #[schema(example = "mistral-small-latest")]
    #[schema(example = "gpt-4.1-nano")]
    pub model: String,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct SearchResponse {
    pub hits: Vec<SearchHit>,
    pub summary: String,
}

/// OpenAI-compatible streaming response chunk
#[derive(Debug, Serialize)]
struct StreamChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Serialize)]
struct StreamChoice {
    index: u32,
    delta: StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct StreamDelta {
    role: Option<String>,
    content: Option<String>,
    function_call: Option<serde_json::Value>,
}

/// Response for initial Zenodo data
#[derive(Debug, Serialize)]
struct ZenodoDataChunk {
    datasets: Vec<SearchHit>,
    step: String,
}

/// LLM response format
#[derive(Debug, Deserialize, Serialize, ToSchema)]
struct LLMStructuredOutput {
    summary: String,
    datasets: Vec<LLMDataset>,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
struct LLMDataset {
    doi: String,
    score: f64,
}

/// Search data relevant to a user question in a conversation
#[utoipa::path(
    post,
    path = "/search",
    request_body(content = SearchInput, description = "List of messages in the chat"),
    responses(
        (status = 200, description = "Search results", body = SearchResponse),
        (status = 401, description = "Unauthorized"),
        (status = 500, description = "Internal server error")
    ),
)]
pub async fn search_handler(headers: HeaderMap, Json(resp): Json<SearchInput>) -> Response {
    if resp.stream {
        streaming_search_handler(headers, resp)
            .await
            .into_response()
    } else {
        regular_search_handler(headers, resp).await.into_response()
    }
}

/// Streaming search handler for SSE responses
async fn streaming_search_handler(headers: HeaderMap, resp: SearchInput) -> impl IntoResponse {
    // Validate API key only if SEARCH_API_KEY is set
    if let Ok(secret_key) = std::env::var("SEARCH_API_KEY") {
        let auth_header = headers.get("authorization");
        if auth_header.is_none() || auth_header.unwrap().to_str().unwrap_or("") != secret_key {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "unauthorized"})),
            )
                .into_response();
        }
    }
    let stream = create_search_stream(resp);
    Sse::new(stream)
        // .keep_alive(
        //     sse::KeepAlive::new()
        //         .interval(std::time::Duration::from_secs(10)),
        //         // .text("keep-alive-text"),
        // )
        .into_response()
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
                "datasets": {
                    "type": "array",
                    "description": "List of most relevant datasets",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doi": {
                                "type": "string",
                                "description": "Digital Object Identifier of the dataset"
                            },
                            "score": {
                                "type": "number",
                                "description": "Relevance score of the dataset based on the search query (between 0 and 1)"
                            }
                        },
                        "additionalProperties": false,
                        "required": ["doi", "score"]
                    }
                }
            },
            "additionalProperties": false,
            "required": ["summary", "datasets"]
        },
        "strict": true
    }
"#;

/// Helper function to create error SSE events
fn send_error_event(error_message: &str) -> sse::Event {
    sse::Event::default().event("error")
        .data(
            serde_json::to_string(&serde_json::json!({
                "error": error_message.to_string(),
            }))
            .unwrap(),
        )
    // sse::Event::default().data(
    //     serde_json::to_string(&StreamChunk {
    //         id: msg_id.to_string(),
    //         object: "chat.completion.chunk".to_string(),
    //         created,
    //         model: model.to_string(),
    //         choices: vec![StreamChoice {
    //             index: 0,
    //             delta: StreamDelta {
    //                 role: Some("assistant".to_string()),
    //                 content: Some(error_message.to_string()),
    //                 function_call: None,
    //             },
    //             finish_reason: Some(finish_reason.to_string()),
    //         }],
    //     })
    //     .unwrap(),
    // )
}

fn create_search_stream(resp: SearchInput) -> impl Stream<Item = Result<sse::Event, Infallible>> {
    async_stream::stream! {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let msg_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

        // Connect to MCP server
        let transport = StreamableHttpClientTransport::from_uri(format!("http://{ADDRESS}/mcp"));
        let client_info = ClientInfo {
            protocol_version: Default::default(),
            capabilities: ClientCapabilities::default(),
            client_info: Implementation {
                name: "MCP HTTP client".to_string(),
                version: "0.0.1".to_string(),
            },
        };
        let client = match client_info.serve(transport).await {
            Ok(client) => client,
            Err(e) => {
                tracing::error!("client error: {:?}", e);
                yield Ok(send_error_event("Error connecting to search service"));
                return;
            }
        };
        let api_key = match std::env::var(LLM_API_KEY) {
            Ok(key) => key,
            Err(_) => {
                tracing::error!("{LLM_API_KEY} environment variable not set");
                return;
            }
        };

        // Convert messages to LLM ChatMessage format
        let mut chat_messages: Vec<ChatMessage> = resp
            .messages
            .iter()
            .map(|msg| {
                match msg.role.as_str() {
                    "user" => ChatMessage::user().content(&msg.content).build(),
                    "assistant" => ChatMessage::assistant().content(&msg.content).build(),
                    _ => ChatMessage::user().content(&msg.content).build(), // Default to user if unknown role
                }
            })
            .collect();

        // Configure Mistral LLM client with dynamic tools from MCP
        let mut llm_builder = LLMBuilder::new()
            .backend(LLM_BACKEND)
            .api_key(&api_key)
            .model(&resp.model)
            .max_tokens(1024)
            .temperature(0.1)
            .stream(false)
            .system(SYSTEM_PROMPT_TOOLS);

        // Convert MCP tools to LLM functions and add them to the llm builder
        let tools = client.list_tools(Default::default()).await.unwrap();
        tracing::debug!("Available tools: {tools:#?}");
        for tool in &tools.tools {
            let schema_value = serde_json::Value::Object(tool.input_schema.as_ref().clone());
            let function = FunctionBuilder::new(tool.name.to_string())
                .description(tool.description.as_deref().unwrap_or(""))
                .json_schema(schema_value);
            llm_builder = llm_builder.function(function);
        }
        let llm = llm_builder.build().expect("Failed to build LLM client");

        // Query the LLM to check if tool call necessary (with tools extracted from MCP server)
        // https://github.com/graniet/llm/blob/main/examples/openai_example.rs
        // https://github.com/graniet/llm/blob/main/examples/google_tool_calling_example.rs
        let (response_text, tool_calls) = match llm.chat_with_tools(&chat_messages, llm.tools()).await {
            Ok(response) => {
                tracing::debug!("LLM response: {response:#?}");
                (response.text().unwrap_or_default().to_string(), response.tool_calls())
            },
            Err(e) => {
                tracing::error!("Chat error: {}", e);
                return;
            },
        };
        yield Ok(sse::Event::default()
            .event("tool_call_requested")
            .data(serde_json::to_string(&tool_calls).unwrap()));

        // Get last msg from the user, and init search results
        let last_message_content = resp
            .messages
            .last()
            .map(|msg| msg.content.as_str())
            .unwrap_or("");
        let mut search_results = SearchResult {
            total_found: 0,
            hits: vec![],
        };
        // Execute each tool call if any
        if let Some(tc) = tool_calls {
            // tracing::debug!("Tool calls requested: {tc:#?}");
            for call in &tc {
                tracing::debug!("Calling tool {}", call.function.name);
                let arguments = match serde_json::from_str::<serde_json::Value>(&call.function.arguments) {
                    Ok(value) => value.as_object().cloned(),
                    Err(_) => None,
                };
                // Call MCP tools
                let tool_results = match client
                    .call_tool(CallToolRequestParam {
                        name: call.function.name.clone().into(),
                        arguments,
                    })
                    .await
                {
                    Ok(results) => results,
                    Err(e) => {
                        tracing::error!("MCP tool call failed: {}", e);
                        yield Ok(send_error_event("Error searching for datasets"));
                        return;
                    }
                };

                // Extract and parse structured JSON content from the tool result
                let tool_result_text = tool_results
                    .content
                    .iter()
                    .filter_map(|annotated| match &annotated.raw {
                        rmcp::model::RawContent::Text(text_content) => Some(text_content.text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" ");

                // Parse the structured JSON response from MCP search data tool
                let new_search_result = match serde_json::from_str::<SearchResult>(&tool_result_text) {
                    Ok(result) => result,
                    Err(e) => {
                        tracing::error!("Failed to parse search result JSON: {e}");
                        yield Ok(send_error_event("Issue querying the search service."));
                        return;
                    }
                };
                // Accumulate results from multiple tool calls
                search_results.hits.extend(new_search_result.hits);
                search_results.total_found += new_search_result.total_found;

                // Stream the tool results (without scores or summary)
                yield Ok(sse::Event::default()
                    .event("tool_call_result")
                    .data(serde_json::to_string(&search_results).unwrap()));
            } // Tool call end

            // Format the context for the LLM based on tool call results
            let mut formatted_context = format!(
                "Found {} datasets relevant to the query '{}':\n\n",
                search_results.total_found, last_message_content
            );
            for (i, dataset) in search_results.hits.iter().enumerate() {
                formatted_context.push_str(&format!("{}. **{}**\n", i + 1, dataset.title));
                if let Some(doi) = &dataset.doi {
                    formatted_context.push_str(&format!("   DOI: https://doi.org/{doi}\n"));
                }
                formatted_context.push_str(&format!("   Zenodo: {}\n", dataset.zenodo_url));
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
            chat_messages.push(ChatMessage::assistant().content(&formatted_context).build());
        } else {
            // Handle case where no tool calls were made (regular chat response)
            tracing::debug!("Direct response! {}", response_text);
            yield Ok(sse::Event::default().event("error")
                .data(
                    serde_json::to_string(&serde_json::json!({
                        "id": msg_id.to_string(),
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": resp.model.clone(),
                        "choices": [{
                            "index": 0,
                            // "message": { when not streaming
                            "delta": {
                                "role": "assistant",
                                "content": response_text.to_string()
                            },
                            "finish_reason": "stop"
                        }]
                    })).unwrap()
                ));
                // Similar to llm crate message format:
                // .data(
                //     serde_json::to_string(&serde_json::json!({
                //         "choices": {"message": {
                //             "role": "assistant",
                //             "content": response_text.to_string(),
                //             "tool_calls": null,
                //         }},
                //     }))
                //     .unwrap(),
                // ));
            return;
        }

        // Check if no datasets were found and return early
        if search_results.total_found == 0 || search_results.hits.is_empty() {
            yield Ok(send_error_event("Nothing found for your query."));
            return;
        }

        // Create LLM client with structured output schema to answer user question
        let schema: StructuredOutputFormat = serde_json::from_str(SEARCH_OUTPUT_SCHEMA).unwrap();
        let llm_resolution = LLMBuilder::new()
            .backend(LLM_BACKEND)
            .api_key(&api_key)
            .model(&resp.model)
            .max_tokens(512)
            .temperature(0.1)
            .stream(false)
            .system(SYSTEM_PROMPT_RESOLUTION)
            .schema(schema)
            .build().expect("Failed to build LLM client");

        tracing::debug!("Calling the LLM");
        // Send chat request using additional infos retrieved by the tool call
        let response_text = match llm_resolution.chat(&chat_messages).await {
            Ok(response) => {
                // Extract the response text immediately to avoid Send issues
                response.text().unwrap_or_default().to_string()
            }
            Err(e) => {
                tracing::error!("Chat error: {e}");
                return;
            }
        };
        tracing::debug!("Done calling the LLM");

        // Parse the response after dropping the response object
        match serde_json::from_str::<LLMStructuredOutput>(&response_text) {
            Ok(llm_response) => {
                // Create a lookup map for LLM scores by DOI
                let score_lookup: std::collections::HashMap<String, f64> = llm_response
                    .datasets
                    .into_iter()
                    .map(|llm_dataset| (llm_dataset.doi, llm_dataset.score))
                    .collect();

                // Add scores to all datasets from search results
                for hit in &mut search_results.hits {
                    if let Some(doi) = &hit.doi {
                        let doi_url = format!("https://doi.org/{doi}");
                        hit.score =
                            Some(score_lookup.get(&doi_url).copied().unwrap_or(0.0));
                    } else {
                        hit.score = Some(0.0);
                    }
                }

                // Sort hits by score in descending order (highest score first)
                search_results.hits.sort_by(|a, b| {
                    let score_a = a.score.unwrap_or(0.0);
                    let score_b = b.score.unwrap_or(0.0);
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let res = SearchResponse {
                    hits: search_results.hits,
                    summary: llm_response.summary,
                };
                // Stream response
                yield Ok(sse::Event::default()
                    .event("search_response")
                    .data(serde_json::to_string(&res).unwrap()));
                // Final stop chunk
                yield Ok(sse::Event::default()
                    .data(serde_json::to_string(&StreamChunk {
                        id: msg_id.to_string(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: resp.model.clone(),
                        choices: vec![StreamChoice {
                            index: 0,
                            delta: StreamDelta {
                                role: None,
                                content: None,
                                function_call: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                    }).unwrap()));
                return;
            }
            Err(e) => {
                tracing::error!("Failed to parse LLM response as JSON: {}", e);
                // Fallback: add to messages and return the original response format
                // resp.messages.push(Message {
                //     role: "assistant".into(),
                //     content: response_text,
                // });
            }
        }
        tracing::debug!("Done calling the LLM");
    }
}

async fn regular_search_handler(headers: HeaderMap, mut resp: SearchInput) -> impl IntoResponse {
    // TODO: get the response by just calling the streaming function
    // (can't find how to properly get the event data)
    // // Create the stream using the same logic as streaming handler
    // let stream = create_search_stream(resp);
    // // Collect events from the stream and look for search_response
    // let mut stream = Box::pin(stream);
    // let mut search_response_data: Option<String> = None;
    // while let Some(event_result) = stream.next().await {
    //     let event = event_result.unwrap(); // Stream always returns Ok
    //     tracing::info!("Received event: {:?}", event);
    //     // We need to manually reconstruct the event to check its content
    //     // The event is built with .event("search_response") and .data(json_string)
    //     // Since we can't directly inspect the event type, we'll use debug output
    //     let event_debug = format!("{event:?}");
    //     // Check if this looks like a search_response event
    //     if event_debug.contains("search_response") {
    //         tracing::info!("Found search_response event: {}", event_debug);
    //         // Extract data from debug string - look for the JSON data field
    //         if let Some(data_start) = event_debug.find("data: Some(\"") {
    //             let data_content = &event_debug[data_start + 12..];
    //             if let Some(data_end) = data_content.find("\")") {
    //                 let escaped_json = &data_content[..data_end];
    //                 // Properly unescape the JSON string
    //                 let unescaped_json = escaped_json
    //                     .replace("\\\"", "\"")
    //                     .replace("\\\\", "\\")
    //                     .replace("\\n", "\n")
    //                     .replace("\\r", "\r")
    //                     .replace("\\t", "\t");
    //                 // Try to parse as SearchResponse
    //                 match serde_json::from_str::<SearchResponse>(&unescaped_json) {
    //                     Ok(_) => {
    //                         search_response_data = Some(unescaped_json);
    //                         break;
    //                     }
    //                     Err(e) => {
    //                         tracing::warn!("Failed to parse SearchResponse: {}", e);
    //                         tracing::warn!("Unescaped JSON: {}", unescaped_json);
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    // // If we found search response data, parse and return it
    // if let Some(data) = search_response_data {
    //     if let Ok(search_response) = serde_json::from_str::<SearchResponse>(&data) {
    //         return (StatusCode::OK, Json(serde_json::json!(search_response)));
    //     }
    // }

    // Validate API key only if SEARCH_API_KEY is set
    if let Ok(secret_key) = std::env::var("SEARCH_API_KEY") {
        let auth_header = headers.get("authorization");
        if auth_header.is_none() || auth_header.unwrap().to_str().unwrap_or("") != secret_key {
            return (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": "unauthorized"})),
            );
        }
    }
    let api_key = match std::env::var(LLM_API_KEY) {
        Ok(key) => key,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    serde_json::json!({"error": format!("{LLM_API_KEY} environment variable not set")}),
                ),
            );
        }
    };
    // let llm_model = std::env::var("MISTRAL_MODEL").unwrap_or(DEFAULT_MODEL.into());

    // Connect to MCP server
    let transport = StreamableHttpClientTransport::from_uri(format!("http://{ADDRESS}/mcp"));
    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "test HTTP client".to_string(),
            version: "0.0.1".to_string(),
        },
    };
    let client = client_info
        .serve(transport)
        .await
        .inspect_err(|e| {
            tracing::error!("client error: {:?}", e);
        })
        .unwrap();
    // Check server info
    // let server_info = client.peer_info();
    // tracing::info!("Connected to server: {server_info:#?}");

    // Convert input messages to llm ChatMessage format, replacing last message content with tool result
    // let mut search_resp = SearchResponse {
    //     model: llm_model,
    //     messages: messages
    // };

    // Define a simple JSON schema for structured output
    let schema = r#"
        {
            "name": "search_results",
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of the findings in 1 sentence"
                    },
                    "datasets": {
                        "type": "array",
                        "description": "List of most relevant datasets",
                        "items": {
                            "type": "object",
                            "properties": {
                                "doi": {
                                    "type": "string",
                                    "description": "Digital Object Identifier of the dataset"
                                },
                                "score": {
                                    "type": "number",
                                    "description": "Relevance score of the dataset based on the search query (between 0 and 1)"
                                }
                            },
                            "additionalProperties": false,
                            "required": ["doi", "score"]
                        }
                    }
                },
                "additionalProperties": false,
                "required": ["summary", "datasets"]
            },
            "strict": true
        }
    "#;
    let schema: StructuredOutputFormat = serde_json::from_str(schema).unwrap();

    // Configure Mistral LLM client with dynamic tools from MCP
    let llm_builder = LLMBuilder::new()
        .backend(LLM_BACKEND)
        .api_key(api_key)
        .model(&resp.model)
        .max_tokens(512)
        .temperature(0.1)
        .stream(false)
        .system(SYSTEM_PROMPT)
        .schema(schema);

    let llm = llm_builder.build().expect("Failed to build LLM client");

    tracing::info!("Calling Zenodo API");
    // Call a MCP tool
    let last_message_content = resp
        .messages
        .last()
        .map(|msg| msg.content.as_str())
        .unwrap_or("");
    let tool_results = client
        .call_tool(CallToolRequestParam {
            name: "search_data".into(),
            arguments: serde_json::json!({"question": last_message_content})
                .as_object()
                .cloned(),
        })
        .await
        .expect("MCP tool call failed");
    tracing::info!("Done calling Zenodo API");
    // Extract and parse structured JSON content from the tool result
    let tool_result_text = tool_results
        .content
        .iter()
        .filter_map(|annotated| match &annotated.raw {
            rmcp::model::RawContent::Text(text_content) => Some(text_content.text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");

    // Parse the structured JSON response from MCP search data tool
    let mut search_result = match serde_json::from_str::<SearchResult>(&tool_result_text) {
        Ok(result) => result,
        Err(e) => {
            tracing::error!("Failed to parse search result JSON: {}", e);
            // Return early with no datasets found message
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "summary": "No datasets found for your query.",
                    "datasets": []
                })),
            );
        }
    };
    // Check if no datasets were found and return early
    if search_result.total_found == 0 || search_result.hits.is_empty() {
        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "summary": "No datasets found for your query.",
                "datasets": []
            })),
        );
    }

    // Format the context for the LLM
    let mut formatted_context = format!(
        "Found {} datasets relevant to the query '{}':\n\n",
        search_result.total_found, last_message_content
    );
    for (i, dataset) in search_result.hits.iter().enumerate() {
        formatted_context.push_str(&format!("{}. **{}**\n", i + 1, dataset.title));
        if let Some(doi) = &dataset.doi {
            formatted_context.push_str(&format!("   DOI: https://doi.org/{doi}\n"));
        }
        formatted_context.push_str(&format!("   Zenodo: {}\n", dataset.zenodo_url));
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

    // Convert messages to LLM ChatMessage format, replacing last message content with formatted context
    let chat_messages: Vec<ChatMessage> = resp
        .messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            let content = if i == resp.messages.len() - 1 {
                format!("{}\n\n{}", &msg.content, formatted_context)
            } else {
                msg.content.clone()
            };
            // Use the to_chat_message method but with modified content
            match msg.role.as_str() {
                "user" => ChatMessage::user().content(&content).build(),
                "assistant" => ChatMessage::assistant().content(&content).build(),
                _ => ChatMessage::user().content(&content).build(), // Default to user if unknown role
            }
        })
        .collect();

    // TODO: call with structured output to retrieve the list of most relevant according to the LLM

    tracing::info!("Calling the LLM");
    // Send chat request using additional infos retrieved by the tool call
    match llm.chat(&chat_messages).await {
        Ok(response) => {
            // Parse the JSON response from the LLM (structured output)
            let response_text = response.text().unwrap_or_default();
            match serde_json::from_str::<LLMStructuredOutput>(&response_text) {
                Ok(llm_response) => {
                    // Create a lookup map for LLM scores by DOI
                    let score_lookup: std::collections::HashMap<String, f64> = llm_response
                        .datasets
                        .into_iter()
                        .map(|llm_dataset| (llm_dataset.doi, llm_dataset.score))
                        .collect();

                    // Add scores to all datasets from search results
                    for dataset in &mut search_result.hits {
                        if let Some(doi) = &dataset.doi {
                            let doi_url = format!("https://doi.org/{doi}");
                            dataset.score =
                                Some(score_lookup.get(&doi_url).copied().unwrap_or(0.0));
                        } else {
                            dataset.score = Some(0.0);
                        }
                    }

                    // Sort datasets by score in descending order (highest score first)
                    search_result.hits.sort_by(|a, b| {
                        let score_a = a.score.unwrap_or(0.0);
                        let score_b = b.score.unwrap_or(0.0);
                        score_b
                            .partial_cmp(&score_a)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    let resp = SearchResponse {
                        hits: search_result.hits,
                        summary: llm_response.summary,
                    };
                    return (StatusCode::OK, Json(serde_json::json!(resp)));
                }
                Err(e) => {
                    tracing::error!("Failed to parse LLM response as JSON: {}", e);
                    // Fallback: add to messages and return the original response format
                    resp.messages.push(ApiChatMessage::assistant(response_text));
                }
            }
        }
        Err(e) => {
            tracing::error!("Chat error: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Chat error: {}", e)})),
            );
        }
    }
    tracing::info!("Done calling the LLM");

    (StatusCode::OK, Json(serde_json::to_value(resp).unwrap()))
}
