use axum::{
    extract::Json,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use llm::{
    // FunctionCall, ToolCall,
    builder::{LLMBackend, LLMBuilder},
    chat::{ChatMessage, StructuredOutputFormat},
};

use rmcp::{
    ServiceExt,
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    transport::StreamableHttpClientTransport,
};

use crate::mcp::{DatasetSummary, SearchResult};

pub const ADDRESS: &str = "0.0.0.0:8000";
const SYSTEM_PROMPT: &str = "Given the user question and datasets retrieved from the search API, summarize the findings in 1 sentence, extract which datasets might be the most interesting to answer the user question, and give them a relevance score between 0 and 1";
// const DEFAULT_MODEL: &str = "mistral-small-latest";
// const DEFAULT_MODEL: &str = "mistral-medium-latest";

/// Represents a message in the chat
#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct Message {
    #[schema(example = "user")]
    pub role: String,
    #[schema(
        example = "I am looking for data about glucose level evolution in liver on people with type 1 diabetes in Europe between 1980 and 2020"
    )]
    pub content: String,
}

#[derive(Debug, Deserialize, Serialize, ToSchema)]
pub struct SearchResponse {
    pub messages: Vec<Message>,
    #[schema(example = "mistral-small-latest")]
    pub model: String,
}

/// Enhanced dataset with LLM score and additional metadata
#[derive(Debug, Deserialize, Serialize)]
struct EnhancedDataset {
    doi: String,
    score: f64,
    title: String,
    description: String,
    publication_date: String,
    keywords: Option<Vec<String>>,
    creators: Option<Vec<String>>,
    zenodo_url: String,
}

/// LLM response format
#[derive(Debug, Deserialize, Serialize)]
struct LLMResponse {
    summary: String,
    datasets: Vec<LLMDataset>,
}

#[derive(Debug, Deserialize, Serialize)]
struct LLMDataset {
    doi: String,
    score: f64,
}

/// Search data relevant to a user question in a conversation
#[utoipa::path(
    post,
    path = "/search",
    request_body(content = SearchResponse, description = "List of messages in the chat"),
)]
pub async fn search_handler(
    headers: HeaderMap,
    Json(mut resp): Json<SearchResponse>,
) -> impl IntoResponse {
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
    let api_key = match std::env::var("MISTRAL_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            // eprintln!("MISTRAL_API_KEY environment variable not set");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "MISTRAL_API_KEY environment variable not set"})),
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
        .backend(LLMBackend::Mistral)
        .api_key(api_key)
        .model(&resp.model)
        .max_tokens(512)
        .temperature(0.1)
        .stream(false)
        .system(SYSTEM_PROMPT)
        .schema(schema);

    // // Convert MCP tools to LLM functions and add them to the llm builder
    // let tools = client.list_tools(Default::default()).await.unwrap();
    // tracing::info!("Available tools: {tools:#?}");
    // for tool in &tools.tools {
    //     let schema_value = serde_json::Value::Object(tool.input_schema.as_ref().clone());
    //     let function = FunctionBuilder::new(tool.name.to_string())
    //         .description(tool.description.as_deref().unwrap_or(""))
    //         .json_schema(schema_value);
    //     llm_builder = llm_builder.function(function);
    // }
    let llm = llm_builder.build().expect("Failed to build LLM client");

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

    // Parse the structured JSON response from MCP search data tool and format it for the LLM
    let formatted_context = match serde_json::from_str::<SearchResult>(&tool_result_text) {
        Ok(search_result) => {
            if search_result.total_found == 0 {
                "No datasets found for your query.".to_string()
            } else {
                let mut formatted = format!(
                    "Found {} datasets relevant to the query '{}':\n\n",
                    search_result.total_found, search_result.query
                );
                for (i, dataset) in search_result.datasets.iter().enumerate() {
                    formatted.push_str(&format!("{}. **{}**\n", i + 1, dataset.title));
                    if let Some(doi) = &dataset.doi {
                        formatted.push_str(&format!("   DOI: https://doi.org/{}\n", doi));
                    }
                    formatted.push_str(&format!("   Zenodo: {}\n", dataset.zenodo_url));
                    formatted.push_str(&format!("   Published: {}\n", dataset.publication_date));
                    if let Some(creators) = &dataset.creators {
                        if !creators.is_empty() {
                            formatted.push_str(&format!("   Authors: {}\n", creators.join(", ")));
                        }
                    }
                    if let Some(keywords) = &dataset.keywords {
                        if !keywords.is_empty() {
                            formatted.push_str(&format!("   Keywords: {}\n", keywords.join(", ")));
                        }
                    }
                    formatted.push_str(&format!("   Description: {}\n\n", dataset.description));
                }
                formatted
            }
        }
        Err(e) => {
            eprintln!("Failed to parse search result JSON: {}", e);
            // Fallback to raw text if JSON parsing fails
            tool_result_text.clone()
        }
    };

    // Convert messages to LLM ChatMessage format, replacing last message content with formatted context
    let chat_messages: Vec<ChatMessage> = resp
        .messages
        .iter()
        .enumerate()
        .map(|(i, msg)| {
            let content = if i == resp.messages.len() - 1 {
                &format!("{}\n\n{}", &msg.content, formatted_context)
            } else {
                &msg.content
            };
            match msg.role.as_str() {
                "user" => ChatMessage::user().content(content).build(),
                "assistant" => ChatMessage::assistant().content(content).build(),
                _ => ChatMessage::user().content(content).build(), // Default to user if unknown role
            }
        })
        .collect();

    // TODO: call with structured output to retrieve the list of most relevant according to the LLM

    // Send chat request using additional infos retrieved by the tool call
    match llm.chat(&chat_messages).await {
        Ok(response) => {
            // Parse the JSON response from the LLM (structured output)
            let response_text = response.text().unwrap_or_default();
            match serde_json::from_str::<LLMResponse>(&response_text) {
                Ok(llm_response) => {
                    // Parse the original search results to get full dataset metadata
                    let search_result = serde_json::from_str::<SearchResult>(&tool_result_text)
                        .unwrap_or_else(|_| SearchResult {
                            total_found: 0,
                            query: String::new(),
                            datasets: vec![],
                        });

                    // Create a lookup map for full dataset info by DOI
                    let dataset_lookup: std::collections::HashMap<String, &DatasetSummary> = search_result
                        .datasets
                        .iter()
                        .filter_map(|ds| ds.doi.as_ref().map(|doi| (format!("https://doi.org/{}", doi), ds)))
                        .collect();

                    // Enrich LLM datasets with full metadata
                    let enhanced_datasets: Vec<EnhancedDataset> = llm_response
                        .datasets
                        .into_iter()
                        .filter_map(|llm_dataset| {
                            dataset_lookup.get(&llm_dataset.doi).map(|full_dataset| EnhancedDataset {
                                doi: llm_dataset.doi,
                                score: llm_dataset.score,
                                title: full_dataset.title.clone(),
                                description: full_dataset.description.clone(),
                                publication_date: full_dataset.publication_date.clone(),
                                keywords: full_dataset.keywords.clone(),
                                creators: full_dataset.creators.clone(),
                                zenodo_url: full_dataset.zenodo_url.clone(),
                            })
                        })
                        .collect();

                    let enhanced_response = serde_json::json!({
                        "summary": llm_response.summary,
                        "datasets": enhanced_datasets
                    });

                    return (StatusCode::OK, Json(enhanced_response));
                }
                Err(e) => {
                    eprintln!("Failed to parse LLM response as JSON: {}", e);
                    // Fallback: add to messages and return the original response format
                    resp.messages.push(Message {
                        role: "assistant".into(),
                        content: response_text,
                    });
                }
            }
        }
        Err(e) => {
            eprintln!("Chat error: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Chat error: {}", e)})),
            );
        }
    }


    // // Send chat request with tools is crashing after SEND
    // // Chat error: Response Format Error: Failed to decode Mistral API response: missing field `type` at line 1 column 376. Raw response: {"id":"294dbcf061ef4b1cb5f82eb09bae6bea","created":1753771664,"model":"mistral-medium-2505","usage":{"prompt_tokens":647,"total_tokens":664,"completion_tokens":17},"object":"chat.completion","choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[{"id":"jxsN10Cs0","function":{"name":"sum","arguments":"{\"a\": 5, \"b\": 76}"},"index":0}],"content":""}}]}
    // // https://github.com/graniet/llm/blob/main/examples/openai_example.rs
    // // Use tool calling with tools extracted from MCP server
    // // https://github.com/graniet/llm/blob/main/examples/google_tool_calling_example.rs
    // println!("SEND");
    // match llm.chat_with_tools(&chat_messages, llm.tools()).await {
    //     Ok(response) => {
    //         println!("RESP");
    //         if let Some(tool_calls) = response.tool_calls() {
    //             println!("Tool calls requested:");
    //             for call in &tool_calls {
    //                 println!("Function: {}", call.function.name);
    //                 println!("Arguments: {}", call.function.arguments);
    //                 // let result = process_tool_call(call)?;
    //                 // println!("Result: {}", serde_json::to_string_pretty(&result)?);
    //             }
    //             all_msgs.push(Message {
    //                 role: "assistant".into(),
    //                 content: response.text().unwrap_or_default(),
    //             });
    //             // let mut conversation = messages;
    //             // conversation.push(
    //             //     ChatMessage::assistant()
    //             //         .tool_use(tool_calls.clone())
    //             //         .build(),
    //             // );
    //             // let tool_results: Vec<ToolCall> = tool_calls
    //             //     .iter()
    //             //     .map(|call| {
    //             //         let result = process_tool_call(call).unwrap();
    //             //         ToolCall {
    //             //             id: call.id.clone(),
    //             //             call_type: "function".to_string(),
    //             //             function: FunctionCall {
    //             //                 name: call.function.name.clone(),
    //             //                 arguments: serde_json::to_string(&result).unwrap(),
    //             //             },
    //             //         }
    //             //     })
    //             //     .collect();
    //             // conversation.push(ChatMessage::user().tool_result(tool_results).build());
    //             // let final_response = llm.chat_with_tools(&conversation, llm.tools()).await?;
    //             // println!("\nFinal response: {}", final_response);
    //         } else {
    //             println!("Direct response: {}", response);
    //             all_msgs.push(Message {
    //                 role: "assistant".into(),
    //                 content: response.text().unwrap_or_default(),
    //             });
    //         }
    //     },
    //     Err(e) => eprintln!("Chat error: {}", e),
    // }
    (StatusCode::OK, Json(serde_json::to_value(resp).unwrap()))
}
