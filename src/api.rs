use axum::{
    extract::Json,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use reqwest;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use rmcp::{
    model::{CallToolRequestParam, ClientCapabilities, ClientInfo, Implementation},
    transport::StreamableHttpClientTransport,
    ServiceExt,
};

use crate::mcp::{DatasetSummary, SearchResult};

pub const ADDRESS: &str = "0.0.0.0:8000";
const SYSTEM_PROMPT: &str = "Given the user question and datasets retrieved from the search API, summarize the findings in 1 sentence, extract which datasets might be the most interesting to answer the user question, and give them a relevance score between 0 and 1";

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

/// LLM response format for structured output
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

    // Configure local LLM settings
    let local_base_url = std::env::var("LOCAL_LLM_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:1234/v1".to_string());
    let local_model = std::env::var("LOCAL_LLM_MODEL")
        .unwrap_or_else(|_| "mistralai/mistral-small-3.2".to_string());

    tracing::info!("Local LLM config - URL: {}, Model: {}", local_base_url, local_model);

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

    // Call MCP tool to search for datasets
    let last_message_content = resp
        .messages
        .last()
        .map(|msg| msg.content.as_str())
        .unwrap_or("");

    tracing::info!("Making MCP tool call with question: {}", last_message_content);

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

    tracing::info!("MCP tool result length: {} chars", tool_result_text.len());

    // Parse and format the search results for the LLM
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
            tool_result_text.clone()
        }
    };

    // Prepare request for local LLM
    let llm_request = serde_json::json!({
        "model": local_model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": format!("{}\n\nContext:\n{}", last_message_content, formatted_context)
            }
        ],
        "temperature": 0.1,
        "max_tokens": 512,
        "stream": false
    });

    tracing::info!("About to send request to LLM at {}/chat/completions", local_base_url);

    // Make direct HTTP request to local LLM (LM Studio)
    let http_client = reqwest::Client::new();
    let response = http_client
        .post(&format!("{}/chat/completions", local_base_url))
        .header("Content-Type", "application/json")
        .json(&llm_request)
        .send()
        .await;

    match response {
        Ok(response) => {
            let status = response.status();
            if status.is_success() {
                match response.json::<serde_json::Value>().await {
                    Ok(json_response) => {
                        tracing::info!("LLM response received successfully");

                        // Extract the response text from OpenAI format
                        let response_text = json_response
                            .get("choices")
                            .and_then(|choices| choices.get(0))
                            .and_then(|choice| choice.get("message"))
                            .and_then(|message| message.get("content"))
                            .and_then(|content| content.as_str())
                            .unwrap_or("No response content found");

                        // Try to parse LLM response as structured JSON, fallback to text summary
                        match serde_json::from_str::<LLMResponse>(&response_text) {
                            Ok(llm_response) => {
                                // Structured response: match LLM recommendations with full dataset metadata
                                let search_result = serde_json::from_str::<SearchResult>(&tool_result_text)
                                    .unwrap_or_else(|_| SearchResult {
                                        total_found: 0,
                                        query: String::new(),
                                        datasets: vec![],
                                    });

                                let dataset_lookup: std::collections::HashMap<String, &DatasetSummary> = search_result
                                    .datasets
                                    .iter()
                                    .filter_map(|ds| ds.doi.as_ref().map(|doi| (format!("https://doi.org/{}", doi), ds)))
                                    .collect();

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
                            Err(_) => {
                                // Text response: use LLM text as summary with all datasets
                                let search_result = serde_json::from_str::<SearchResult>(&tool_result_text)
                                    .unwrap_or_else(|_| SearchResult {
                                        total_found: 0,
                                        query: String::new(),
                                        datasets: vec![],
                                    });

                                let enhanced_datasets: Vec<EnhancedDataset> = search_result
                                    .datasets
                                    .into_iter()
                                    .map(|dataset| EnhancedDataset {
                                        doi: dataset.doi.map(|d| format!("https://doi.org/{}", d)).unwrap_or_default(),
                                        score: 0.5, // Default score since LLM didn't provide structured output
                                        title: dataset.title,
                                        description: dataset.description,
                                        publication_date: dataset.publication_date,
                                        keywords: dataset.keywords,
                                        creators: dataset.creators,
                                        zenodo_url: dataset.zenodo_url,
                                    })
                                    .collect();

                                let enhanced_response = serde_json::json!({
                                    "summary": response_text,
                                    "datasets": enhanced_datasets
                                });

                                return (StatusCode::OK, Json(enhanced_response));
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse LLM response JSON: {}", e);
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({"error": format!("Failed to parse LLM response: {}", e)})),
                        );
                    }
                }
            } else {
                let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                tracing::error!("LLM request failed with status {}: {}", status, error_text);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({"error": format!("LLM request failed: {}", error_text)})),
                );
            }
        }
        Err(e) => {
            tracing::error!("Failed to send request to LLM: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("Failed to connect to LLM: {}", e)})),
            );
        }
    }
}
