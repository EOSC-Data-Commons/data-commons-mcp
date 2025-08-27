use std::sync::Arc;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use opensearch::{OpenSearch, http::transport::Transport};
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::*,
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Mutex;
use utoipa::ToSchema;

use crate::error::AppResult;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct UserQuestion {
    pub question: String,
    // pub topics: vec<String>, // potential topics and classes relevant to the query
    // pub time: String, // time range relevant to the query
}

/// Structured response for search results
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub total_found: u64,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct SearchHit {
    #[schema(example = "9x6qrJgBTkyZK1Kx4HAB")]
    pub id: String,
    #[schema(example = "Item Title")]
    pub title: String,
    #[schema(example = "Item Description")]
    pub description: String,
    #[schema(example = "2025-10-08")]
    pub publication_date: String,
    // #[schema(example = "['Information Retrieval, Data Science']")]
    pub keywords: Option<Vec<String>>,
    // #[schema(example = "['Emonet, Vincent']")]
    pub creators: Option<Vec<String>>,
    #[schema(example = "https://zenodo.org/record/5173026")]
    pub url: String,
    #[schema(example = "dataset")]
    pub resource_type: String,
    #[schema(example = "0.5")]
    pub score: Option<f64>,
    // #[schema(example = "10.48550/arXiv.2410.06062")]
    // pub doi: Option<String>,
}

const OPENSEARCH_INDEX_DATASETS: &str = "test_datacite";

#[derive(Clone)]
pub struct DataCommonsTools {
    tool_router: ToolRouter<DataCommonsTools>,
    opensearch_client: OpenSearch,
    embedding_model: Arc<Mutex<TextEmbedding>>,
}

#[tool_router]
impl DataCommonsTools {
    pub fn new(opensearch_url: &str) -> AppResult<Self> {
        Ok(Self {
            tool_router: Self::tool_router(),
            opensearch_client: OpenSearch::new(Transport::single_node(opensearch_url)?),
            embedding_model: Arc::new(Mutex::new(
                TextEmbedding::try_new(InitOptions::new(EmbeddingModel::BGESmallENV15)).unwrap(),
            )),
        })
    }

    /// Generate embedding for a given text using the shared fastembed model
    async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>, ErrorData> {
        // Lock the model for the duration of embedding generation
        let mut model = self.embedding_model.lock().await;
        let embeddings = model
            .embed(vec![format!("passage: {}", text)], None)
            // TODO: use rmcp::ErrorData
            .map_err(|e| {
                ErrorData::new(
                    ErrorCode(500),
                    format!("Failed to generate embedding: {e}"),
                    None,
                )
            })?;
        if embeddings.is_empty() {
            return Err(ErrorData::new(
                ErrorCode(500),
                "No embeddings generated",
                None,
            ));
        }
        Ok(embeddings[0].clone())
    }

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }

    #[tool(description = "Search for data relevant to the user question")]
    async fn search_data(
        &self,
        Parameters(UserQuestion { question }): Parameters<UserQuestion>,
    ) -> Result<CallToolResult, McpError> {
        // Build the OpenSearch query here
        let query_embedding = self.generate_embedding(&question).await?;
        let query_body = json!({
            "_source": ["titles.title", "subjects.subject", "descriptions.description", "url", "doi"],
            "query": {
                "knn": {
                    "emb": {
                        "vector": query_embedding,
                        "k": 5
                    }
                }
            }
        });
        // let query_body = json!({
        //     "query": {
        //         "query_string": {
        //             "default_operator": "AND",
        //             "default_field": "_all_fields",
        //             "query": question,
        //             // "query": format!("*{question}*"),
        //         }
        //     }
        // });

        // let query_body = json!({
        //     "query": {
        //         "bool": {
        //             "filter": [
        //                 {"term": {"format": "CSV"}}
        //             ],
        //             "must": {
        //                 "knn": {
        //                     "embedding": {
        //                         "vector": query_embedding,
        //                         "k": 5
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // });
        // Or nested field to have multiple embeddings per document
        // {
        //     "query": {
        //         "nested": {
        //             "path": "my_vectors",
        //             "query": {
        //                 "knn": {
        //                     "my_vectors.vector": {
        //                         "vector": query_embedding,
        //                         "k": 10
        //                     }
        //                 }
        //             },
        //             "score_mode": "max"
        //         }
        //     }
        // }

        // Execute the OpenSearch request
        let response = self
            .opensearch_client
            .search(opensearch::SearchParts::Index(&[OPENSEARCH_INDEX_DATASETS]))
            .body(query_body)
            .send()
            .await;
        match response {
            Ok(resp) => {
                tracing::debug!("OpenSearch response status: {resp:?}");
                let resp_json = resp.json::<serde_json::Value>().await.map_err(|e| {
                    tracing::error!("Failed to parse OpenSearch response: {}", e);
                    McpError::internal_error(
                        "Failed to parse OpenSearch response",
                        Some(json!({"error": e.to_string()})),
                    )
                })?;
                let total_found = resp_json["hits"]["total"]["value"].as_u64().unwrap_or(0);
                // tracing::debug!("MCP OpenSearch JSON response: {resp_json:?}");
                let empty_hits = vec![];
                let hits_array = resp_json["hits"]["hits"].as_array().unwrap_or(&empty_hits);
                tracing::debug!("{hits_array:?}");
                // Convert OpenSearch hits to our own SearchHit struct
                let hits: Vec<SearchHit> = hits_array
                    .iter()
                    .take(10)
                    .map(|hit| {
                        let source = &hit["_source"];
                        let id = hit["_id"].as_str().unwrap_or("").to_string();
                        let empty_titles = vec![];
                        let titles = source["titles"].as_array().unwrap_or(&empty_titles);
                        let title = titles
                            .first()
                            .and_then(|t| t["title"].as_str())
                            .unwrap_or("")
                            .to_string();
                        let description = source["descriptions"]
                            .as_array()
                            .and_then(|arr| arr.first())
                            .and_then(|d| d["description"].as_str())
                            .map(|d| {
                                if d.len() > 300 {
                                    format!("{}...", &d[..300])
                                } else {
                                    d.to_string()
                                }
                            })
                            .unwrap_or_else(|| "No description available".to_string());
                        let publication_date =
                            source["publicationYear"].as_str().unwrap_or("").to_string();
                        let keywords = source["subjects"].as_array().map(|subjects| {
                            subjects
                                .iter()
                                .filter_map(|s| s["subject"].as_str().map(|ss| ss.to_string()))
                                .collect()
                        });
                        let creators = source["creators"].as_array().map(|creators| {
                            creators
                                .iter()
                                .filter_map(|c| c["creatorName"].as_str().map(|n| n.to_string()))
                                .collect()
                        });
                        SearchHit {
                            id,
                            title,
                            description,
                            publication_date,
                            keywords,
                            creators,
                            url: source["url"].as_str().unwrap_or("").to_string(),
                            resource_type: "dataset".to_string(),
                            score: hit["_score"].as_f64(),
                        }
                    })
                    .collect();
                let search_result = SearchResult { total_found, hits };
                let json_content = serde_json::to_string_pretty(&search_result).map_err(|e| {
                    McpError::internal_error(
                        "Failed to serialize search results",
                        Some(json!({"error": e.to_string()})),
                    )
                })?;
                // tracing::debug!("MCP search results: {json_content}");
                Ok(CallToolResult::success(vec![Content::text(json_content)]))
            }
            Err(e) => {
                tracing::error!("Failed to make request to OpenSearch: {}", e);
                Err(McpError::internal_error(
                    "Failed to connect to OpenSearch",
                    Some(json!({"error": e.to_string()})),
                ))
            }
        }
    }

    #[tool(description = "Search for tools relevant to the user question")]
    async fn search_tool(
        &self,
        Parameters(UserQuestion {
            question: _question,
        }): Parameters<UserQuestion>,
    ) -> Result<CallToolResult, McpError> {
        // TODO: implement
        let search_result = SearchResult {
            total_found: 8292030,
            hits: vec![SearchHit {
                id: "9x6qrJgBTkyZK1Kx4HAC".to_string(),
                title: "JupyterLab".to_string(),
                description: "Notebooks".to_string(),
                publication_date: "2016-08-29".to_string(),
                keywords: Some(vec!["Data Science".to_string()]),
                creators: Some(vec!["Lastname, Firstname".to_string()]),
                url: "https://jupyter.org/".to_string(),
                resource_type: "dataset".to_string(),
                score: None,
            }],
        };
        let json_content = serde_json::to_string_pretty(&search_result).map_err(|e| {
            McpError::internal_error(
                "Failed to serialize search results",
                Some(json!({"error": e.to_string()})),
            )
        })?;
        Ok(CallToolResult::success(vec![Content::text(json_content)]))
    }

    // TODO: search_citations
    // #[tool(description = "Search citations related to datasets or tools relevant to the user question")]
    // async fn search_citations(
}

#[tool_handler]
impl ServerHandler for DataCommonsTools {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_prompts()
                .enable_resources()
                .enable_tools()
                .build(),
            server_info: Implementation::from_build_env(),
            instructions: Some("This server provides a search tool that can search for scientific data in public repositories.".to_string()),
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            resources: vec![self._create_resource_text("meta://repositories", "zenodo")],
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        ReadResourceRequestParam { uri }: ReadResourceRequestParam,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        match uri.as_str() {
            "meta://repositories" => {
                // TODO: list of data repositories included in the search API?
                Ok(ReadResourceResult {
                    contents: vec![ResourceContents::text("zenodo", uri)],
                })
            }
            _ => Err(McpError::resource_not_found(
                "resource_not_found",
                Some(json!({
                    "uri": uri
                })),
            )),
        }
    }

    async fn list_prompts(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListPromptsResult, McpError> {
        Ok(ListPromptsResult {
            next_cursor: None,
            prompts: vec![Prompt::new(
                "example_prompt",
                Some("This is an example prompt that takes one required argument, search_query"),
                Some(vec![PromptArgument {
                    name: "search_query".to_string(),
                    description: Some("Search query to find data".to_string()),
                    required: Some(true),
                }]),
            )],
        })
    }

    async fn get_prompt(
        &self,
        GetPromptRequestParam { name, arguments }: GetPromptRequestParam,
        _: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        match name.as_str() {
            "example_prompt" => {
                let search_query = arguments
                    .and_then(|json| json.get("search_query")?.as_str().map(|s| s.to_string()))
                    .ok_or_else(|| {
                        McpError::invalid_params("No message provided to example_prompt", None)
                    })?;

                let prompt = format!("I am looking for data about {search_query}");
                Ok(GetPromptResult {
                    description: None,
                    messages: vec![PromptMessage {
                        role: PromptMessageRole::User,
                        content: PromptMessageContent::text(prompt),
                    }],
                })
            }
            _ => Err(McpError::invalid_params("prompt not found", None)),
        }
    }

    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(),
        })
    }

    async fn initialize(
        &self,
        _request: InitializeRequestParam,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
            let initialize_headers = &http_request_part.headers;
            let initialize_uri = &http_request_part.uri;
            tracing::info!(?initialize_headers, %initialize_uri, "initialize from http server");
        }
        Ok(self.get_info())
    }
}
