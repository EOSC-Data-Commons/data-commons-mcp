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
use utoipa::ToSchema;

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
    // pub query: String,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Serialize, Deserialize, ToSchema, Clone)]
pub struct SearchHit {
    #[schema(example = "5173026")]
    pub id: u64,
    #[schema(example = "Item Title")]
    pub title: String,
    #[schema(example = "Item Description")]
    pub description: String,
    #[schema(example = "10.48550/arXiv.2410.06062")]
    pub doi: Option<String>,
    #[schema(example = "2025-10-08")]
    pub publication_date: String,
    // #[schema(example = "['Information Retrieval, Data Science']")]
    pub keywords: Option<Vec<String>>,
    // #[schema(example = "['Emonet, Vincent']")]
    pub creators: Option<Vec<String>>,
    #[schema(example = "https://zenodo.org/record/5173026")]
    pub zenodo_url: String,
    #[schema(example = "dataset")]
    pub resource_type: String,
    #[schema(example = "0.5")]
    pub score: Option<f64>,
}

/// Represents a response from Zenodo API
#[derive(Debug, Deserialize, Serialize)]
pub struct ZenodoResponse {
    pub hits: ZenodoHits,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ZenodoHits {
    pub total: u64,
    pub hits: Vec<ZenodoRecord>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ZenodoRecord {
    pub id: u64,
    pub title: String,
    pub description: Option<String>,
    pub doi: Option<String>,
    pub created: String,
    pub modified: String,
    #[serde(rename = "conceptdoi")]
    pub concept_doi: Option<String>,
    pub metadata: Option<ZenodoMetadata>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ZenodoMetadata {
    pub title: String,
    pub description: Option<String>,
    pub creators: Option<Vec<ZenodoCreator>>,
    pub publication_date: Option<String>,
    pub keywords: Option<Vec<String>>,
    pub subjects: Option<Vec<ZenodoSubject>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ZenodoCreator {
    pub name: Option<String>,
    pub affiliation: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ZenodoSubject {
    pub term: Option<String>,
    pub identifier: Option<String>,
}

#[derive(Clone)]
pub struct DataCommonsTools {
    tool_router: ToolRouter<DataCommonsTools>,
    http_client: reqwest::Client,
}

#[tool_router]
impl DataCommonsTools {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            http_client: reqwest::Client::new(),
        }
    }

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }

    // TODO: search_citations
    // #[tool(description = "Search citations related to datasets or tools relevant to the user question")]
    // async fn search_citations(

    #[tool(description = "Search for tools relevant to the user question")]
    async fn search_tool(
        &self,
        Parameters(UserQuestion { question }): Parameters<UserQuestion>,
    ) -> Result<CallToolResult, McpError> {
        // TODO: implement
        let search_result = SearchResult {
            total_found: 8292030,
            hits: vec![SearchHit {
                id: 427542,
                title: "JupyterLab".to_string(),
                description: "Notebooks".to_string(),
                doi: Some("10.5281/zenodo.427542".to_string()),
                publication_date: "2016-08-29".to_string(),
                keywords: Some(vec!["Data Science".to_string()]),
                creators: Some(vec!["Lastname, Firstname".to_string()]),
                zenodo_url: "https://zenodo.org/record/427542".to_string(),
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

    #[tool(description = "Search for data relevant to the user question")]
    async fn search_data(
        &self,
        Parameters(UserQuestion { question }): Parameters<UserQuestion>,
    ) -> Result<CallToolResult, McpError> {
        // tracing::info!("User question: {}", question);
        const ZENODO_API_URL: &str = "https://zenodo.org/api/records";
        // Build query parameters
        let mut params = vec![("q", question.as_str()), ("size", "10"), ("page", "1")];
        // Add access token if available from environment
        let access_token = std::env::var("ZENODO_ACCESS_TOKEN").ok();
        if let Some(token) = &access_token {
            params.push(("access_token", token.as_str()));
        }
        // Make the HTTP request
        match self
            .http_client
            .get(ZENODO_API_URL)
            .query(&params)
            .send()
            .await
        {
            Ok(response) => {
                if !response.status().is_success() {
                    let status = response.status();
                    let error_text = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Unknown error".to_string());
                    tracing::error!("Zenodo API error: {} - {}", status, error_text);
                    return Err(McpError::internal_error(
                        format!(
                            "API Error: {} {}",
                            status.as_u16(),
                            status.canonical_reason().unwrap_or("Unknown")
                        ),
                        Some(json!({
                            "status": status.as_u16(),
                            "error": error_text
                        })),
                    ));
                }
                match response.json::<ZenodoResponse>().await {
                    Ok(zenodo_data) => {
                        let hits: Vec<SearchHit> = zenodo_data
                            .hits
                            .hits
                            .iter()
                            .take(10) // Limit to first 10 results
                            .map(|record| {
                                let title = record
                                    .metadata
                                    .as_ref()
                                    .map(|m| &m.title)
                                    .unwrap_or(&record.title)
                                    .clone();
                                let description = record
                                    .metadata
                                    .as_ref()
                                    .and_then(|m| m.description.as_ref())
                                    .or(record.description.as_ref())
                                    .map(|d| {
                                        // Truncate long descriptions
                                        if d.len() > 300 {
                                            format!("{}...", &d[..300])
                                        } else {
                                            d.to_string()
                                        }
                                    })
                                    .unwrap_or_else(|| "No description available".to_string());
                                let publication_date = record
                                    .metadata
                                    .as_ref()
                                    .and_then(|m| m.publication_date.as_ref())
                                    .unwrap_or(&record.created)
                                    .clone();
                                let keywords =
                                    record.metadata.as_ref().and_then(|m| m.keywords.clone());
                                let creators = record
                                    .metadata
                                    .as_ref()
                                    .and_then(|m| m.creators.as_ref())
                                    .map(|creators| {
                                        creators.iter().filter_map(|c| c.name.clone()).collect()
                                    });
                                SearchHit {
                                    id: record.id,
                                    title,
                                    description,
                                    doi: record.doi.clone(),
                                    publication_date,
                                    keywords,
                                    creators,
                                    zenodo_url: format!("https://zenodo.org/record/{}", record.id),
                                    resource_type: "dataset".to_string(),
                                    score: None,
                                }
                            })
                            .collect();
                        let search_result = SearchResult {
                            total_found: zenodo_data.hits.total,
                            // query: question.clone(),
                            hits,
                        };

                        // Return as JSON content
                        let json_content =
                            serde_json::to_string_pretty(&search_result).map_err(|e| {
                                McpError::internal_error(
                                    "Failed to serialize search results",
                                    Some(json!({"error": e.to_string()})),
                                )
                            })?;

                        Ok(CallToolResult::success(vec![Content::text(json_content)]))
                    }
                    Err(e) => {
                        tracing::error!("Failed to parse search API response: {}", e);
                        Err(McpError::internal_error(
                            "Failed to parse search API response",
                            Some(json!({"error": e.to_string()})),
                        ))
                    }
                }
            }
            Err(e) => {
                tracing::error!("Failed to make request to the search API: {}", e);
                Err(McpError::internal_error(
                    "Failed to connect to the search API",
                    Some(json!({"error": e.to_string()})),
                ))
            }
        }
    }
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
                // TODO: list of data repositories included in the search API
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
