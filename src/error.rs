use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;
use std::fmt;

/// Custom error type for the application
#[derive(Debug)]
pub enum AppError {
    /// Serialization/deserialization errors
    Serde(serde_json::Error),
    /// MCP client errors
    Mcp(rmcp::ServiceError),
    /// OpenSearch errors
    OpenSearch(opensearch::Error),
    /// System time errors
    SystemTime(std::time::SystemTimeError),
    /// I/O errors
    Io(std::io::Error),
    /// LLM client errors
    Llm(String),
    /// Fastembed errors
    Embed(String),
    /// No data found errors
    NoDataFound(String),
    // /// HTTP client errors (reqwest)
    // Http(reqwest::Error),
    // /// API key missing or invalid
    // ApiKey(String),
    // /// Search service errors
    // Search(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::Serde(err) => write!(f, "Serialization error: {err}"),
            AppError::Mcp(err) => write!(f, "MCP error: {err}"),
            AppError::OpenSearch(err) => write!(f, "OpenSearch error: {err}"),
            AppError::SystemTime(err) => write!(f, "System time error: {err}"),
            AppError::Io(err) => write!(f, "I/O error: {err}"),
            AppError::Llm(msg) => write!(f, "LLM error: {msg}"),
            AppError::Embed(msg) => write!(f, "Embeddings generation error: {msg}"),
            AppError::NoDataFound(msg) => write!(f, "No data found: {msg}"),
            // AppError::Http(err) => write!(f, "HTTP error: {}", err),
            // AppError::ApiKey(msg) => write!(f, "API key error: {}", msg),
            // AppError::Search(msg) => write!(f, "Search error: {}", msg),
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AppError::Serde(err) => Some(err),
            AppError::Mcp(err) => Some(err),
            AppError::OpenSearch(err) => Some(err),
            AppError::SystemTime(err) => Some(err),
            AppError::Io(err) => Some(err),
            AppError::Llm(_) => None,
            AppError::Embed(_) => None,
            AppError::NoDataFound(_) => None,
            // AppError::Http(err) => Some(err),
            // _ => None,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            AppError::Serde(_) => (StatusCode::BAD_REQUEST, "Invalid JSON format"),
            AppError::Mcp(_) => (StatusCode::BAD_GATEWAY, "MCP client error"),
            AppError::OpenSearch(_) => (StatusCode::BAD_GATEWAY, "OpenSearch service error"),
            AppError::SystemTime(_) => (StatusCode::INTERNAL_SERVER_ERROR, "System time error"),
            AppError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "I/O error"),
            AppError::Llm(_) => (StatusCode::INTERNAL_SERVER_ERROR, "LLM service error"),
            AppError::Embed(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Embedding generation error",
            ),
            AppError::NoDataFound(_) => (StatusCode::NOT_FOUND, "No data found"),
            // AppError::Http(_) => (StatusCode::BAD_GATEWAY, "External service error"),
            // AppError::ApiKey(_) => (StatusCode::UNAUTHORIZED, "Invalid API key"),
            // AppError::Search(_) => (StatusCode::BAD_GATEWAY, "Search service error"),
        };

        let body = Json(json!({
            "error": error_message,
            "details": self.to_string()
        }));

        (status, body).into_response()
    }
}

// Conversion implementations for common error types
impl From<serde_json::Error> for AppError {
    fn from(err: serde_json::Error) -> Self {
        AppError::Serde(err)
    }
}

impl From<rmcp::ServiceError> for AppError {
    fn from(err: rmcp::ServiceError) -> Self {
        AppError::Mcp(err)
    }
}

impl From<opensearch::Error> for AppError {
    fn from(err: opensearch::Error) -> Self {
        AppError::OpenSearch(err)
    }
}

impl From<std::time::SystemTimeError> for AppError {
    fn from(err: std::time::SystemTimeError) -> Self {
        AppError::SystemTime(err)
    }
}

impl From<std::io::Error> for AppError {
    fn from(err: std::io::Error) -> Self {
        AppError::Io(err)
    }
}

/// Convenient type alias for Results using AppError
pub type AppResult<T> = Result<T, AppError>;

// impl From<reqwest::Error> for AppError {
//     fn from(err: reqwest::Error) -> Self {
//         AppError::Http(err)
//     }
// }

// /// Helper functions for creating common errors
// impl AppError {
//     pub fn mcp(msg: impl Into<String>) -> Self {
//         AppError::Mcp(msg.into())
//     }

//     pub fn search(msg: impl Into<String>) -> Self {
//         AppError::Search(msg.into())
//     }

//     pub fn api_key(msg: impl Into<String>) -> Self {
//         AppError::ApiKey(msg.into())
//     }
// }
