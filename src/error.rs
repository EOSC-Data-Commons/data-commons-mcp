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
    /// System time errors
    SystemTime(std::time::SystemTimeError),
    /// I/O errors
    Io(std::io::Error),
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
            AppError::SystemTime(err) => write!(f, "System time error: {err}"),
            AppError::Io(err) => write!(f, "I/O error: {err}"),
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
            AppError::SystemTime(err) => Some(err),
            AppError::Io(err) => Some(err),
            // AppError::Http(err) => Some(err),
            // _ => None,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_message) = match &self {
            AppError::Serde(_) => (StatusCode::BAD_REQUEST, "Invalid JSON format"),
            AppError::Mcp(_) => (StatusCode::BAD_GATEWAY, "Search service error"),
            AppError::SystemTime(_) => (StatusCode::INTERNAL_SERVER_ERROR, "System time error"),
            AppError::Io(_) => (StatusCode::INTERNAL_SERVER_ERROR, "I/O error"),
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

// impl From<reqwest::Error> for AppError {
//     fn from(err: reqwest::Error) -> Self {
//         AppError::Http(err)
//     }
// }

/// Convenient type alias for Results using AppError
pub type AppResult<T> = Result<T, AppError>;

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
