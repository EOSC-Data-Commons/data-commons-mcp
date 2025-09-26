use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

use llm::builder::{LLMBackend, LLMBuilder};
use serde::Serialize;

use crate::error::{AppError, AppResult};
use crate::search::{ApiChatMessage, SearchResponse};

/// Structure for logging search conversations and responses
#[derive(Debug, Serialize)]
pub struct SearchLog {
    timestamp: String,
    request_id: String,
    model: String,
    stream: bool,
    conversation: Vec<ApiChatMessage>,
    response: SearchResponse,
    execution_time_ms: u64,
}

impl SearchLog {
    pub fn new(
        request_id: String,
        model: String,
        stream: bool,
        conversation: Vec<ApiChatMessage>,
        response: SearchResponse,
        execution_time_ms: u64,
    ) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let timestamp = format!("{timestamp}");
        Self {
            timestamp,
            request_id,
            model,
            stream,
            conversation,
            response,
            execution_time_ms,
        }
    }

    /// Write the log entry to the search.log file
    pub fn write_to_file(&self) -> AppResult<()> {
        let log_entry = serde_json::to_string(self)?;
        let log_path = "data/search.log";
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
            .map_err(|e| AppError::Llm(format!("Failed to open log file: {e}")))?;
        writeln!(file, "{log_entry}")
            .map_err(|e| AppError::Llm(format!("Failed to write to log file: {e}")))?;
        Ok(())
    }
}

/// Determines the LLM backend and API key based on available environment variables
/// OpenAI takes priority if both are available
pub fn get_llm_config(model: &str) -> Result<(LLMBackend, String, String, Option<String>), String> {
    // Parse provider/model_name from input
    let parts: Vec<&str> = model.splitn(2, '/').collect();
    let (provider, model_name) = match parts.as_slice() {
        [provider, model_name] => (provider.to_string(), model_name.to_string()),
        [single] => (single.to_string(), single.to_string()),
        _ => ("openai".to_string(), model.to_string()), // fallback
    };
    let backend_result: Result<(LLMBackend, String, Option<String>), String> = match provider.as_str() {
        "openai" => {
            if std::env::var("OPENAI_API_KEY").is_ok() {
                match std::env::var("OPENAI_API_KEY") {
                    Ok(key) => Ok((LLMBackend::OpenAI, key, None)),
                    Err(_) => Err("OPENAI_API_KEY environment variable not set".to_string()),
                }
            } else {
                Err("OPENAI_API_KEY environment variable not set".to_string())
            }
        }
        "mistralai" => {
            if std::env::var("MISTRAL_API_KEY").is_ok() {
                match std::env::var("MISTRAL_API_KEY") {
                    Ok(key) => Ok((LLMBackend::Mistral, key, None)),
                    Err(_) => Err("MISTRAL_API_KEY environment variable not set".to_string()),
                }
            } else {
                Err("MISTRAL_API_KEY environment variable not set".to_string())
            }
        }
        "einfracz" => {
            if std::env::var("EINFRACZ_API_KEY").is_ok() {
                match std::env::var("EINFRACZ_API_KEY") {
                    Ok(key) => Ok((LLMBackend::Groq, key, Some("https://chat.ai.e-infra.cz/api/v1/".to_string()))),
                    Err(_) => Err("EINFRACZ_API_KEY environment variable not set".to_string()),
                }
            } else {
                Err("EINFRACZ_API_KEY environment variable not set".to_string())
            }
        }
        "openrouter" => {
            if std::env::var("OPENROUTER_API_KEY").is_ok() {
                match std::env::var("OPENROUTER_API_KEY") {
                    Ok(key) => Ok((LLMBackend::OpenRouter, key, None)),
                    Err(_) => Err("OPENROUTER_API_KEY environment variable not set".to_string()),
                }
            } else {
                Err("OPENROUTER_API_KEY environment variable not set".to_string())
            }
        }
        "groq" => {
            if std::env::var("GROQ_API_KEY").is_ok() {
                match std::env::var("GROQ_API_KEY") {
                    Ok(key) => Ok((LLMBackend::Groq, key, None)),
                    Err(_) => Err("GROQ_API_KEY environment variable not set".to_string()),
                }
            } else {
                Err("GROQ_API_KEY environment variable not set".to_string())
            }
        }
        _ => Err(format!("Unknown provider: {provider}")),
    };

    match backend_result {
        Ok((backend, api_key, base_url)) => Ok((backend, api_key, model_name, base_url)),
        Err(e) => Err(e),
    }
}
