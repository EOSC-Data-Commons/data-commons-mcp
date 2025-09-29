use axum::routing::get_service;
use clap::Parser;
use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::fs::ServeDir;
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_swagger_ui::SwaggerUi;

mod chat;
mod error;
mod mcp;
mod utils;
use crate::error::AppResult;
use crate::mcp::DataCommonsTools;

/// Command line arguments for the `data-commons-mcp` server
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Address where to deploy this MCP endpoint
    #[arg(
        short = 'b',
        long = "bind",
        env = "BIND_ADDRESS",
        default_value = "0.0.0.0:8000"
    )]
    pub bind_address: String,

    /// OpenSearch URL
    #[arg(
        short = 's',
        long = "opensearch-url",
        env = "OPENSEARCH_URL",
        default_value = "http://127.0.0.1:9200"
    )]
    pub opensearch_url: String,

    /// Only deploy the MCP endpoint without the chat search endpoint
    #[arg(long = "mcp-only", env = "MCP_ONLY", default_value_t = false)]
    pub mcp_only: bool,

    /// Enable CORS to allow requests from any origin (useful in dev with local webapps)
    #[arg(long = "cors", env = "CORS_ENABLED", default_value_t = false)]
    pub cors: bool,
}

/// OpenAPI documentation for the API
#[derive(OpenApi)]
#[openapi(info(
    title = "EOSC Data Commons Conversational Search API",
    version = "1.0.0",
    description = "Conversational Search API to find relevant data and services for a user question in natural language, developed for the EOSC Data Commons project"
))]
struct ApiDoc;

#[derive(Clone)]
pub struct AppState {
    pub bind_address: String,
}

/// Build the Axum router with the configured endpoints and services
pub async fn build_router(args: &Args) -> AppResult<axum::Router> {
    let opensearch_url = args.opensearch_url.clone();
    let mcp_service = StreamableHttpService::new(
        move || {
            DataCommonsTools::new(&opensearch_url).map_err(|e| std::io::Error::other(e.to_string()))
        },
        LocalSessionManager::default().into(),
        Default::default(),
    );
    let app_state = AppState {
        bind_address: args.bind_address.clone(),
    };

    // Serve the webapp static files on /
    let webapp_service =
        get_service(ServeDir::new("src/webapp")).handle_error(|error| async move {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Internal error: {error}"),
            )
        });

    let mut router = if args.mcp_only {
        // MCP-only mode: just serve the MCP endpoint
        axum::Router::new()
            .nest_service("/mcp", mcp_service)
    } else {
        // Default mode includes a search endpoint and OpenAPI docs
        let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
            .routes(routes!(chat::chat_handler))
            .with_state(app_state)
            .split_for_parts();
        router
            .fallback_service(webapp_service.clone()) // Serve root /
            .nest_service("/search", webapp_service) // Handle UI search
            .nest_service("/mcp", mcp_service)
            .merge(SwaggerUi::new("/docs").url("/openapi.json", api))
            // .fallback(spa_fallback) // SPA fallback for client-side routing
    };
    // Apply CORS layer if enabled
    if args.cors {
        router = router.layer(CorsLayer::new().allow_origin(Any).allow_headers([
            axum::http::header::AUTHORIZATION,
            axum::http::header::CONTENT_TYPE,
        ]));
    }
    Ok(router)
}

// /// SPA fallback handler that serves index.html for non-JS file requests
// async fn spa_fallback(
//     uri: axum::http::Uri,
// ) -> Result<axum::response::Html<String>, axum::http::StatusCode> {
//     let path = uri.path();

//     // Don't serve index.html for JS/CSS/asset files - let them 404
//     if path.ends_with(".js")
//         || path.ends_with(".css")
//         || path.ends_with(".map")
//         || path.ends_with(".ico")
//         || path.ends_with(".png")
//         || path.ends_with(".svg")
//         || path.ends_with(".jpg")
//         || path.ends_with(".jpeg")
//         || path.ends_with(".gif")
//         || path.ends_with(".woff")
//         || path.ends_with(".woff2")
//         || path.ends_with(".ttf")
//     {
//         return Err(axum::http::StatusCode::NOT_FOUND);
//     }

//     // For all other paths, try to serve index.html
//     match tokio::fs::read_to_string("src/webapp/index.html").await {
//         Ok(content) => Ok(axum::response::Html(content)),
//         Err(_) => Err(axum::http::StatusCode::NOT_FOUND),
//     }
// }
