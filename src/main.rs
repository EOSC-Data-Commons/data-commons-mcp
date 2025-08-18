use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use utoipa::OpenApi;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;
use utoipa_swagger_ui::SwaggerUi;

use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};

mod error;
mod mcp;
use mcp::DataCommonsTools;
mod search;
use error::AppResult;
use search::ADDRESS;
// OpenAPI generation: https://github.com/juhaku/utoipa/blob/master/examples/axum-multipart/src/main.rs
// MCP client: https://github.com/modelcontextprotocol/rust-sdk/blob/main/examples/clients/src/streamable_http.rs

/// Command line arguments for the `data-commons-mcp` server
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// OpenSearch URL
    #[arg(
        short = 's',
        long = "opensearch-url",
        env = "OPENSEARCH_URL",
        default_value = "http://127.0.0.1:9200"
    )]
    opensearch_url: String,

    /// Only deploy the MCP endpoint without the search endpoint
    #[arg(long = "mcp-only", default_value_t = false)]
    mcp_only: bool,
}

/// OpenAPI documentation for the API
#[derive(OpenApi)]
#[openapi(info(
    title = "EOSC Data Commons Conversational Search API",
    version = "1.0.0",
    description = "Conversational Search API to find relevant data for a user question in natural language, developed for the EOSC Data Commons project"
))]
struct ApiDoc;

#[tokio::main]
async fn main() -> AppResult<()> {
    let args = Args::parse();
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                // In dev: info for all, debug for data_commons_mcp only
                #[cfg(debug_assertions)]
                return "info,data_commons_mcp=debug".to_string().into();
                #[cfg(not(debug_assertions))]
                return "info".to_string().into();
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let opensearch_url = args.opensearch_url.clone();
    let mcp_service = StreamableHttpService::new(
        move || {
            DataCommonsTools::new(&opensearch_url).map_err(|e| std::io::Error::other(e.to_string()))
        },
        LocalSessionManager::default().into(),
        Default::default(),
    );
    // let mcp_service = StreamableHttpService::new(
    //     || Ok(DataCommonsTools::new()),
    //     LocalSessionManager::default().into(),
    //     Default::default(),
    // );

    // Configure router with MCP and conditionally with search endpoint and OpenAPI docs
    let router = if args.mcp_only {
        // MCP-only mode: just serve the MCP endpoint
        axum::Router::new().nest_service("/mcp", mcp_service)
    } else {
        // Default mode includes a search endpoint and OpenAPI docs
        let (router, api) = OpenApiRouter::with_openapi(ApiDoc::openapi())
            .routes(routes!(search::search_handler))
            .split_for_parts();
        router
            .nest_service("/mcp", mcp_service)
            .merge(SwaggerUi::new("/docs").url("/openapi.json", api))
    };

    let app = router.into_make_service();
    let listener = tokio::net::TcpListener::bind(ADDRESS).await?;

    // Improve graceful shutdown
    let shutdown_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        tracing::info!("Received CTRL+C, shutting down gracefully...");
    };
    tracing::info!(
        "Starting Streamable HTTP MCP server at http://{ADDRESS}/mcp, {}with OpenSearch at {}",
        if args.mcp_only {
            "".to_string()
        } else {
            format!("OpenAPI UI at http://{ADDRESS}/docs, ")
        },
        args.opensearch_url
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;
    Ok(())
}
