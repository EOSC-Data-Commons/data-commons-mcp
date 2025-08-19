use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub mod error;
pub mod mcp;
pub mod search;
use data_commons_mcp::{AppState, Args, build_router};
use error::AppResult;

// OpenAPI generation: https://github.com/juhaku/utoipa/blob/master/examples/axum-multipart/src/main.rs
// MCP client: https://github.com/modelcontextprotocol/rust-sdk/blob/main/examples/clients/src/streamable_http.rs

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

    let router = build_router(&args).await.unwrap();
    let app = router.into_make_service();
    let listener = tokio::net::TcpListener::bind(&args.bind_address).await?;

    // Improve graceful shutdown
    let shutdown_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        tracing::info!("Received CTRL+C, shutting down gracefully...");
    };
    tracing::info!(
        "Starting Streamable HTTP MCP server at http://{}/mcp, {}with OpenSearch at {}",
        args.bind_address,
        if args.mcp_only {
            "".to_string()
        } else {
            format!("OpenAPI UI at http://{}/docs, ", args.bind_address)
        },
        args.opensearch_url
    );
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;
    Ok(())
}
