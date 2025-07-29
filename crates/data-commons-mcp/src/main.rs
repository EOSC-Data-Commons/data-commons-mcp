use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    {self},
};

mod tools;
use tools::DataCommonsTools;

const ADDRESS: &str = "0.0.0.0:8000";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".to_string().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let service = StreamableHttpService::new(
        || Ok(DataCommonsTools::new()),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    tracing::info!("Started Streamable HTTP MCP server on http://{ADDRESS}/mcp");
    let router = axum::Router::new().nest_service("/mcp", service);
    let tcp_listener = tokio::net::TcpListener::bind(ADDRESS).await?;

    // Improve graceful shutdown
    let shutdown_signal = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        tracing::info!("Received CTRL+C, shutting down gracefully...");
    };

    axum::serve(tcp_listener, router)
        .with_graceful_shutdown(shutdown_signal)
        .await?;
    tracing::info!("Server shutdown complete");
    Ok(())
}
