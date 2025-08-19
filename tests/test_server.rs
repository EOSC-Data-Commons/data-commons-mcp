use data_commons_mcp::{Args, build_router};
use reqwest::Client;
use tokio::spawn;

use rmcp::{
    ServiceExt,
    model::{ClientCapabilities, ClientInfo, Implementation},
    transport::StreamableHttpClientTransport,
};

/// Spawn a test server
async fn spawn_test_server(port: &str) {
    let args = Args {
        opensearch_url: "http://localhost:9200".to_string(),
        mcp_only: false,
        bind_address: format!("0.0.0.0:{port}"),
    };
    let router = build_router(&args).await.unwrap();
    let listener = tokio::net::TcpListener::bind(&args.bind_address)
        .await
        .unwrap();
    spawn(async move {
        axum::serve(listener, router.into_make_service())
            .await
            .unwrap();
    });
}

const ADDRESS: &str = "127.0.0.1";

#[tokio::test]
async fn test_search_endpoint() {
    spawn_test_server("8000").await;
    let body = serde_json::json!({
        "messages": [
            {"role": "user", "content": "insulin"}
        ],
        "model": "mistral/mistral-small-latest",
        // "model": "openai/gpt-4.1-nano",
        // "model": "groq/moonshotai/kimi-k2-instruct",
        "stream": false,
    });
    let client = Client::new();
    let res = client
        .post(format!("http://{ADDRESS}:8000/search"))
        .header("Content-Type", "application/json")
        .header("Authorization", "SECRET_KEY")
        .json(&body)
        .send()
        .await
        .unwrap();

    assert!(res.status().is_success());
    let json: serde_json::Value = serde_json::from_str(&res.text().await.unwrap()).unwrap();
    println!("Response JSON: {json}");
    assert!(
        json["hits"].is_array() && !json["hits"].as_array().unwrap().is_empty(),
        "No hits found in response JSON"
    );
    assert!(
        json["summary"].is_string() && !json["summary"].as_str().unwrap().is_empty(),
        "Summary is empty in response JSON"
    );
}

#[tokio::test]
async fn test_mcp_endpoint() {
    spawn_test_server("8001").await;
    let transport = StreamableHttpClientTransport::from_uri(format!("http://{ADDRESS}:8001/mcp"));
    let client_info = ClientInfo {
        protocol_version: Default::default(),
        capabilities: ClientCapabilities::default(),
        client_info: Implementation {
            name: "MCP streamable HTTP client".to_string(),
            version: "0.0.1".to_string(),
        },
    };
    let mcp_client = client_info.serve(transport).await.unwrap();

    let tools = mcp_client.list_tools(Default::default()).await.unwrap();
    for tool in &tools.tools {
        let schema_value = serde_json::Value::Object(tool.input_schema.as_ref().clone());
        println!("Tool: {} - {}", tool.name, schema_value);
        assert!(!tool.name.trim().is_empty(), "Tool name is empty");
        // assert!(schema_value.is && !tool.input_schema.as_ref().unwrap().is_empty(), "Tool '{}' input_schema is empty", tool.name);

        // let schema_value = serde_json::Value::Object(tool.input_schema.as_ref().clone());
        // let function = FunctionBuilder::new(tool.name.to_string())
        //     .description(tool.description.as_deref().unwrap_or(""))
        //     .json_schema(schema_value);
        // llm_builder = llm_builder.function(function);
    }
    // Assert required tools are present
    let tool_names: Vec<_> = tools
        .tools
        .iter()
        .map(|tool| tool.name.to_string())
        .collect();
    assert!(
        tool_names.contains(&"search_data".to_string()),
        "Tool 'search_data' is missing"
    );
    assert!(
        tool_names.contains(&"search_tool".to_string()),
        "Tool 'search_tool' is missing"
    );
}
