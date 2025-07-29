# 🔭 EOSC Data Commons MCP server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server with an HTTP API endpoint to access data from various open access data publishers, developed for the [EOSC Data Commons project](https://eosc.eu/horizon-europe-projects/eosc-data-commons/).

## 🧩 Endpoints

The HTTP API comprises 2 main endpoints:

- `/mcp`: **MCP server** to search for relevant data using the EOSC Data Commons search APU based on the user question
  - Use [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) with Streamable HTTP transport

- `/search`: regular **HTTP POST** JSON that enables querying the MCP server with a LLM provider
  - Use [`axum`](https://github.com/tokio-rs/axum), [`utoipa`](https://github.com/juhaku/utoipa) for OpenAPI spec generation, [`llm`](*https://github.com/graniet/llm*) to interact with LLM providers (e.g. [Mistral](https://admin.mistral.ai/organization/api-keys))


## 🛠️ Development

### 📥 Install dependencies

```sh
rustup update
cargo install cargo-deny
```

Create a `.cargo/config.toml` file with your [Mistral API key](https://admin.mistral.ai/organization/api-keys) (required for the HTTP API in dev):

```toml
[env]
MISTRAL_API_KEY = "YOUR_API_KEY"
MISTRAL_MODEL = "mistral-medium-latest"
```

### ⚡️ Start server

Start the **MCP server** in dev at http://localhost:8000/mcp, with OpenAPI UI at http://localhost:8000/docs

```sh
cargo run
```

> Example `curl` request:
>
> ```sh
> curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -H "Authorization: SECRET_KEY" -d '[{"role": "user", "content": "data about insulin in EU"}]'
> ```

### 🔌 Connect MCP client

Follow the instructions of your client, and use the `/mcp` URL of your deployed server (e.g. http://127.0.0.1:8000/mcp)

#### 🐙 VSCode GitHub Copilot

Add a new MCP server through the VSCode UI:

- Open the Command Palette (`ctrl+shift+p` or `cmd+shift+p`)
- Search for `MCP: Add Server...`
- Choose `HTTP`, and provide the MCP server URL http://127.0.0.1:8000/mcp

Your `mcp.json` should look like:

```json
{
    "servers": {
        "data-commons-mcp-server": {
            "url": "http://127.0.0.1:8000/mcp",
            "type": "http"
        }
    },
    "inputs": []
}
```

### 📦 Build for production

Build binary in `target/release/`

```sh
cargo build --release
```

> Start the server with:
>
> ```sh
> chmod +x target/release/data-commons-mcp
> ./target/release/data-commons-mcp
> ```

### 🐳 Deploy with Docker

Create a `.env` file with the API keys:

```sh
MISTRAL_API_KEY=YOUR_API_KEY
```

Build and deploy the service:

```sh
docker compose up
```

### 🧼 Format & lint

Automatically format the codebase using `rustfmt`:

```sh
cargo fmt
```

Lint with `clippy`:

```sh
cargo clippy --all
```

### ⛓️ Check supply chain

Check the dependency supply chain: licenses (only accept dependencies with OSI or FSF approved licenses), and vulnerabilities (CVE advisories).

```sh
cargo deny check
```

Make sure dependencies have been updated:

```sh
cargo update
```

