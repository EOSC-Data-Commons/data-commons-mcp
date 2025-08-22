# 🔭 EOSC Data Commons MCP server

[![Tests](https://github.com/EOSC-Data-Commons/data-commons-mcp/actions/workflows/test.yml/badge.svg)](https://github.com/EOSC-Data-Commons/data-commons-mcp/actions/workflows/test.yml)

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server exposing an HTTP POST endpoint to access data from various open-access data publishers, developed for the [EOSC Data Commons project](https://eosc.eu/horizon-europe-projects/eosc-data-commons/).

It uses a search API, and a Large Language Model (LLM) to help users find the datasets and tools they need.

## 🧩 Endpoints

The HTTP API comprises 2 main endpoints:

- `/mcp`: **MCP server** that searches for relevant data to answer a user question using the EOSC Data Commons search API
  - Uses [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) with Streamable HTTP transport
  - Available tools:
    - [x] Search datasets
    - [ ] Search tools
    - [ ] Search citations related to datasets or tools

- `/search`: simple **HTTP POST** endpoint (JSON) for querying the MCP server via an LLM provider
  - Uses [`axum`](https://github.com/tokio-rs/axum), [`utoipa`](https://github.com/juhaku/utoipa) for OpenAPI spec generation, [`llm`](https://github.com/graniet/llm) to interact with LLM providers (e.g. [Mistral](https://admin.mistral.ai/organization/api-keys), OpenAI)
  - Returns a streaming response: tool call requested, then tool call results, and final search results.

## 🛠️ Development

> [!IMPORTANT]
>
> Requirements:
>
> - [Rust](https://www.rust-lang.org/tools/install)
> - API key for a LLM provider: [Mistral.ai](https://console.mistral.ai/api-keys) or OpenAI, you can use the free tier, you just need to login
>
> Recommended VSCode extension: [`rust-analyzer`](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

### 📥 Install dev dependencies

```sh
rustup update
cargo install cargo-release cargo-deny cargo-watch git-cliff
```

Create a `.cargo/config.toml` file with your [Mistral API key](https://admin.mistral.ai/organization/api-keys) or OpenAI API key:

```toml
[env]
MISTRAL_API_KEY = "YOUR_API_KEY"
OPENAI_API_KEY = "YOUR_API_KEY"
```

### ⚡️ Start dev server

Start the **MCP server** in dev at http://localhost:8000/mcp, with OpenAPI UI at http://localhost:8000/docs

```sh
cargo run
```

Customize server configuration through CLI arguments:

```sh
cargo run -- --mcp-only --opensearch-url http://localhost:9200
```

Run and reload on change to the code:

```sh
cargo watch -x run
```

> [!NOTE]
>
> Example `curl` request:
>
> ```sh
> curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -H "Authorization: SECRET_KEY" -d '{"messages": [{"role": "user", "content": "insulin data"}], "model": "mistral/mistral-small-latest", "stream": true}'
> ```
>
> Recommended model per supported provider: 
>
> - `openai/gpt-4.1`
> - `mistral/mistral-large-latest`
> - `groq/moonshotai/kimi-k2-instruct`

### 🔌 Connect MCP client

Follow the instructions of your client, and use the `/mcp` URL of your deployed server (e.g. http://localhost:8000/mcp)

#### 🐙 VSCode GitHub Copilot

Add a new MCP server through the VSCode UI:

- Open the Command Palette (`ctrl+shift+p` or `cmd+shift+p`)
- Search for `MCP: Add Server...`
- Choose `HTTP`, and provide the MCP server URL http://localhost:8000/mcp

Your VSCode `mcp.json` should look like:

```json
{
    "servers": {
        "data-commons-mcp-server": {
            "url": "http://localhost:8000/mcp",
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

> [!NOTE]
>
> Start the server with:
>
> ```sh
> ./target/release/data-commons-mcp
> ```

### 🐳 Deploy with Docker

Create a `keys.env` file with the API keys:

```sh
MISTRAL_API_KEY=YOUR_API_KEY
SEARCH_API_KEY=SECRET_KEY_YOU_CAN_USE_IN_FRONTEND_TO_AVOID_SPAM
```

> [!TIP]
>
> `SEARCH_API_KEY` can be used to add a layer of protection against bots that might spam the LLM, if not provided no API key will be needed to query the API.

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

Automatically apply possible fixes:

```sh
cargo clippy --fix
```

### ⛓️ Check supply chain

Check the dependency supply chain: licenses (only accept dependencies with OSI or FSF approved licenses), and vulnerabilities (CVE advisories).

```sh
cargo deny check
```

Update dependencies in `Cargo.lock`:

```sh
cargo update
```

### 🏷️ Release

Dry run:

```sh
cargo release patch
```

> Or `minor` / `major`

Create release:

```sh
cargo release patch --execute
```
