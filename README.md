# 🔭 EOSC Data Commons Conversational Search

An API to search data from various open access data publishers using natural language, implemented as a web service querying a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server, developed for the [EOSC Data Commons project](https://eosc.eu/horizon-europe-projects/eosc-data-commons/).

## 🧩 Components

- **MCP server** to search for relevant data using the EOSC Data Commons search APU based on the user question
  - Use [`rmcp`](https://github.com/modelcontextprotocol/rust-sdk) with Streamable HTTP transport

- **HTTP API** that enable easily querying the MCP server with a LLM provider from the public webapp
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

### ⚡️ Start servers

Start the **MCP server** at http://localhost:8000/mcp

```sh
cargo run -p data-commons-mcp
```

> See the [MCP server readme](/crates/data-commons-mcp) for more details on how to connect with MCP clients.

Start the **HTTP API** at http://localhost:3000/docs, alongside the MCP server:

```sh
cargo run -p data-commons-search-api
```

> Example `curl` request:
>
> ```sh
> curl -X POST http://127.0.0.1:3000/search -H "Content-Type: application/json" -H "Authorization: SECRET_KEY" -d '[{"role": "user", "content": "data about insulin in EU"}]'
> ```

### 📦 Build for production

Build binaries in `target/prod/`

```sh
cargo build --release
```

### 🐳 Deploy with Docker

Create a `.env` file with the API keys:

```sh
MISTRAL_API_KEY=YOUR_API_KEY
```

Build and deploy the 2 services:

```sh
docker compose up
```

### 🧼 Format & lint

Automatically format the codebase using `rustfmt`:

```bash
cargo fmt
```

Lint with `clippy`:

```bash
cargo clippy --all
```

### ⛓️ Check supply chain

Check the dependency supply chain: licenses (only accept dependencies with OSI or FSF approved licenses), and vulnerabilities (CVE advisories).

```bash
cargo deny check
```

Make sure dependencies have been updated:

```bash
cargo update
cargo outdated
```

