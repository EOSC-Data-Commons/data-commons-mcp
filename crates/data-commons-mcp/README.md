# 🔭 EOSC Data Commons MCP server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server to access data from various open access data publishers, developed for the [EOSC Data Commons project](https://eosc.eu/horizon-europe-projects/eosc-data-commons/).

## 🪄 Available tools

- 🕵 Search for data relevant to the user question
  - Arguments:
    - `question` (string): the user's question

## 🔌 Connect client

Follow the instructions of your client, and use the `/mcp` URL of your deployed server (e.g. http://127.0.0.1:8000/mcp)

### 🐙 VSCode GitHub Copilot

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
