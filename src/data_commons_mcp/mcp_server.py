import argparse
import asyncio
from typing import Any
from urllib.parse import quote, unquote, urlparse

import httpx
from fastembed import TextEmbedding
from mcp.server.fastmcp import FastMCP
from opensearchpy import OpenSearch

from data_commons_mcp.config import settings
from data_commons_mcp.models import (
    FileMetrixExtensionsResponse,
    FileMetrixFilesResponse,
    OpenSearchResults,
    SearchHit,
    ToolRegistryTool,
)
from data_commons_mcp.utils import logger

# Create MCP server https://github.com/modelcontextprotocol/python-sdk
mcp = FastMCP(
    name="EOSC Data Commons MCP",
    debug=settings.debug_enabled,
    dependencies=["mcp", "httpx", "opensearch-py", "fastembed", "pydantic"],
    instructions="Provide tools that helps users access data from various open-access data publishers, developed for the EOSC Data Commons project.",
    json_response=True,
    # stateless_http=True,
)

FILEMETRIX_API = "https://filemetrix.labs.dansdemo.nl/api/v1"
TOOL_REGISTRY_API = "https://tool-registry.labs.dansdemo.nl/tools"

embedding_model = TextEmbedding(settings.embedding_model)
opensearch_client = OpenSearch(hosts=[settings.opensearch_url])


@mcp.tool()
async def search_data(
    search_input: str, start_date: str | None = None, end_date: str | None = None
) -> OpenSearchResults:
    """Search for data relevant to the user question.

    Args:
        question: Natural language question
        start_date: Optional start date in yyyy-MM-dd
        end_date: Optional end date in yyyy-MM-dd

    Returns:
        Results from OpenSearch (total_found, hits[])
    """
    # Generate embedding for the query
    # embedding = next(iter(embedding_model.embed([f"passage: {question}"])))
    embedding = next(iter(embedding_model.embed([search_input])))

    # Define filters
    filters = [
        # TODO: latest indexing does not seems to include resourceTypeGeneral field
        # {
        #     "nested": {
        #         "path": "types",
        #         "query": {"term": {"types.resourceTypeGeneral": "Dataset"}},
        #     }
        # }
    ]

    if start_date or end_date:
        date_range = {"format": "yyyy-MM-dd"}
        if start_date:
            date_range["gte"] = start_date
        if end_date:
            date_range["lte"] = end_date
        filters.append(
            {
                "nested": {
                    "path": "dates",
                    "query": {"range": {"dates.date": date_range}},
                }
            }
        )

    emb: dict[str, Any] = {
        "vector": embedding,
        "k": settings.opensearch_results_count,
    }
    if filters:
        emb["filter"] = {"bool": {"must": filters}}
    body = {
        "_source": [
            "titles",
            "subjects",
            "descriptions",
            "url",
            "doi",
            "dates",
            "publicationYear",
            "creators",
        ],
        "query": {
            "knn": {
                "emb": emb,
            }
        },
    }
    try:
        resp = opensearch_client.search(index=settings.opensearch_index, body=body)
    except Exception as e:
        raise Exception(f"OpenSearch query failed: {e}") from e
    # print(f"OpenSearch response: {json.dumps(resp, indent=2)}")
    # Extract hits from OpenSearch response
    res = OpenSearchResults(
        total_found=int(resp.get("hits", {}).get("total", {}).get("value", 0)),
        hits=[SearchHit(**hit) for hit in resp.get("hits", {}).get("hits", [])],
    )
    await get_relevant_tools(res)
    # print(f"Processed OpenSearch results: {res}")
    return res


@mcp.tool()
async def get_dataset_files(dataset_doi: str) -> FileMetrixFilesResponse:
    """Get metadata for the files in a dataset (name, description, type, dates).

    Args:
        dataset_doi: DOI of the dataset

    Returns:
        Search results with a single dataset matching the DOI
    """
    # https://filemetrix.labs.dansdemo.nl/api/v1/10.17026%2FSS%2FR5XWCC
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{FILEMETRIX_API}/{quote(dataset_doi, safe='')}",
            headers={"accept": "application/json"},
        )
        if resp.status_code == 200:
            return FileMetrixFilesResponse.model_validate(resp.json())
    return FileMetrixFilesResponse(files=[])


@mcp.tool()
async def search_tools(question: str) -> OpenSearchResults:
    """Search for tools relevant to the user question

    Args:
        question: Natural language question

    Returns:
        Search results with a list of tools and services relevant to the question
    """
    search_results = {
        "total_found": 1,
        "hits": [
            {
                "_id": "https://jupyter.org/",
                "_score": 0.8,
                "_source": {
                    "titles": [{"title": "JupyterLab", "lang": "en"}],
                    "descriptions": [{"description": "Notebooks", "lang": "en"}],
                    "url": "https://jupyter.org/",
                    "doi": None,
                    "dates": [{"date": "2016-08-29", "dateType": "Issued"}],
                    "publicationYear": "2016",
                    "creators": [{"creatorName": "Lastname, Firstname"}],
                },
            }
        ],
    }
    return OpenSearchResults.model_validate(search_results)


@mcp.tool()
async def search_citations(items_id: list[str]) -> OpenSearchResults:
    """Search for citations relevant to datasets and/or tools by DOI or URL

    Args:
        items_id: List of DOIs or URLs of datasets/tools

    Returns:
        Search results with a list of citations relevant to the request
    """
    search_results = {
        "total_found": 1,
        "hits": [
            {
                "_id": "https://doi.org/10.1109/MSR.2019.00077",
                "_score": 0.8,
                "_source": {
                    "titles": [
                        {
                            "title": "A Large-Scale Study About Quality and Reproducibility of Jupyter Notebooks",
                            "lang": "en",
                        }
                    ],
                    "descriptions": [
                        {
                            "description": "Jupyter Notebooks have been widely adopted by many different communities, both in science and industry. They support the creation of literate programming documents that combine code, text, and execution results with visualizations and all sorts of rich media. The self-documenting aspects and the ability to reproduce results have been touted as significant benefits of notebooks. At the same time, there has been growing criticism that the way notebooks are being used leads to unexpected behavior, encourage poor coding practices, and that their results can be hard to reproduce. To understand good and bad practices used in the development of real notebooks, we studied 1.4 million notebooks from GitHub. We present a detailed analysis of their characteristics that impact reproducibility. We also propose a set of best practices that can improve the rate of reproducibility and discuss open challenges that require further research and development.",
                            "lang": "en",
                        }
                    ],
                    "url": "https://doi.org/10.1109/MSR.2019.00077",
                    "doi": "10.1109/MSR.2019.00077",
                    "dates": [{"date": "2019-08-29", "dateType": "Issued"}],
                    "publicationYear": "2019 ",
                    "creators": [{"creatorName": "Lastname, Firstname"}],
                },
            }
        ],
    }
    return OpenSearchResults.model_validate(search_results)


# In OpenSearch and Filemetrix: https://doi.org/10.17026/DANS-2B8-ZGY2
# Data to Monitor Soil Aggregate Breakdown
# Data on fair evaluation


# https://confluence.egi.eu/display/EOSCDATACOMMONS/API+Definitions+and+Implementation+Guidelines
# https://dev.matchmaker.eosc-data-commons.eu/search?q=search for data about Cognitive load in cyclists while navigating in traffic&model=einfracz%2Fqwen3-coder
# curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -H "Authorization: SECRET_KEY" -d '{"messages": [{"role": "user", "content": "Datasets about representation of dogs in medieval time"}], "model": "einfracz/qwen3-coder", "stream": true}'
# curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -H "Authorization: SECRET_KEY" -d '{"messages": [{"role": "user", "content": "search for data about Harelbeke Evolis"}], "model": "einfracz/qwen3-coder", "stream": true}'
# curl -X POST http://localhost:8001/chat -H "Content-Type: application/json" -H "Authorization: SECRET_KEY" -d '{"messages": [{"role": "user", "content": "search for data about Cognitive load in cyclists while navigating in traffic"}], "model": "einfracz/qwen3-coder", "stream": true}'
async def get_relevant_tools(search_results: OpenSearchResults) -> None:
    """Fetch file extensions and relevant tools from the FileMetrix API in parallel for each hit's DOI,
    and update hits in-place.

    Args:
        search_results: The OpenSearch results to enhance with file extensions and relevant tools.
    """

    async def fetch_extensions(client: httpx.AsyncClient, doi: str) -> FileMetrixExtensionsResponse | None:
        """Fetch extensions for a single DOI."""
        try:
            encoded = quote(doi, safe="")
            resp = await client.get(
                f"{FILEMETRIX_API}/extensions/{encoded}",
                headers={"accept": "application/json"},
            )
            if resp.status_code == 200:
                return FileMetrixExtensionsResponse.model_validate(resp.json())
            logger.warning(f"FileMetrix returned {resp.status_code} for DOI {doi}")
        except Exception as e:
            logger.warning(f"FileMetrix fetch error for {doi}: {e}")
        return None

    async def fetch_tools_for_extension(client: httpx.AsyncClient, extension: str) -> list[dict[str, str]] | None:
        """Fetch relevant tools for a file extension from the tool registry."""
        try:
            resp = await client.get(
                f"{TOOL_REGISTRY_API}/input/{extension}",
                headers={"accept": "application/json"},
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(f"Tool registry returned {resp.status_code} for extension {extension}")
        except Exception as e:
            logger.warning(f"Tool registry fetch error for {extension}: {e}")
        return None

    # Extract DOI from hit and create fetch task
    async def process_hit(client: httpx.AsyncClient, hit: SearchHit) -> None:
        """Extract DOI from hit and fetch/apply extensions and relevant tools."""
        doi = None
        try:
            if hit.id.startswith("http"):
                parsed = urlparse(hit.id)
                if "doi.org" in parsed.netloc:
                    doi = unquote(parsed.path.lstrip("/"))
            else:
                doi = hit.id
        except Exception:
            return
        if not doi:
            return

        # Fetch file extensions
        fm = await fetch_extensions(client, doi)
        if fm:
            hit.file_extensions = fm.extensions
            logger.info(f"📁 https://doi.org/{doi} -> extensions: {fm.extensions}")

            # Fetch relevant tools for each extension
            all_tools = []
            for ext in fm.extensions:
                tools_data = await fetch_tools_for_extension(client, ext)
                if tools_data:
                    try:
                        for tool_dict in tools_data:
                            tool = ToolRegistryTool.model_validate(tool_dict)
                            all_tools.append(tool)
                            logger.info(f"🔧 {ext} -> tool: {tool.tool_label}")
                    except Exception as e:
                        logger.warning(f"Error parsing tool data for {ext}: {e}")

            # Remove duplicates by tool_uri while preserving order
            seen = set()
            unique_tools = []
            for tool in all_tools:
                if tool.tool_uri not in seen:
                    seen.add(tool.tool_uri)
                    unique_tools.append(tool)

            hit.relevant_tools = unique_tools

    async with httpx.AsyncClient(timeout=10.0) as client:
        await asyncio.gather(*(process_hit(client, hit) for hit in search_results.hits))


def cli() -> None:
    """Run the MCP server with appropriate transport."""
    parser = argparse.ArgumentParser(
        description="A Model Context Protocol (MCP) server for BioData resources at the SIB."
    )
    parser.add_argument("--http", action="store_true", help="Use Streamable HTTP transport")
    parser.add_argument("--port", type=int, default=8888, help="Port to run the server on")
    # parser.add_argument("settings_filepath", type=str, nargs="?", default="sparql-mcp.json", help="Path to settings file")
    args = parser.parse_args()
    # settings = Settings.from_file(args.settings_filepath)
    if args.http:
        mcp.run()
        mcp.settings.port = args.port
        mcp.settings.log_level = "INFO"
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
