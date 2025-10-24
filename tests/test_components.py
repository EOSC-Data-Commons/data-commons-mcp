import json
from typing import Any

import httpx
import pytest
from ag_ui.core import RunStartedEvent
from starlette.testclient import TestClient

from data_commons_mcp.config import settings
from data_commons_mcp.main import app, get_relevant_tools
from data_commons_mcp.models import RankedSearchResponse


def opensearch_is_available() -> bool:
    """Lightweight check whether an OpenSearch URL is reachable."""
    try:
        r = httpx.get(settings.opensearch_url.rstrip("/") + "/_cluster/health", timeout=1.0)
        return 200 <= r.status_code < 400
    except Exception:
        return False


@pytest.mark.skipif(
    not opensearch_is_available(),
    reason=f"OpenSearch unreachable at {settings.opensearch_url}",
)
def test_app() -> None:
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200

    # Test chat call streaming endpoint
    payload = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Educational datasets from Switzerland covering student assessments, language competencies, "
                    "and learning outcomes, including experimental or longitudinal studies on pupils or students."
                ),
            }
        ],
        # "model": "einfracz/qwen3-coder",
        "model": "einfracz/gpt-oss-120b",
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer SECRET_KEY"}

    with client.stream("POST", "/chat", headers=headers, json=payload) as resp:
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("text/event-stream")
        events = process_stream(resp)

    # print(events)
    assert len(events) >= 1
    # Ensure a `RunStartedEvent` is received first
    assert RunStartedEvent.model_validate(events[0])

    # NOTE: The resp.text contains all SSE lines as a single string
    # resp = client.post("/chat", headers=headers, json=payload)
    # # Check chat endpoint returns 200 with text/event-stream
    # assert resp.status_code == 200
    # assert resp.headers.get("content-type", "").startswith("text/event-stream")
    # print("resp.text:", resp.text)
    # first_line = next((line for line in resp.text.splitlines() if line.startswith("data:")), None)
    # parsed = json.loads(first_line.split("data:", 1)[1].strip())
    # assert RunStartedEvent.model_validate(parsed)


async def test_get_relevant_tools() -> None:
    """Test reranking dummy search results."""
    dummy_search_res = RankedSearchResponse.model_validate(
        {
            "summary": "Found 1 dataset related to cognitive load in cyclists.",
            "hits": [
                {
                    "id": "https://doi.org/10.17026/PT/PIYWW5",
                    "source": {
                        "doi": "10.17026/PT/PIYWW5",
                        "url": None,
                        "titles": [
                            {
                                "title": "Replication Data for: Cognitive load in cyclists while navigating in traffic: Effects of static and dynamic route events on neural activity of cyclists measured by fNIRS"
                            }
                        ],
                        "descriptions": [
                            {
                                "description": "Neural activity data collected during a real-life field experiment by a non-invasive portable method, namely Functional Near-Infrared Spectroscopy (fNIRS), sensitive to neural activity in the prefrontal cortex region."
                            },
                        ],
                        "publicationYear": "2025",
                        "publicationDate": None,
                        "subjects": [{"subject": "Engineering"}],
                        "creators": [{"creatorName": "Nidegger, Christian"}],
                        "resourceType": "dataset",
                    },
                    "opensearch_score": 0.91528225,
                    "score": None,
                    # "file_extensions": [],
                },
            ],
        }
    )
    # search_res = RankedSearchResponse.model_validate(dummy_search_res)
    await get_relevant_tools(dummy_search_res)
    # print(docs)
    assert len(dummy_search_res.hits) >= 1
    for hit in dummy_search_res.hits:
        assert hit.file_extensions and len(hit.file_extensions) >= 1


def process_stream(resp: Any) -> list[Any]:
    """Read an SSE streaming and return a list of parsed JSON objects from `data:` lines.

    Returns:
        List of parsed JSON objects (only events where the joined `data:` value parsed as JSON).
    """
    parsed_events: list[Any] = []
    current_data_lines: list[str] = []
    for raw_line in resp.iter_lines():
        if raw_line is None:
            continue
        # Normalize bytes -> str
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else raw_line
        # Strip trailing CR/LF but preserve leading spaces for data content handling
        line = line.rstrip("\r\n")
        # Ignore keep-alive comments
        if line.lstrip().startswith(":"):
            continue
        # End of event when line is empty
        if line.strip() == "":
            if current_data_lines:
                # Join multi-line data fields with newline as per SSE spec
                joined = "\n".join(dl.split("data:", 1)[1].lstrip() for dl in current_data_lines)
                try:
                    parsed = json.loads(joined)
                except json.JSONDecodeError:
                    # Skip non-JSON data events
                    current_data_lines = []
                    continue
                parsed_events.append(parsed)
                current_data_lines = []
            continue
        # Collect only data: lines
        stripped = line.lstrip()
        if stripped.startswith("data:"):
            current_data_lines.append(stripped)
        else:
            continue
    # Handle case where stream ended without a trailing blank line
    if current_data_lines:
        joined = "\n".join(dl.split("data:", 1)[1].lstrip() for dl in current_data_lines)
        try:
            parsed = json.loads(joined)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            parsed_events.append(parsed)
    return parsed_events
