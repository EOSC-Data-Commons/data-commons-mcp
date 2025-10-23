from data_commons_mcp.main import get_relevant_tools
from data_commons_mcp.models import RankedSearchResponse


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
