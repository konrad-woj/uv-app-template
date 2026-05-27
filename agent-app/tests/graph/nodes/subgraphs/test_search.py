"""Tests for search subgraph: fan-out/fan-in via Send API."""

from langchain_core.runnables import RunnableConfig

from app.graph.nodes.subgraphs.search import SearchState, search_subgraph

_CONFIG: RunnableConfig = {"configurable": {"thread_id": "search-test"}}


class TestSearchSubgraph:
    async def test_single_query_produces_one_result(self) -> None:
        state: SearchState = {"queries": ["python asyncio"], "query": "", "results": []}
        result = await search_subgraph.ainvoke(state, _CONFIG)
        assert len(result["results"]) == 1
        assert "python asyncio" in result["results"][0]

    async def test_multiple_queries_fan_out_and_accumulate(self) -> None:
        queries = ["topic A", "topic B", "topic C"]
        state: SearchState = {"queries": queries, "query": "", "results": []}
        result = await search_subgraph.ainvoke(state, _CONFIG)
        assert len(result["results"]) == len(queries)

    async def test_each_query_appears_in_results(self) -> None:
        queries = ["alpha", "beta"]
        state: SearchState = {"queries": queries, "query": "", "results": []}
        result = await search_subgraph.ainvoke(state, _CONFIG)
        all_results = " ".join(result["results"])
        assert "alpha" in all_results
        assert "beta" in all_results

    async def test_empty_queries_produces_no_results(self) -> None:
        state: SearchState = {"queries": [], "query": "", "results": []}
        result = await search_subgraph.ainvoke(state, _CONFIG)
        assert result["results"] == []
