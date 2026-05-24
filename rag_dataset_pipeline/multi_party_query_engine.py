"""
Multi-party query engine with multi-source retrieval and LLM generation.
"""

from typing import List, Dict, Any, Optional
import httpx
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor


class RemoteRetriever:
    """
    Remote Retriever - calls remote retrieval service via HTTP API.
    Implements the same interface as local Retriever.
    """

    def __init__(self, api_url: str, party_id: str, timeout: float = 120.0):
        """
        Initialize RemoteRetriever.

        Args:
            api_url: Remote API URL (e.g., "http://localhost:8001")
            party_id: Party ID
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.party_id = party_id
        self.timeout = timeout

    async def retrieve(self, query: str, top_k: int = 5, expand_context: bool = True) -> List[NodeWithScore]:
        """
        Async retrieval.

        Args:
            query: Query text
            top_k: Number of documents to return
            expand_context: Whether to expand neighboring chunks

        Returns:
            List of retrieved nodes
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.post(
                    f"{self.api_url}/retrieve",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "return_scores": True,
                        "expand_context": expand_context
                    }
                )
            response.raise_for_status()
            data = response.json()

            # Convert to NodeWithScore format
            nodes = []
            for result in data.get("results", []):
                node = TextNode(
                    text=result["text"],
                    metadata={
                        **result.get("metadata", {}),
                        "party_id": self.party_id,
                        "source": "remote"
                    }
                )
                node_with_score = NodeWithScore(
                    node=node,
                    score=result.get("score", 0.0)
                )
                nodes.append(node_with_score)

            return nodes
        except Exception as e:
            print(f"[{self.party_id}] Error retrieving: {e}")
            return []

    def retrieve_sync(self, query: str, top_k: int = 5, expand_context: bool = True) -> List[NodeWithScore]:
        """
        Sync retrieval (for compatibility).

        Args:
            query: Query text
            top_k: Number of documents to return
            expand_context: Whether to expand neighboring chunks

        Returns:
            List of retrieved nodes
        """
        return asyncio.run(self.retrieve(query, top_k, expand_context))

    async def health_check(self) -> bool:
        """Check remote service health status."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=False) as client:
                response = await client.get(f"{self.api_url}/health")
            return response.status_code == 200
        except:
            return False


class MultiPartyQueryEngine(BaseQueryEngine):
    """
    Multi-party query engine.
    Supports retrieval from multiple local and remote data sources, with LLM-generated answers.
    """

    def __init__(
        self,
        retrievers: Dict[str, Any],  # Dict[party_id, Retriever]
        llm: Optional[OpenAI] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        top_k_per_party: int = 5,
        max_total_results: int = 15,
        expand_context: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs
    ):
        """
        Initialize multi-party query engine.

        Args:
            retrievers: Retriever dict, key is party_id, value is Retriever instance
            llm: LLM instance (if provided)
            llm_base_url: LLM API base URL (OpenAI protocol)
            llm_api_key: LLM API key
            top_k_per_party: Number of documents per party
            max_total_results: Max documents after merging
            expand_context: Whether to expand neighboring chunks
        """
        super().__init__(callback_manager=callback_manager or CallbackManager([]), **kwargs)
        self.retrievers = retrievers
        self.top_k_per_party = top_k_per_party
        self.max_total_results = max_total_results
        self.expand_context = expand_context

        # Initialize LLM
        if llm is None:
            if llm_base_url and llm_api_key:
                self.llm = OpenAI(
                    api_base=llm_base_url,
                    api_key=llm_api_key,
                    model="gpt-3.5-turbo",
                )
            else:
                self.llm = None
                print("Warning: No LLM configured. Set llm_base_url and llm_api_key later.")
        else:
            self.llm = llm

    def set_llm(self, llm_base_url: str, llm_api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Set LLM configuration.

        Args:
            llm_base_url: LLM API base URL
            llm_api_key: LLM API key
            model: Model name
        """
        self.llm = OpenAI(
            api_base=llm_base_url,
            api_key=llm_api_key,
            model=model,
        )

    async def _retrieve_all(self, query: str) -> List[NodeWithScore]:
        """
        Retrieve from all parties.

        Args:
            query: Query text

        Returns:
            Merged retrieval results
        """
        async def retrieve_task(party_id: str, retriever: Any) -> tuple:
            """Single retrieval task."""
            try:
                if isinstance(retriever, RemoteRetriever):
                    nodes = await retriever.retrieve(query, self.top_k_per_party, self.expand_context)
                else:
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor() as executor:
                        nodes = await loop.run_in_executor(
                            executor,
                            lambda: retriever._retrieve(QueryBundle(query_str=query))
                        )
                return (party_id, nodes)
            except Exception as e:
                print(f"[{party_id}] Error during retrieval: {e}")
                return (party_id, [])

        # Execute all retrieval tasks in parallel
        tasks = [
            asyncio.create_task(retrieve_task(party_id, retriever))
            for party_id, retriever in self.retrievers.items()
        ]

        # Wait for all tasks with timeout
        try:
            done, pending = await asyncio.wait(tasks, timeout=60.0)  # Reduced from 180s to 60s

            # Cancel pending tasks
            for task in pending:
                task.cancel()

            # Collect results
            task_results = []
            for task in tasks:
                try:
                    if task in done:
                        result = task.result()
                        task_results.append(result)
                    else:
                        # Timed out tasks return empty results
                        task_results.append((None, []))
                except Exception as e:
                    print(f"Task failed with exception: {e}")
                    task_results.append((None, []))

            if pending:
                print(f"Multi-party retrieval timed out after 180 seconds. {len(pending)} tasks were cancelled, {len(done)} completed.")

        except Exception as e:
            print(f"Error during multi-party retrieval: {e}")
            task_results = []

        # Collect results
        results = {}
        for result in task_results:
            if result and len(result) == 2 and result[0] is not None:
                party_id, nodes = result
                results[party_id] = nodes

        # Merge results and sort by score
        all_nodes = []
        for party_id, nodes in results.items():
            all_nodes.extend(nodes)

        # Deduplicate by chunk_id (fallback to text content)
        seen_ids = set()
        unique_nodes = []
        for node in sorted(all_nodes, key=lambda x: x.score, reverse=True):
            metadata = getattr(node.node, 'metadata', {}) or {}
            dedupe_id = metadata.get('chunk_id') or metadata.get('context_id') or node.node.text
            if dedupe_id not in seen_ids:
                seen_ids.add(dedupe_id)
                unique_nodes.append(node)

        # Return top-k
        return unique_nodes[:self.max_total_results]

    def _retrieve_all_sync(self, query: str) -> List[NodeWithScore]:
        """Sync version of retrieval."""
        return asyncio.run(self._retrieve_all(query))

    def _generate_response(self, query: str, nodes: List[NodeWithScore]) -> str:
        """
        Generate response using LLM.

        Args:
            query: Query text
            nodes: List of retrieved nodes

        Returns:
            LLM-generated answer
        """
        return self._format_results_summary(query, nodes)

    def _format_results_summary(self, query: str, nodes: List[NodeWithScore]) -> str:
        """
        Format retrieval results summary (used when no LLM available).

        Args:
            query: Query text
            nodes: List of retrieved nodes

        Returns:
            Formatted summary
        """
        if not nodes:
            return f"No relevant documents found for query: {query}"

        summary = f"Found {len(nodes)} relevant documents for query: {query}\n\n"
        for i, node in enumerate(nodes[:5], 1):
            summary += f"[Document {i}] (Score: {node.score:.4f})\n"
            summary += f"{node.node.text[:200]}...\n\n"

        return summary

    def _query(self, query_bundle: QueryBundle) -> Response:
        """
        Execute query.

        Args:
            query_bundle: Query bundle

        Returns:
            Query response
        """
        query_str = query_bundle.query_str

        # Retrieve
        nodes = self._retrieve_all_sync(query_str)

        # Generate answer
        answer = self._generate_response(query_str, nodes)

        # Build response
        response = Response(
            response=answer,
            source_nodes=nodes,
        )

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        """
        Async query.

        Args:
            query_bundle: Query bundle

        Returns:
            Query response
        """
        query_str = query_bundle.query_str

        # Async retrieval
        nodes = await self._retrieve_all(query_str)

        # Generate answer (LLM call may need async handling)
        if self.llm:
            # If LLM supports async, can use it here
            answer = self._generate_response(query_str, nodes)
        else:
            answer = self._generate_response(query_str, nodes)

        # Build response
        response = Response(
            response=answer,
            source_nodes=nodes,
        )

        return response

    # Compatible with new BaseQueryEngine abstract interface (prompt modules)
    def _get_prompt_modules(self):
        # This engine constructs prompts internally, does not rely on pluggable prompt modules
        return {}


def create_multi_party_engine(
    party_configs: List[Dict[str, Any]],
    llm_base_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    **kwargs
) -> MultiPartyQueryEngine:
    """
    Convenience function to create multi-party query engine.

    Args:
        party_configs: List of party configurations, each containing:
            - party_id: Party ID
            - type: "local" or "remote"
            - retriever: Retriever instance (local) or api_url (remote)
        llm_base_url: LLM API base URL
        llm_api_key: LLM API key
        **kwargs: Additional arguments passed to MultiPartyQueryEngine

    Returns:
        MultiPartyQueryEngine instance
    """
    retrievers = {}

    for config in party_configs:
        party_id = config["party_id"]
        party_type = config.get("type", "local")

        if party_type == "local":
            # Local Retriever
            retrievers[party_id] = config["retriever"]
        elif party_type == "remote":
            # Remote Retriever
            api_url = config["api_url"]
            retrievers[party_id] = RemoteRetriever(
                api_url=api_url,
                party_id=party_id
            )
        else:
            raise ValueError(f"Unknown party type: {party_type}")

    engine = MultiPartyQueryEngine(
        retrievers=retrievers,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        **kwargs
    )

    return engine
