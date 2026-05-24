"""
LlamaIndex Custom Retriever - Wraps AdvancedHybridRetrieval
Implements BaseRetriever interface for integration with LlamaIndex framework
"""

from typing import List, Optional
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from advanced_hybrid_retrieval import AdvancedHybridRetrieval


class HybridRetriever(BaseRetriever):
    """
    LlamaIndex Custom Retriever
    Wraps AdvancedHybridRetrieval, implements BaseRetriever interface
    """

    def __init__(
        self,
        hybrid_retrieval: AdvancedHybridRetrieval,
        **kwargs
    ):
        """
        Initialize HybridRetriever

        Args:
            hybrid_retrieval: AdvancedHybridRetrieval instance
        """
        super().__init__(**kwargs)
        self.hybrid_retrieval = hybrid_retrieval

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Execute retrieval

        Args:
            query_bundle: LlamaIndex query bundle

        Returns:
            List of retrieved result nodes
        """
        query_str = query_bundle.query_str

        # Call hybrid retrieval system
        results = self.hybrid_retrieval.retrieve(
            query=query_str,
            return_scores=True
        )

        # Convert to LlamaIndex NodeWithScore format
        nodes = []
        for result in results:
            # Format returned from AdvancedHybridRetrieval's retrieve method
            # result contains: context_id, text, metadata, score
            from llama_index.core.schema import TextNode

            node = TextNode(
                text=result['text'],
                metadata=result.get('metadata', {})
            )

            score = result.get('score', 0.0)
            node_with_score = NodeWithScore(
                node=node,
                score=float(score)
            )
            nodes.append(node_with_score)

        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Async retrieval (optional implementation)

        Args:
            query_bundle: LlamaIndex query bundle

        Returns:
            List of retrieved result nodes
        """
        # Current implementation is synchronous, async version calls sync method directly
        # For true async, implement async methods in AdvancedHybridRetrieval
        return self._retrieve(query_bundle)
