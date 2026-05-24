"""
Advanced Hybrid Retrieval System.
Supports three retrieval methods: Dense (semantic similarity), Sparse (lexical matching), ColBERT (fine-grained matching).
Uses BGE-M3 model and bge-reranker-v2-m3 reranker.
"""

import json
import hashlib
import pickle
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import os
import time
import gc
import torch

try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
except ImportError:
    print("Warning: FlagEmbedding not installed. Please install it with: pip install -U FlagEmbedding")
    BGEM3FlagModel = None
    FlagReranker = None

from llama_index.core.schema import NodeWithScore, TextNode

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
BGE_M3_PATH = str(DEFAULT_ROOT / "bge-m3")
BGE_RERANKER_PATH = str(DEFAULT_ROOT / "bge-reranker-v2-m3")


class AdvancedHybridRetrieval:
    """Advanced hybrid retrieval system supporting Dense, Sparse, and ColBERT retrieval methods."""
    
    def __init__(
        self,
        contexts_path: str = "contexts.jsonl",
        bge_m3_path: str = None,
        reranker_path: str = None,
        dense_top_k: int = 10,
        sparse_top_k: int = 10,
        colbert_top_k: int = 10,
        rerank_top_n: int = 5,
        dense_weight: float = 0.35,
        sparse_weight: float = 0.25,
        colbert_weight: float = 0.4,
        use_fp16: bool = True,
        max_length: int = 8192,
        use_cache: bool = True,
        cache_dir: str = ".cache",
        cache_key: Optional[str] = None,
        batch_size: int = 8,
        devices: Optional[str] = None,
        expand_context: bool = True,
        expand_window: int = 2,
    ):
        """
        Initialize the advanced hybrid retrieval system.

        Args:
            contexts_path: Path to the contexts file.
            bge_m3_path: Path to the BGE-M3 model (local).
            reranker_path: Path to the reranker model (local).
            dense_top_k: Top-k for dense retrieval.
            sparse_top_k: Top-k for sparse retrieval.
            colbert_top_k: Top-k for ColBERT retrieval.
            rerank_top_n: Number of documents to return after reranking.
            dense_weight: Weight for dense retrieval.
            sparse_weight: Weight for sparse retrieval.
            colbert_weight: Weight for ColBERT retrieval.
            use_fp16: Whether to use FP16 acceleration.
            max_length: Maximum sequence length.
            expand_context: Whether to expand neighboring chunks.
            expand_window: Expansion window size (chunks to expand before and after).
        """
        if BGEM3FlagModel is None:
            raise ImportError("FlagEmbedding is required. Install with: pip install -U FlagEmbedding")
        
        self.contexts_path = contexts_path
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.colbert_top_k = colbert_top_k
        self.rerank_top_n = rerank_top_n
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.colbert_weight = colbert_weight
        self.max_length = max_length
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_key = cache_key
        self.batch_size = batch_size
        self.devices = devices
        self.expand_context = expand_context
        self.expand_window = expand_window

        # Set model paths
        if bge_m3_path is None:
            if os.path.exists(BGE_M3_PATH):
                bge_m3_path = BGE_M3_PATH
            else:
                bge_m3_path = os.environ.get("RETRIEVAL_EMBEDDING_MODEL", "BAAI/bge-m3")
        
        if reranker_path is None:
            if os.path.exists(BGE_RERANKER_PATH):
                reranker_path = BGE_RERANKER_PATH
            else:
                reranker_path = os.environ.get("RETRIEVAL_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        
        # Load BGE-M3 model
        print(f"Loading BGE-M3 model from: {bge_m3_path}")
        bge_kwargs = {
            "use_fp16": use_fp16,
        }
        if self.devices is not None:
            bge_kwargs["devices"] = self.devices
        self.bge_model = BGEM3FlagModel(
            bge_m3_path,
            **bge_kwargs
        )
        print("BGE-M3 model loaded successfully!")
        
        # Load reranker
        print(f"Loading reranker from: {reranker_path}")
        reranker_kwargs = {
            "use_fp16": use_fp16,
        }
        # Always set devices parameter, including None
        reranker_kwargs["devices"] = self.devices
        self.reranker = FlagReranker(
            reranker_path,
            **reranker_kwargs
        )
        print("Reranker loaded successfully!")
        
        # Load context data
        self.contexts = self._load_contexts()
        self.documents = self._create_documents()
        print(f"Loaded {len(self.contexts)} contexts")

        # Build uid to chunks index (for neighboring chunk expansion)
        self.uid_to_chunks = self._build_uid_index()

        # Pre-compute or load cached document embeddings (Dense, Sparse, ColBERT)
        if self.use_cache and self._try_load_cache(
            contexts_path=self.contexts_path,
            bge_m3_path=bge_m3_path,
            reranker_path=reranker_path,
            max_length=self.max_length,
        ):
            print("Loaded document embeddings from cache.")
        else:
            print("\nPre-computing document embeddings...")
            self._precompute_document_embeddings()
            print("Document embeddings computed successfully!")
            if self.use_cache:
                self._save_cache(
                    contexts_path=self.contexts_path,
                    bge_m3_path=bge_m3_path,
                    reranker_path=reranker_path,
                    max_length=self.max_length,
                )
        
    def _load_contexts(self) -> List[Dict[str, Any]]:
        """Load context data."""
        contexts = []
        with open(self.contexts_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    contexts.append(json.loads(line))
        return contexts
    
    def _create_documents(self) -> List[Dict[str, Any]]:
        """Create document object list."""
        documents = []
        for ctx in self.contexts:
            # Compatible with multiple input formats
            text = ctx.get('content') or ctx.get('text') or ctx.get('context') or ''

            # Determine context_id (priority: chunk_id / context_id / uid)
            context_id = ctx.get('chunk_id') or ctx.get('context_id') or ctx.get('uid') or 'unknown'

            # Build metadata field
            md = ctx.get('metadata') or {}
            source_line = ctx.get('source_line')
            if not source_line:
                src_file = md.get('source_file') or md.get('source') or None
                if src_file:
                    source_line = f"{src_file}:{md.get('chunk_index', 0)}"

            relevant = ctx.get('relevant') or md.get('relevant') or 'unknown'
            private = ctx.get('private') or md.get('private') or 'unknown'
            length = ctx.get('length') or md.get('length') or (len(text) if text else 'unknown')
            chunk_id_val = ctx.get('chunk_id') or md.get('chunk_id') or md.get('context_id') or f"{context_id}"

            doc = {
                'text': text,
                'context_id': str(context_id),
                'metadata': {
                    **md,
                    'chunk_id': chunk_id_val,
                    'context_id': str(context_id),
                    'source_line': source_line,
                    'length': length,
                    'uid': ctx.get('uid') or md.get('uid'),
                    'document_type': ctx.get('document_type') or md.get('document_type'),
                    'source_file': ctx.get('source_file') or md.get('source_file'),
                    'sample_index': ctx.get('sample_index') or md.get('sample_index'),
                    'chunk_index': ctx.get('chunk_index') or md.get('chunk_index'),
                    'global_context_index': ctx.get('global_context_index') or md.get('global_context_index'),
                }
            }
            documents.append(doc)
        return documents

    def _build_uid_index(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        Build uid to chunks index.

        Returns:
            Dict[uid, List[(doc_index, chunk_index)]]
        """
        uid_index = {}
        for doc_idx, doc in enumerate(self.documents):
            chunk_id = doc['metadata'].get('chunk_id', '')
            # Parse chunk_id format: {uid}_chunk_{chunk_index}
            if '_chunk_' in chunk_id:
                parts = chunk_id.rsplit('_chunk_', 1)
                if len(parts) == 2:
                    uid = parts[0]
                    try:
                        chunk_idx = int(parts[1])
                        if uid not in uid_index:
                            uid_index[uid] = []
                        uid_index[uid].append((doc_idx, chunk_idx))
                    except ValueError:
                        pass

        # Sort by chunk_index
        for uid in uid_index:
            uid_index[uid].sort(key=lambda x: x[1])

        print(f"Built UID index with {len(uid_index)} unique documents")
        return uid_index

    def _expand_chunks(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Expand retrieval results with neighboring chunks.

        Args:
            results: List of retrieval results.

        Returns:
            Expanded results list.
        """
        if not self.expand_context or self.expand_window <= 0:
            return results

        expanded_results = []
        seen_chunk_ids = set()

        for result in results:
            chunk_id = result.get('chunk_id', '')

            # Add original result
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                expanded_results.append(result)

            # Parse chunk_id
            if '_chunk_' not in chunk_id:
                continue

            parts = chunk_id.rsplit('_chunk_', 1)
            if len(parts) != 2:
                continue

            uid = parts[0]
            try:
                current_chunk_idx = int(parts[1])
            except ValueError:
                continue

            # Find neighboring chunks in the same document
            if uid not in self.uid_to_chunks:
                continue

            doc_chunks = self.uid_to_chunks[uid]

            # Find the position of current chunk in the list
            current_pos = None
            for i, (doc_idx, chunk_idx) in enumerate(doc_chunks):
                if chunk_idx == current_chunk_idx:
                    current_pos = i
                    break

            if current_pos is None:
                continue

            # Expand neighboring chunks
            for offset in range(-self.expand_window, self.expand_window + 1):
                if offset == 0:
                    continue  # Skip current chunk

                neighbor_pos = current_pos + offset
                if 0 <= neighbor_pos < len(doc_chunks):
                    neighbor_doc_idx, neighbor_chunk_idx = doc_chunks[neighbor_pos]
                    neighbor_chunk_id = f"{uid}_chunk_{neighbor_chunk_idx}"

                    if neighbor_chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(neighbor_chunk_id)

                        # Get neighboring chunk info from documents
                        neighbor_doc = self.documents[neighbor_doc_idx]
                        neighbor_result = {
                            'chunk_id': neighbor_chunk_id,
                            'global_context_index': neighbor_doc['metadata'].get('global_context_index', 'unknown'),
                            'text': neighbor_doc['text'],
                            'metadata': neighbor_doc['metadata'],
                            'score': result.get('score', 0.0) * 0.9,  # Slightly lower score for expanded chunks
                            'is_expanded': True,  # Mark as expanded chunk
                        }
                        expanded_results.append(neighbor_result)

        # Sort by chunk_id to ensure chunks from the same document are adjacent
        def sort_key(r):
            chunk_id = r.get('chunk_id', '')
            if '_chunk_' in chunk_id:
                parts = chunk_id.rsplit('_chunk_', 1)
                try:
                    return (parts[0], int(parts[1]))
                except:
                    pass
            return (chunk_id, 0)

        expanded_results.sort(key=sort_key)

        print(f"  Expanded from {len(results)} to {len(expanded_results)} chunks")
        return expanded_results

    def _precompute_document_embeddings(self):
        """Pre-compute embeddings for all documents."""
        corpus_texts = [doc['text'] for doc in self.documents]

        # Batch encode documents (Dense, Sparse, ColBERT)
        print("  Computing Dense embeddings...")
        dense_output = self.bge_model.encode(
            corpus_texts,
            batch_size=12,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        self.dense_embeddings = dense_output['dense_vecs']
        
        print("  Computing Sparse embeddings...")
        sparse_output = self.bge_model.encode(
            corpus_texts,
            batch_size=12,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        self.sparse_embeddings = sparse_output['lexical_weights']
        
        print("  Computing ColBERT embeddings...")
        colbert_output = self.bge_model.encode(
            corpus_texts,
            batch_size=12,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        self.colbert_embeddings = colbert_output['colbert_vecs']
        
        print(f"  Dense embeddings shape: {self.dense_embeddings.shape}")
        print(f"  Sparse embeddings: {len(self.sparse_embeddings)} documents")
        print(f"  ColBERT embeddings: {len(self.colbert_embeddings)} documents")
    
    # ----------------------- Cache related -----------------------
    def _cache_base_path(
        self,
        contexts_path: str,
        bge_m3_path: str,
        reranker_path: str,
        max_length: int,
    ) -> Path:
        # Build cache key based on context file content hash + model path + parameters
        ctx_path = Path(contexts_path)
        file_sig = f"{ctx_path.resolve()}|{ctx_path.stat().st_mtime}|{ctx_path.stat().st_size}"
        with open(ctx_path, "rb") as f:
            # Only take part of the content for hashing to avoid long processing time for large files
            head = f.read(1024 * 1024)
        payload = "|".join(
            [
                self.cache_key or "",
                file_sig,
                str(hashlib.sha1(head).hexdigest()),
                str(bge_m3_path),
                str(reranker_path),
                f"maxlen={max_length}",
                f"counts={len(self.documents)}",
            ]
        )
        key = hashlib.sha1(payload.encode("utf-8")).hexdigest()
        return self.cache_dir / f"emb_{key}"
    
    def _try_load_cache(
        self,
        contexts_path: str,
        bge_m3_path: str,
        reranker_path: str,
        max_length: int,
    ) -> bool:
        try:
            base = self._cache_base_path(contexts_path, bge_m3_path, reranker_path, max_length)
            meta_path = base.with_suffix(".meta.json")
            dense_path = base.with_suffix(".dense.npy")
            sparse_path = base.with_suffix(".sparse.pkl")
            colbert_path = base.with_suffix(".colbert.pkl")
            if not (meta_path.exists() and dense_path.exists() and sparse_path.exists() and colbert_path.exists()):
                return False
            # Read metadata and perform basic consistency check
            meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
            if meta.get("num_docs") != len(self.documents):
                return False
            self.dense_embeddings = np.load(dense_path)
            with open(sparse_path, "rb") as f:
                self.sparse_embeddings = pickle.load(f)
            with open(colbert_path, "rb") as f:
                self.colbert_embeddings = pickle.load(f)
            print(f"\nCache hit. Loaded embeddings from {base.parent}")
            return True
        except Exception as e:
            print(f"Cache load failed, will recompute. Reason: {e}")
            return False
    
    def _save_cache(
        self,
        contexts_path: str,
        bge_m3_path: str,
        reranker_path: str,
        max_length: int,
    ) -> None:
        try:
            base = self._cache_base_path(contexts_path, bge_m3_path, reranker_path, max_length)
            meta_path = base.with_suffix(".meta.json")
            dense_path = base.with_suffix(".dense.npy")
            sparse_path = base.with_suffix(".sparse.pkl")
            colbert_path = base.with_suffix(".colbert.pkl")
            np.save(dense_path, self.dense_embeddings)
            with open(sparse_path, "wb") as f:
                pickle.dump(self.sparse_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            with open(colbert_path, "wb") as f:
                pickle.dump(self.colbert_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            meta = {
                "saved_at": time.time(),
                "num_docs": len(self.documents),
                "contexts_path": str(Path(contexts_path).resolve()),
                "bge_m3_path": str(bge_m3_path),
                "reranker_path": str(reranker_path),
                "max_length": max_length,
            }
            Path(meta_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Cache saved to {base.parent}")
        except Exception as e:
            print(f"Cache save failed (skipped). Reason: {e}")
    
    def _dense_retrieve(self, query: str) -> List[Tuple[int, float]]:
        """
        Dense retrieval: semantic similarity matching.

        Args:
            query: Query text.

        Returns:
            List of (doc_index, score) tuples.
        """
        # Encode query
        query_output = self.bge_model.encode(
            [query],
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_embedding = query_output['dense_vecs'][0]

        # Compute similarity scores
        scores = np.dot(self.dense_embeddings, query_embedding)

        # Get top-k
        top_k_indices = np.argsort(scores)[::-1][:self.dense_top_k]
        
        results = [(int(idx), float(scores[idx])) for idx in top_k_indices]
        return results
    
    def _sparse_retrieve(self, query: str) -> List[Tuple[int, float]]:
        """
        Sparse retrieval: lexical matching.

        Args:
            query: Query text.

        Returns:
            List of (doc_index, score) tuples.
        """
        # Encode query
        query_output = self.bge_model.encode(
            [query],
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        query_sparse = query_output['lexical_weights'][0]

        # Compute sparse score for each document
        scores = []
        for doc_sparse in self.sparse_embeddings:
            score = self.bge_model.compute_lexical_matching_score(
                query_sparse,
                doc_sparse
            )
            scores.append(score)
        
        scores = np.array(scores)

        # Get top-k
        top_k_indices = np.argsort(scores)[::-1][:self.sparse_top_k]
        
        results = [(int(idx), float(scores[idx])) for idx in top_k_indices]
        return results
    
    def _colbert_retrieve(self, query: str, candidate_indices: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """
        ColBERT retrieval: fine-grained multi-vector matching.

        Args:
            query: Query text.
            candidate_indices: List of candidate document indices. If None, compute for all documents.

        Returns:
            List of (doc_index, score) tuples.
        """
        # Encode query
        query_output = self.bge_model.encode(
            [query],
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
        )
        query_colbert = query_output['colbert_vecs'][0]

        # Determine document indices to compute
        if candidate_indices is None:
            target_indices = list(range(len(self.colbert_embeddings)))
        else:
            target_indices = candidate_indices

        # Compute ColBERT scores for target documents
        scores = []
        total_docs = len(target_indices)
        print(f"  Computing ColBERT scores for {total_docs} candidate documents...")
        for i, doc_idx in enumerate(target_indices):
            doc_colbert = self.colbert_embeddings[doc_idx]
            score = self.bge_model.colbert_score(
                query_colbert,
                doc_colbert
            )
            scores.append((doc_idx, score))

        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [(int(idx), float(score)) for idx, score in scores[:self.colbert_top_k]]

        return results
    
    def _fuse_results(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
        colbert_results: List[Tuple[int, float]],
        query: str = None,
    ) -> List[NodeWithScore]:
        """
        Fuse results from three retrieval methods - improved fusion strategy.

        Args:
            dense_results: Dense retrieval results.
            sparse_results: Sparse retrieval results.
            colbert_results: ColBERT retrieval results.
            query: Query text for dynamic weight adjustment.

        Returns:
            Fused node list.
        """
        # Collect all document indices
        all_indices = set()
        for idx, _ in dense_results:
            all_indices.add(idx)
        for idx, _ in sparse_results:
            all_indices.add(idx)
        for idx, _ in colbert_results:
            all_indices.add(idx)

        # Create score mappings
        dense_scores = {idx: score for idx, score in dense_results}
        sparse_scores = {idx: score for idx, score in sparse_results}
        colbert_scores = {idx: score for idx, score in colbert_results}

        # Improved normalization strategy: use max normalization
        dense_values = list(dense_scores.values())
        sparse_values = list(sparse_scores.values())
        colbert_values = list(colbert_scores.values())

        dense_max = max(dense_values) if dense_values else 1.0
        sparse_max = max(sparse_values) if sparse_values else 1.0
        colbert_max = max(colbert_values) if colbert_values else 1.0

        # Dynamic weight adjustment: detect if query contains many keywords
        is_keyword_rich_query = False
        if query:
            query_words = query.split()
            keyword_indicators = ['strategies', 'plan', 'implement', 'improve', 'reduce', 'program', 'training', 'efficiency', 'turnover', 'requirements', 'regulatory', 'compliance', 'critical', 'limits', 'quality', 'assurance', 'responsibilities', 'services', 'haccp', 'food', 'safety']
            keyword_count = sum(1 for word in query_words if any(kw in word.lower() for kw in keyword_indicators))
            is_keyword_rich_query = len(query_words) > 15 or keyword_count > 3
            print(f"  Query analysis: words={len(query_words)}, keyword_count={keyword_count}")

        # Adjust weights based on query type
        if is_keyword_rich_query:
            # Keyword-rich query: increase Sparse weight
            weights = (0.1, 0.75, 0.15)  # dense, sparse, colbert
        else:
            # Semantic query: use default weights
            weights = (self.dense_weight, self.sparse_weight, self.colbert_weight)

        # Compute fused scores
        fused_results = []
        for idx in all_indices:
            doc = self.documents[idx]
            
            # Get normalized scores (using max normalization)
            dense_score = dense_scores.get(idx, 0.0) / dense_max if dense_max > 0 else 0.0
            sparse_score = sparse_scores.get(idx, 0.0) / sparse_max if sparse_max > 0 else 0.0
            colbert_score = colbert_scores.get(idx, 0.0) / colbert_max if colbert_max > 0 else 0.0

            # Improved adaptive weights: moderate boost for keyword queries
            if is_keyword_rich_query:
                dense_boost = 1.0 + (dense_score > 0.7) * 0.2
                sparse_boost = 1.0 + (sparse_score > 0.4) * 0.5
                colbert_boost = 1.0 + (colbert_score > 0.5) * 0.4
            else:
                dense_boost = 1.0 + (dense_score > 0.6) * 0.3
                sparse_boost = 1.0 + (sparse_score > 0.6) * 0.3
                colbert_boost = 1.0 + (colbert_score > 0.6) * 0.3

            # Weighted fusion (considering adaptive weights)
            fused_score = (
                weights[0] * dense_boost * dense_score +
                weights[1] * sparse_boost * sparse_score +
                weights[2] * colbert_boost * colbert_score
            )

            # Create node
            node = NodeWithScore(
                node=TextNode(
                    text=doc['text'],
                    metadata=doc['metadata']
                ),
                score=fused_score
            )
            
            fused_results.append((fused_score, node, {
                'dense_score': dense_scores.get(idx, 0.0),
                'sparse_score': sparse_scores.get(idx, 0.0),
                'colbert_score': colbert_scores.get(idx, 0.0),
                'fused_score': fused_score,
            }))

        # Sort by fused score
        fused_results.sort(key=lambda x: x[0], reverse=True)

        # Return node list
        return [node for _, node, _ in fused_results]
    
    def retrieve(self, query: str, return_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Execute hybrid retrieval with reranking.

        Args:
            query: Query text.
            return_scores: Whether to return scores.

        Returns:
            List of retrieval results.
        """
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")

        # 1. Three retrieval methods
        print("\n[1] Dense Retrieval (Semantic Similarity)...")
        dense_results = self._dense_retrieve(query)
        print(f"  Retrieved {len(dense_results)} documents")
        if dense_results:
            print(f"  Top score: {dense_results[0][1]:.4f}")

        print("\n[2] Sparse Retrieval (Lexical Matching)...")
        sparse_results = self._sparse_retrieve(query)
        print(f"  Retrieved {len(sparse_results)} documents")
        if sparse_results:
            print(f"  Top score: {sparse_results[0][1]:.4f}")

        print("\n[3] ColBERT Retrieval (Fine-grained Matching)...")
        # Provide larger candidate set for ColBERT: Dense + Sparse + additional expansion
        candidate_indices = list(set([idx for idx, _ in dense_results + sparse_results]))
        # If candidate documents are too few, expand to larger set
        if len(candidate_indices) < 25:
            # Re-encode query for expanding candidate documents
            query_output = self.bge_model.encode(
                [query],
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
            query_embedding = query_output['dense_vecs'][0]
            # Select top similar documents as additional candidates
            all_dense_scores = np.dot(self.dense_embeddings, query_embedding)
            top_additional_indices = np.argsort(all_dense_scores)[::-1][:35]  # Get more candidates
            candidate_indices = list(set(candidate_indices + top_additional_indices.tolist()))
        print(f"  Using {len(candidate_indices)} candidate documents for ColBERT scoring...")
        colbert_results = self._colbert_retrieve(query, candidate_indices)
        print(f"  Retrieved {len(colbert_results)} documents")
        if colbert_results:
            print(f"  Top score: {colbert_results[0][1]:.4f}")

        # 2. Fuse results
        print("\n[4] Fusing results...")
        fused_nodes = self._fuse_results(dense_results, sparse_results, colbert_results, query)
        print(f"  Fused {len(fused_nodes)} documents")

        # 3. Reranking
        print("\n[5] Reranking...")
        # Prepare reranking data: query-document pairs
        pairs = []
        nodes_to_rerank = fused_nodes[:self.rerank_top_n * 2]  # Rerank 2x documents
        for node in nodes_to_rerank:
            pairs.append([query, node.text])
        
        if pairs:
            rerank_scores = self.reranker.compute_score(pairs, normalize=True)
            if isinstance(rerank_scores, (list, np.ndarray)):
                # Update node scores
                for i, node in enumerate(nodes_to_rerank):
                    node.score = float(rerank_scores[i])

                # Re-sort by rerank scores
                fused_nodes = sorted(
                    nodes_to_rerank,
                    key=lambda x: x.score,
                    reverse=True
                )[:self.rerank_top_n]
            else:
                # Single score (unlikely, but handle it)
                fused_nodes = nodes_to_rerank[:self.rerank_top_n]
        else:
            fused_nodes = fused_nodes[:self.rerank_top_n]
        
        print(f"  After reranking: {len(fused_nodes)} documents")

        # 4. Parse results
        results = []
        for node in fused_nodes:
            metadata = node.metadata or {}
            chunk_id_val = metadata.get('chunk_id') or metadata.get('context_id') or 'unknown'
            global_ctx_idx = metadata.get('global_context_index') or metadata.get('global_index') or metadata.get('global_context_index', 'unknown')

            result = {
                'chunk_id': chunk_id_val,
                'global_context_index': global_ctx_idx,
                'text': node.text,
                'metadata': metadata,
            }
            if return_scores:
                result['score'] = node.score
            results.append(result)

        # 5. Neighboring chunk expansion
        if self.expand_context:
            print("\n[6] Expanding context with neighboring chunks...")
            results = self._expand_chunks(results)

        # 6. Print results
        print(f"\nTop {len(results)} results after reranking:")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Chunk ID: {result.get('chunk_id','unknown')}  |  Global Index: {result.get('global_context_index','unknown')}")
            if 'score' in result:
                print(f"   Rerank Score: {result['score']:.4f}")
            print(f"   Text: {result['text'][:200]}...")

        return results
    
    def evaluate(self, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Args:
            test_queries: List of test queries.

        Returns:
            Evaluation metrics.
        """
        total_recall = 0.0
        total_precision = 0.0
        total_mrr = 0.0
        
        for query_data in test_queries:
            query = query_data['rewritten_query']
            ground_truth_ids = set(query_data['context_ids'])
            
            # Retrieve
            results = self.retrieve(query, return_scores=True)
            retrieved_ids = [r.get('chunk_id') or r.get('context_id') for r in results]

            # Compute metrics
            correct = len(set(retrieved_ids) & ground_truth_ids)
            recall = correct / len(ground_truth_ids) if ground_truth_ids else 0
            total_recall += recall
            
            # Precision@k
            precision = correct / len(retrieved_ids) if retrieved_ids else 0
            total_precision += precision
            
            # MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, ret_id in enumerate(retrieved_ids, 1):
                if ret_id in ground_truth_ids:
                    mrr = 1.0 / i
                    break
            total_mrr += mrr
            
            print(f"\n{'='*60}")
            print(f"Query: {query[:80]}...")
            print(f"  Recall@{len(retrieved_ids)}: {recall:.3f}")
            print(f"  Precision@{len(retrieved_ids)}: {precision:.3f}")
            print(f"  MRR: {mrr:.3f}")
            print(f"  Ground truth: {ground_truth_ids}")
            print(f"  Retrieved: {retrieved_ids[:5]}...")
        
        n = len(test_queries)
        metrics = {
            'recall': total_recall / n,
            'precision': total_precision / n,
            'mrr': total_mrr / n,
        }
        
        print(f"\n{'='*80}")
        print("Overall Performance Metrics:")
        print(f"{'='*80}")
        print(f"  Average Recall@{self.rerank_top_n}: {metrics['recall']:.3f}")
        print(f"  Average Precision@{self.rerank_top_n}: {metrics['precision']:.3f}")
        print(f"  Average MRR: {metrics['mrr']:.3f}")
        print(f"{'='*80}")
        
        return metrics

    def cleanup_gpu_memory(self):
        """
        Clean up GPU memory and cache.
        """
        # Lightweight cleanup: release PyTorch cache and trigger GC
        try:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                gc.collect()
            except Exception:
                pass

            print("GPU memory cleaned up successfully")
        except Exception as e:
            print(f"Warning: Failed to cleanup GPU resources: {e}")

    def __del__(self):
        """
        Destructor: clean up resources.
        """
        try:
            self._is_shutting_down = True
        except Exception:
            pass


def load_test_queries(queries_path: str, num_samples: int = 5) -> List[Dict[str, Any]]:
    """Load test queries from either generated QA records or flat query records."""
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            if line.strip():
                record = json.loads(line)
                if record.get("qa_pairs"):
                    for qa_pair in record["qa_pairs"]:
                        query = qa_pair.get("query")
                        if query:
                            queries.append(
                                {
                                    "query": query,
                                    "chunk_ids": record.get("chunk_ids", []),
                                    "source_record": record,
                                }
                            )
                else:
                    queries.append(record)
    return queries


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Run BGE-M3 hybrid retrieval over a prepared RAG context file")
    parser.add_argument("--contexts-path", required=True, help="Path to a *_contexts.jsonl file")
    parser.add_argument("--query", default=None, help="Single query to run")
    parser.add_argument("--queries-file", default=None, help="Optional generated rag_qa_pairs.jsonl or flat query JSONL")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of query records to run from --queries-file")
    parser.add_argument("--bge-m3-path", default=None, help="BGE-M3 model path")
    parser.add_argument("--reranker-path", default=None, help="BGE reranker model path")
    parser.add_argument("--devices", default=None, help="Device string passed to FlagEmbedding, e.g. cuda:0")
    parser.add_argument("--rerank-top-n", type=int, default=5, help="Number of reranked chunks to return")
    parser.add_argument("--no-expand-context", action="store_true", help="Disable neighboring chunk expansion")
    args = parser.parse_args()

    print("="*80)
    print("Advanced Hybrid Retrieval System (Dense + Sparse + ColBERT + Reranking)")
    print("="*80)

    # 1. Initialize system
    print("\n[Step 1] Initializing hybrid retrieval system...")
    system = AdvancedHybridRetrieval(
        contexts_path=args.contexts_path,
        bge_m3_path=args.bge_m3_path,
        reranker_path=args.reranker_path,
        dense_top_k=5,
        sparse_top_k=5,
        colbert_top_k=5,
        rerank_top_n=args.rerank_top_n,
        dense_weight=0.2,
        sparse_weight=0.5,
        colbert_weight=0.3,
        use_fp16=True,
        batch_size=16,
        devices=args.devices,
        expand_context=not args.no_expand_context,
    )

    if args.query:
        system.retrieve(args.query)

    if args.queries_file:
        print("\n[Step 2] Loading generated QA queries...")
        test_queries = load_test_queries(args.queries_file, num_samples=args.num_samples)
        print(f"Loaded {len(test_queries)} test queries")
        for query_data in test_queries[: args.num_samples]:
            query = query_data.get("query") or query_data.get("rewritten_query")
            if not query:
                continue
            system.retrieve(query)

    if not args.query and not args.queries_file:
        print("No query supplied. Use --query or --queries-file.")


if __name__ == "__main__":
    main()
