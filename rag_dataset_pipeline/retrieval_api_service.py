from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from pathlib import Path

from advanced_hybrid_retrieval import AdvancedHybridRetrieval
from llamaindex_hybrid_retriever import HybridRetriever


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    return_scores: Optional[bool] = True
    expand_context: Optional[bool] = True
    enable_privacy_filter: Optional[bool] = False
    requesting_client: Optional[str] = None


class QueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total: int
    privacy_filter_stats: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    party_id: str
    total_contexts: int


class RetrievalService:

    def __init__(
        self,
        party_id: str,
        contexts_path: str,
        bge_m3_path: Optional[str] = None,
        reranker_path: Optional[str] = None,
        **kwargs
    ):
        self.party_id = party_id
        self.contexts_path = contexts_path

        print(f"\n[{party_id}] Initializing AdvancedHybridRetrieval...")
        self.hybrid_retrieval = AdvancedHybridRetrieval(
            contexts_path=contexts_path,
            bge_m3_path=bge_m3_path,
            reranker_path=reranker_path,
            use_cache=True,
            cache_key=party_id,
            **kwargs
        )

        self.retriever = HybridRetriever(hybrid_retrieval=self.hybrid_retrieval)

        print(f"[{party_id}] Retrieval service initialized successfully!")
        print(f"[{party_id}] Total contexts: {len(self.hybrid_retrieval.contexts)}")
        print(f"[{party_id}] BGE-M3 and Reranker models are loaded and ready for inference!")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        return_scores: bool = True,
        expand_context: bool = True,
        enable_privacy_filter: bool = False,
        requesting_client: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        original_rerank_top_n = self.hybrid_retrieval.rerank_top_n
        original_expand_context = self.hybrid_retrieval.expand_context
        self.hybrid_retrieval.rerank_top_n = top_k
        self.hybrid_retrieval.expand_context = expand_context

        try:
            results = self.hybrid_retrieval.retrieve(
                query=query,
                return_scores=return_scores
            )
        finally:
            self.hybrid_retrieval.rerank_top_n = original_rerank_top_n
            self.hybrid_retrieval.expand_context = original_expand_context

        return results, None

    def get_retriever(self) -> HybridRetriever:
        return self.retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = RetrievalService(
        party_id=app.state.party_id,
        contexts_path=app.state.contexts_path,
        **app.state.kwargs
    )
    app.state.service = service

    yield

    try:
        if hasattr(app.state.service, 'hybrid_retrieval') and app.state.service.hybrid_retrieval:
            app.state.service.hybrid_retrieval.cleanup_gpu_memory()
        print(f"[{app.state.party_id}] Service shutdown complete, GPU memory cleaned up")
    except Exception as e:
        print(f"Warning: Error during service shutdown cleanup: {e}")


def create_app(party_id: str, contexts_path: str, port: int = 8000, **kwargs) -> FastAPI:
    app = FastAPI(
        title=f"Retrieval API - {party_id}",
        description=f"Hybrid Retrieval Service for {party_id}",
        version="1.0.0",
        lifespan=lifespan
    )

    app.state.party_id = party_id
    app.state.contexts_path = contexts_path
    app.state.kwargs = kwargs

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        svc = getattr(app.state, "service", None)
        if svc is None:
            return HealthResponse(status="starting", party_id=app.state.party_id, total_contexts=0)
        return HealthResponse(status="healthy", party_id=app.state.party_id, total_contexts=len(svc.hybrid_retrieval.contexts))

    @app.post("/retrieve", response_model=QueryResponse)
    async def retrieve(request: QueryRequest):
        try:
            svc = getattr(app.state, "service", None)
            if svc is None:
                raise HTTPException(status_code=503, detail="Service is initializing")
            results, privacy_stats = svc.retrieve(
                query=request.query,
                top_k=request.top_k,
                return_scores=request.return_scores,
                expand_context=request.expand_context,
                enable_privacy_filter=request.enable_privacy_filter,
                requesting_client=request.requesting_client,
            )

            return QueryResponse(
                results=results,
                query=request.query,
                total=len(results),
                privacy_filter_stats=privacy_stats,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/info")
    async def get_info():
        svc = getattr(app.state, "service", None)
        if svc is None:
            return {"status": "service_not_initialized"}

        return {
            "party_id": app.state.party_id,
            "contexts_path": app.state.contexts_path,
            "total_contexts": len(svc.hybrid_retrieval.contexts),
            "dense_top_k": svc.hybrid_retrieval.dense_top_k,
            "sparse_top_k": svc.hybrid_retrieval.sparse_top_k,
            "colbert_top_k": svc.hybrid_retrieval.colbert_top_k,
            "rerank_top_n": svc.hybrid_retrieval.rerank_top_n,
        }

    @app.post("/cleanup")
    async def cleanup_memory():
        svc = getattr(app.state, "service", None)
        if svc is None:
            return {"status": "service_not_initialized"}

        try:
            svc.hybrid_retrieval.cleanup_gpu_memory()
            return {"status": "memory_cleaned"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return app


def run_service(
    party_id: str,
    contexts_path: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    **kwargs
):
    app = create_app(party_id, contexts_path, port, **kwargs)

    print(f"\n{'='*80}")
    print(f"Starting Retrieval API Service for {party_id}")
    print(f"{'='*80}")
    print(f"Contexts path: {contexts_path}")
    print(f"Server: http://{host}:{port}")
    print(f"Health check: http://{host}:{port}/health")
    print(f"API docs: http://{host}:{port}/docs")
    print(f"{'='*80}\n")

    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Retrieval API Service")
    parser.add_argument("--party-id", type=str, required=True, help="Party ID (e.g., party1)")
    parser.add_argument("--contexts-path", type=str, required=True, help="Path to contexts file")
    parser.add_argument("--port", type=int, default=8000, help="Service port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Service host")
    parser.add_argument("--bge-m3-path", type=str, default=str(DEFAULT_ROOT / "bge-m3"), help="BGE-M3 model path")
    parser.add_argument("--reranker-path", type=str, default=str(DEFAULT_ROOT / "bge-reranker-v2-m3"), help="Reranker model path")
    parser.add_argument("--devices", type=str, default=None, help="GPU devices to use (e.g., 'cuda:0')")

    args = parser.parse_args()

    devices = args.devices
    if devices == "None" or devices is None:
        devices = None

    run_service(
        party_id=args.party_id,
        contexts_path=args.contexts_path,
        port=args.port,
        host=args.host,
        bge_m3_path=args.bge_m3_path,
        reranker_path=args.reranker_path,
        devices=devices,
    )
