from time import perf_counter

from fastapi import FastAPI

from commercemind.config import get_settings
from commercemind.logging import get_logger
from commercemind.monitoring.metrics import MetricsRegistry
from commercemind.ranking.learned import LearnedLinearProductRanker
from commercemind.ranking.rankers import ProductRanker
from commercemind.retrieval.vector_index import FaissVectorIndex
from commercemind.schemas import (
    EndpointMetricsResponse,
    HealthResponse,
    MetricsResponse,
    RecommendationRequest,
    RecommendationResponse,
    SearchRequest,
    SearchResponse,
)
from commercemind.services.catalog import load_catalog_data
from commercemind.services.recommendations import RecommendationService
from commercemind.services.search import SearchService

settings = get_settings()
logger = get_logger(__name__)
metrics_registry = MetricsRegistry()


def _load_configured_vector_index() -> FaissVectorIndex | None:
    if settings.vector_index_dir is None:
        return None

    index = FaissVectorIndex.load(settings.vector_index_dir)
    logger.info(
        "vector_index_loaded",
        path=str(settings.vector_index_dir),
        vectors=index.size,
        dimensions=index.dimensions,
    )
    return index


def _load_configured_ranker() -> ProductRanker | None:
    if settings.ranker_model_path is None:
        return None

    ranker = LearnedLinearProductRanker.load(catalog_data.products, settings.ranker_model_path)
    logger.info(
        "learned_ranker_loaded",
        path=str(settings.ranker_model_path),
    )
    return ranker


catalog_data = load_catalog_data(settings.catalog_source, settings.database_url)
logger.info(
    "catalog_loaded",
    source=settings.catalog_source,
    products=catalog_data.products.height,
    interactions=catalog_data.interactions.height,
)
search_service = SearchService(
    products=catalog_data.products,
    ranker=_load_configured_ranker(),
    vector_index=_load_configured_vector_index(),
    embedding_backend=settings.embedding_backend,
    embedding_model_name=settings.embedding_model_name,
)
recommendation_service = RecommendationService(
    products=catalog_data.products,
    interactions=catalog_data.interactions,
)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="CommerceMind retrieval and ranking service.",
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    logger.info("health_check_requested", environment=settings.environment)
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(
        endpoints={
            endpoint: EndpointMetricsResponse(**snapshot.__dict__)
            for endpoint, snapshot in metrics_registry.snapshot().items()
        }
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    logger.info("search_requested", query=request.query, top_k=request.top_k)
    started_at = perf_counter()

    try:
        response = search_service.search(request)
    except Exception:
        metrics_registry.record(
            "search",
            latency_ms=_elapsed_ms(started_at),
            failed=True,
        )
        raise

    metrics_registry.record(
        "search",
        latency_ms=_elapsed_ms(started_at),
        result_count=len(response.results),
    )
    return response


@app.post("/recommendations", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest) -> RecommendationResponse:
    logger.info(
        "recommendations_requested",
        user_id=request.user_id,
        top_k=request.top_k,
    )
    started_at = perf_counter()

    try:
        response = recommendation_service.recommend(request)
    except Exception:
        metrics_registry.record(
            "recommendations",
            latency_ms=_elapsed_ms(started_at),
            failed=True,
        )
        raise

    metrics_registry.record(
        "recommendations",
        latency_ms=_elapsed_ms(started_at),
        result_count=len(response.results),
    )
    return response


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
