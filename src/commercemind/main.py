from fastapi import FastAPI

from commercemind.config import get_settings
from commercemind.logging import get_logger
from commercemind.schemas import HealthResponse, SearchRequest, SearchResponse
from commercemind.services.search import SearchService

settings = get_settings()
logger = get_logger(__name__)
search_service = SearchService()

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


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    logger.info("search_requested", query=request.query, top_k=request.top_k)
    return search_service.search(request)
