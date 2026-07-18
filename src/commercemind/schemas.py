from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class HealthResponse(BaseModel):
    status: str = Field(description="Current health state of the service.")
    app: str = Field(description="Application name.")
    environment: str = Field(description="Runtime environment name.")


class EndpointMetricsResponse(BaseModel):
    request_count: int = Field(ge=0, description="Total requests recorded.")
    error_count: int = Field(ge=0, description="Requests that raised an error.")
    empty_result_count: int = Field(ge=0, description="Successful requests with no results.")
    total_latency_ms: float = Field(ge=0.0, description="Total observed latency in milliseconds.")
    average_latency_ms: float = Field(ge=0.0, description="Average request latency.")
    max_latency_ms: float = Field(ge=0.0, description="Maximum observed request latency.")
    error_rate: float = Field(ge=0.0, le=1.0, description="Fraction of requests that failed.")
    empty_result_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of requests that returned no results.",
    )


class MetricsResponse(BaseModel):
    endpoints: dict[str, EndpointMetricsResponse] = Field(
        default_factory=dict,
        description="Runtime metrics grouped by endpoint name.",
    )


class SearchRequest(BaseModel):
    query: NonBlankString = Field(description="Natural language product search query.")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return.")


class RecommendationRequest(BaseModel):
    user_id: NonBlankString = Field(description="User identifier for personalization.")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of items to return.")


class ItemResult(BaseModel):
    item_id: str = Field(description="Unique item identifier.")
    title: str = Field(description="Display title for the item.")
    score: float = Field(description="Model or retrieval score for the item.")


class SearchResponse(BaseModel):
    query: str = Field(description="Original search query.")
    results: list[ItemResult] = Field(default_factory=list, description="Ranked search results.")


class RecommendationResponse(BaseModel):
    user_id: str = Field(description="User identifier for the request.")
    results: list[ItemResult] = Field(
        default_factory=list,
        description="Ranked recommended items.",
    )
