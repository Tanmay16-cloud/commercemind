from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(description="Current health state of the service.")
    app: str = Field(description="Application name.")
    environment: str = Field(description="Runtime environment name.")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural language product search query.")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return.")


class RecommendationRequest(BaseModel):
    user_id: str = Field(min_length=1, description="User identifier for personalization.")
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
