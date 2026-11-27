"""Validated public request/response contracts for the matching service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Product(StrictModel):
    title: str = Field(min_length=1, max_length=500)
    offer_id: str = Field(default="", max_length=120)
    brand: str = Field(default="", max_length=150)
    description: str = Field(default="", max_length=1500)
    price: str = Field(default="", max_length=64)
    currency: str = Field(default="", max_length=12)


class ScoreRequest(StrictModel):
    left: Product
    right: Product


class ScoreResponse(StrictModel):
    model_name: str
    match_probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    is_match: bool
    calibrated: bool = False
    warning: str


class SearchRequest(StrictModel):
    query: Product
    top_k: int = Field(default=5, ge=1, le=30)
    method: Literal["bm25", "reranked"] = "bm25"


class SearchCandidate(StrictModel):
    offer_id: str
    title: str
    brand: str
    score: float
    score_type: Literal["bm25", "match_probability"]


class SearchResponse(StrictModel):
    method: str
    candidates: list[SearchCandidate]
    catalog_size: int


class ReviewSubmission(StrictModel):
    query: Product
    candidate: Product
    decision: Literal["match", "nonmatch", "unsure"]
    reviewer: str = Field(min_length=1, max_length=80)
    notes: str = Field(default="", max_length=1000)


class ReviewRecord(ReviewSubmission):
    review_id: str
    created_at: str
