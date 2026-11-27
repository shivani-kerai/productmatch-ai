"""FastAPI interface for the local ProductMatch AI service.

All inference and feedback endpoints are authenticated when a key is configured.
The application refuses to start without a key outside explicitly allowed local
development, and never silently loads benchmark data or arbitrary artifacts.
"""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import PlainTextResponse

from productmatch.api.metrics import RequestMetrics, monotonic_seconds
from productmatch.api.runtime import Runtime, ServiceConfig
from productmatch.api.schemas import (
    Product,
    ReviewRecord,
    ReviewSubmission,
    ScoreRequest,
    ScoreResponse,
    SearchCandidate,
    SearchRequest,
    SearchResponse,
)
from productmatch.api.store import ReviewStore
from productmatch.retrieval.benchmark import Offer


def _offer(product: Product) -> Offer:
    return Offer(
        offer_id=product.offer_id,
        title=product.title,
        brand=product.brand,
        description=product.description,
        price=product.price,
        currency=product.currency,
    )


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    """App factory with explicit configuration and startup lifecycle."""
    config = config or ServiceConfig.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not config.api_key or not config.review_token:
            raise RuntimeError("Both API key and reviewer token must be configured")
        app.state.runtime = Runtime(config)
        app.state.reviews = ReviewStore(config.review_db)
        yield

    app = FastAPI(
        title="ProductMatch AI",
        version="0.6.0",
        description="Local prototype with optional model-bound development calibration.",
        lifespan=lifespan,
    )

    app.state.metrics = RequestMetrics()

    @app.middleware("http")
    async def observe_request(request: Request, call_next):
        start = monotonic_seconds()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            app.state.metrics.observe(
                request.method, request.url.path, status_code, monotonic_seconds() - start
            )

    def _require_api_key(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    ) -> None:
        if config.api_key and not hmac.compare_digest(x_api_key or "", config.api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")

    def _require_reviewer_token(
        x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
        x_review_token: Annotated[str | None, Header(alias="X-Review-Token")] = None,
    ) -> None:
        _require_api_key(x_api_key)
        if not config.review_token or not hmac.compare_digest(
            x_review_token or "", config.review_token
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "productmatch-ai"}

    @app.get("/metrics")
    def metrics(_: None = Depends(_require_api_key)) -> PlainTextResponse:
        return PlainTextResponse(app.state.metrics.exposition(), media_type="text/plain")

    @app.get("/ready")
    def ready(_: None = Depends(_require_api_key)) -> dict[str, object]:
        runtime = app.state.runtime
        return {
            "service": "productmatch-ai",
            "catalog_ready": bool(runtime.retriever),
            "model_ready": runtime.model is not None,
            "catalog_size": len(runtime.catalog),
            "model_name": runtime.model_name,
            "model_fingerprint": runtime.model_fingerprint,
            "calibrated": runtime.calibrator is not None,
            "calibration_policy": runtime.calibration_policy,
        }

    @app.post("/v1/match", response_model=ScoreResponse)
    def score_pair(body: ScoreRequest, _: None = Depends(_require_api_key)) -> ScoreResponse:
        runtime = app.state.runtime
        if runtime.model is None or runtime.threshold is None:
            raise HTTPException(status_code=503, detail="Matching model not configured")
        probability = runtime.score_pairs([(_offer(body.left), _offer(body.right))])[0]
        return ScoreResponse(
            model_name=runtime.model_name or "unknown",
            match_probability=probability,
            threshold=runtime.threshold,
            is_match=probability >= runtime.threshold,
            calibrated=runtime.calibrator is not None,
            calibration_policy=runtime.calibration_policy,
            warning=(
                "Development-only calibration fitted on all validation pairs; "
                "NO independent audit or production calibration claim."
                if runtime.calibration_policy == "development_only_full_validation"
                else (
                    "Development-calibrated on previously model-selected validation; "
                    "not independently validated for production."
                    if runtime.calibrator is not None
                    else "Development validation threshold; not a calibrated production decision."
                )
            ),
        )

    @app.post("/v1/search", response_model=SearchResponse)
    def search(body: SearchRequest, _: None = Depends(_require_api_key)) -> SearchResponse:
        runtime = app.state.runtime
        if runtime.retriever is None:
            raise HTTPException(status_code=503, detail="Product catalog not configured")
        if body.method == "reranked" and runtime.model is None:
            raise HTTPException(status_code=503, detail="Matching model not configured")
        query = _offer(body.query)
        initial = runtime.retriever.search(
            query, max(body.top_k, 50) if body.method == "reranked" else body.top_k
        )
        if body.method == "reranked":
            selected = initial
            pairs = [(query, runtime.offers[candidate_id]) for candidate_id, _ in selected]
            scores = runtime.score_pairs(pairs)
            initial = sorted(
                [
                    (candidate_id, probability)
                    for (candidate_id, _), probability in zip(selected, scores, strict=True)
                ],
                key=lambda item: (-item[1], item[0]),
            )[: body.top_k]
        candidates = [
            SearchCandidate(
                offer_id=offer_id,
                title=runtime.offers[offer_id].title,
                brand=runtime.offers[offer_id].brand,
                score=score,
                score_type="match_probability" if body.method == "reranked" else "bm25",
            )
            for offer_id, score in initial
        ]
        return SearchResponse(
            method=body.method, candidates=candidates, catalog_size=len(runtime.catalog)
        )

    @app.post("/v1/reviews", response_model=ReviewRecord, status_code=201)
    def submit_review(
        body: ReviewSubmission, _: None = Depends(_require_reviewer_token)
    ) -> ReviewRecord:
        return app.state.reviews.add(body)

    @app.get("/v1/reviews", response_model=list[ReviewRecord])
    def list_reviews(
        _: None = Depends(_require_reviewer_token), limit: int = 100
    ) -> list[ReviewRecord]:
        if not 1 <= limit <= 500:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
        return app.state.reviews.list(limit)

    return app


app = create_app()
