from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.platform import router as platform_router
from app.api.routes import router
from app.core.config import settings
from app.db.session import init_db

ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
)
logger = logging.getLogger("leadscoring")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Path(settings.database_url.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    init_db()
    logger.info("database ready")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        lifespan=lifespan,
        description=(
            "Lead scoring platform. Connect a CRM or spreadsheet, map it onto the "
            "canonical schema, and score, explain, rank, and retrain from real outcomes. "
            "The same services back the MCP server in mcp_server/."
        ),
    )

    # Scoped to the local dev origins the dashboard is served from. A wildcard
    # origin combined with allow_credentials is rejected by browsers anyway.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Attach a request id and log timing — the minimum for debugging in production."""
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error request_id=%s path=%s", request_id, request.url.path)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error.", "request_id": request_id},
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"

        if not request.url.path.startswith("/frontend"):
            logger.info(
                "%s %s -> %s in %.1fms request_id=%s",
                request.method, request.url.path, response.status_code, elapsed_ms, request_id,
            )
        return response

    @app.get("/healthz", include_in_schema=False)
    def liveness() -> dict:
        """Liveness probe — no database, no model, just the process."""
        return {"status": "alive", "version": settings.version}

    @app.get("/readyz", include_in_schema=False)
    def readiness() -> dict:
        """Readiness probe — the database must answer."""
        from sqlalchemy import text

        from app.db.session import engine

        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception as error:  # pragma: no cover
            return JSONResponse(status_code=503, content={"status": "not_ready", "detail": str(error)})
        return {"status": "ready"}

    app.mount("/frontend", StaticFiles(directory=ROOT / "frontend"), name="frontend")
    app.include_router(router)
    app.include_router(platform_router)
    return app


app = create_app()
