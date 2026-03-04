"""
main.py
FastAPI application factory with lifespan, middleware, and router registration.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from core.config import get_settings
from db.database import close_db, init_db, check_postgres, check_mongo, check_redis
from api.routes.analysis import router as analysis_router
from api.routes.analytics import router as analytics_router
from api.routes.auth import router as auth_router
from api.routes.users import router as users_router
from api.middleware.rate_limiter import RateLimiterMiddleware
from api.routes.admin import router as admin_router

settings = get_settings()


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.ENVIRONMENT == "production":
        import sys
        import json
        logger.remove()

        def json_sink(message):
            record = message.record
            log_dict = {
                "time": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "module": record["module"],
            }
            sys.stdout.write(json.dumps(log_dict) + "\n")
            sys.stdout.flush()

        logger.add(json_sink)

        if settings.SENTRY_DSN:
            import sentry_sdk
            sentry_sdk.init(
                dsn=settings.SENTRY_DSN,
                environment=settings.ENVIRONMENT,
                traces_sample_rate=1.0,
            )

    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    await init_db()
    yield
    logger.info("Shutting down…")
    await close_db()


# ── App factory ────────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-powered YouTube video classification and structured information "
            "extraction. Multi-modal analysis combining computer vision, "
            "speech-to-text, and NLP."
        ),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    import uuid
    from starlette.middleware.base import BaseHTTPMiddleware

    class CorrelationIDMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            request.state.request_id = req_id
            
            with logger.contextualize(request_id=req_id):
                response = await call_next(request)
                response.headers["X-Request-ID"] = req_id
                return response

    # Starlette applies middleware in REVERSE registration order.
    # Add CORSMiddleware LAST so it wraps everything and always fires.
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(RateLimiterMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static Files ───────────────────────────────────────────────────────────
    import os
    os.makedirs(settings.FRAMES_DIR, exist_ok=True)
    app.mount(
        "/frames",
        StaticFiles(directory=settings.FRAMES_DIR),
        name="frames"
    )

    # ── Prometheus metrics ─────────────────────────────────────────────────────
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # ── Routers ────────────────────────────────────────────────────────────────
    app.include_router(analysis_router)
    app.include_router(analytics_router)
    app.include_router(auth_router)
    app.include_router(users_router)
    app.include_router(admin_router)

    # ── Health check ───────────────────────────────────────────────────────────
    @app.get("/health", tags=["Ops"])
    async def health():
        """Deep health check — probes each backing service."""
        pg_ok, pg_err = await check_postgres()
        mongo_ok, mongo_err = await check_mongo()
        redis_ok, redis_err = await check_redis()
        
        celery_ok = False
        celery_err = None
        try:
            from services.pipeline import celery_app
            i = celery_app.control.inspect()
            stats = i.stats()
            if stats:
                celery_ok = True
            else:
                celery_err = "No active workers found"
        except Exception as e:
            celery_err = str(e)

        all_ok = pg_ok and mongo_ok and redis_ok and celery_ok
        return JSONResponse(
            status_code=200 if all_ok else 503,
            content={
                "status": "ok" if all_ok else "degraded",
                "version": settings.APP_VERSION,
                "services": {
                    "postgres": {"ok": pg_ok, "error": pg_err},
                    "mongodb":  {"ok": mongo_ok, "error": mongo_err},
                    "redis":    {"ok": redis_ok, "error": redis_err},
                    "celery":   {"ok": celery_ok, "error": celery_err},
                },
            },
        )

    @app.get("/ready", tags=["Ops"])
    async def ready():
        """Kubernetes readiness probe — returns 200 once startup is complete."""
        return {"ready": True}

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled error on {request.url}: {exc}")
        origin = request.headers.get("origin", "")
        headers = {}
        # Mirror CORS headers manually so the browser can read the error body
        allowed = settings.ALLOWED_ORIGINS
        if origin in allowed:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred"},
            headers=headers,
        )

    return app


app = create_app()

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else 4,
        log_level="debug" if settings.DEBUG else "info",
    )
