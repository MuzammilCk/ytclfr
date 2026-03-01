"""
api/middleware/rate_limiter.py

Redis-based sliding-window rate limiter middleware.
Limits requests per IP to settings.RATE_LIMIT_PER_MINUTE per minute.
Returns HTTP 429 with Retry-After header when limit is exceeded.
"""
import time
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import get_settings
from db.database import get_redis

settings = get_settings()

# Paths that are exempt from rate limiting
_EXEMPT_PATHS = {"/health", "/ready", "/metrics", "/api/docs", "/api/redoc", "/api/openapi.json"}
_EXEMPT_GET_PREFIXES = (
    "/api/v1/analyses/",   # status, result, and list endpoints
)

def _is_exempt(request: Request) -> bool:
    path = request.url.path
    if path in _EXEMPT_PATHS:
        return True
    if request.method == "GET":
        for prefix in _EXEMPT_GET_PREFIXES:
            if path.startswith(prefix):
                return True
    return False

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter.
    Each request increments a Redis counter keyed by IP.
    The key expires after 60 seconds (the window).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip exempt paths
        if _is_exempt(request):
            return await call_next(request)

        # Identify client by X-Forwarded-For (behind nginx) or direct IP
        client_ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

        redis_key = f"rate_limit_sw:{client_ip}"
        window    = 60   # seconds
        if request.headers.get("Authorization"):
            limit = settings.RATE_LIMIT_PER_MINUTE  # e.g. 300
        else:
            if request.method == "GET":
                limit = 10
            else:
                limit = 5    # anonymous tier

        try:
            redis = await get_redis()
            now = time.time()
            
            await redis.zadd(redis_key, {str(now): now})
            await redis.zremrangebyscore(redis_key, 0, now - window)
            count = await redis.zcard(redis_key)
            await redis.expire(redis_key, window)

            # Set informational headers
            request.state.rate_limit_remaining = max(0, limit - count)

            if count > limit:
                ttl = window
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": (
                            f"Rate limit exceeded. Maximum {settings.RATE_LIMIT_PER_MINUTE} "
                            f"requests per minute."
                        )
                    },
                    headers={
                        "Retry-After":              str(ttl),
                        "X-RateLimit-Limit":        str(settings.RATE_LIMIT_PER_MINUTE),
                        "X-RateLimit-Remaining":    "0",
                        "X-RateLimit-Reset":        str(int(time.time()) + ttl),
                    },
                )
        except Exception:
            # If Redis is unavailable, fail open (don't block traffic)
            pass

        response = await call_next(request)

        # Attach rate limit headers to all successful responses
        try:
            remaining = getattr(request.state, "rate_limit_remaining", settings.RATE_LIMIT_PER_MINUTE)
            response.headers["X-RateLimit-Limit"]     = str(settings.RATE_LIMIT_PER_MINUTE)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
        except Exception:
            pass

        return response
