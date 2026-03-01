"""
api/routes/auth.py

User authentication endpoints:
  - POST /register   → create account
  - POST /login      → get JWT access + refresh token
  - POST /refresh    → rotate tokens
  - GET  /me         → current user profile
  - GET  /spotify    → initiate Spotify OAuth flow
  - GET  /spotify/callback → handle Spotify OAuth callback
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
import time
from jose import JWTError, jwt
import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import spotipy
from spotipy.oauth2 import SpotifyOAuth

from core.config import get_settings
from db.database import get_db_session
from db.models import User
from models.schemas import (
    OKResponse,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)

settings = get_settings()
router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])


# ── JWT helpers ────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    # bcrypt requires bytes, returns bytes. We store as a string.
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode('utf-8'), hashed.encode('utf-8'))
    except ValueError:
        return False


def _create_token(data: dict, expires_delta: timedelta) -> str:
    payload = {**data, "exp": datetime.now(timezone.utc) + expires_delta}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(user_id: str) -> str:
    return _create_token(
        {"sub": user_id, "type": "access"},
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_id: str) -> str:
    return _create_token(
        {"sub": user_id, "type": "refresh"},
        timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> User:
    """FastAPI dependency — validates Bearer token and returns the User."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise JWTError("wrong token type")
        user_id: str = payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    from db.database import get_redis
    redis = await get_redis()
    if await redis.get(f"token_blacklist:{token}"):
        raise HTTPException(status_code=401, detail="Token has been revoked")

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """FastAPI dependency — requires the user to have the ADMIN role."""
    from db.models import UserRole
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Requires administrative privileges",
        )
    return current_user

# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/register", response_model=UserResponse, status_code=201)
async def register(body: UserRegisterRequest, db: AsyncSession = Depends(get_db_session)):
    """Create a new user account."""
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        email=body.email,
        hashed_password=_hash_password(body.password),
        display_name=body.display_name,
    )
    db.add(user)
    await db.flush()
    return user


MAX_ATTEMPTS = 5
LOCKOUT_SECS = 900  # 15 min

async def check_login_rate_limit(identifier: str):
    from db.database import get_redis
    r = await get_redis()
    attempts = await r.get(f"login_attempts:{identifier}")
    if attempts and int(attempts) >= MAX_ATTEMPTS:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts",
            headers={"Retry-After": str(LOCKOUT_SECS)}
        )

async def record_failed_login(identifier: str):
    from db.database import get_redis
    r = await get_redis()
    key = f"login_attempts:{identifier}"
    await r.incr(key)
    await r.expire(key, LOCKOUT_SECS)

async def clear_login_attempts(identifier: str):
    from db.database import get_redis
    r = await get_redis()
    await r.delete(f"login_attempts:{identifier}")


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLoginRequest, db: AsyncSession = Depends(get_db_session)):
    """Authenticate and return JWT tokens."""
    await check_login_rate_limit(body.email)
    
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()

    if not user or not _verify_password(body.password, user.hashed_password):
        await record_failed_login(body.email)
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # Clear attempts on success
    await clear_login_attempts(body.email)

    user_id = str(user.id)
    return TokenResponse(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(request: Request, db: AsyncSession = Depends(get_db_session)):
    """Exchange a valid refresh token for a new token pair."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Provide refresh token as Bearer")

    token = auth.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        if payload.get("type") != "refresh":
            raise JWTError("not a refresh token")
        user_id: str = payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")

    return TokenResponse(
        access_token=create_access_token(user_id),
        refresh_token=create_refresh_token(user_id),
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user


@router.post("/logout")
async def logout(request: Request):
    """Invalidate current token via Redis blacklist."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            exp = payload.get("exp", 0)
            ttl = max(0, exp - int(time.time()))
            
            from db.database import get_redis
            redis = await get_redis()
            await redis.setex(f"token_blacklist:{token}", ttl, "1")
        except Exception:
            pass
    return OKResponse(message="Logged out")


# ── Spotify OAuth ─────────────────────────────────────────────────────────────

def _spotify_oauth_manager() -> Optional[SpotifyOAuth]:
    if not settings.SPOTIFY_CLIENT_ID or not settings.SPOTIFY_CLIENT_SECRET:
        return None
    return SpotifyOAuth(
        client_id=settings.SPOTIFY_CLIENT_ID,
        client_secret=settings.SPOTIFY_CLIENT_SECRET,
        redirect_uri=settings.SPOTIFY_REDIRECT_URI,
        scope="playlist-modify-public playlist-modify-private user-read-private",
    )


@router.get("/spotify")
async def spotify_oauth_start(current_user: User = Depends(get_current_user)):
    """
    Redirect the user to Spotify's OAuth authorization page.
    The state parameter encodes the user ID for verification on callback.
    """
    oauth = _spotify_oauth_manager()
    if not oauth:
        raise HTTPException(status_code=503, detail="Spotify integration not configured")

    auth_url = oauth.get_authorize_url(state=str(current_user.id))
    return RedirectResponse(url=auth_url)


@router.get("/spotify/callback")
async def spotify_oauth_callback(
    code: str,
    state: str,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Handle Spotify OAuth callback.
    Exchange the auth code for tokens and store them on the user record.
    """
    oauth = _spotify_oauth_manager()
    if not oauth:
        raise HTTPException(status_code=503, detail="Spotify integration not configured")

    try:
        token_info = oauth.get_access_token(code, as_dict=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Spotify token exchange failed: {exc}")

    # Verify state → user ID
    try:
        user_uuid = uuid.UUID(state)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    result = await db.execute(select(User).where(User.id == user_uuid))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.spotify_access_token = token_info.get("access_token")
    user.spotify_refresh_token = token_info.get("refresh_token")
    await db.flush()

    return OKResponse(message="Spotify connected successfully. You can now create playlists.")
