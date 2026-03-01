"""
db/database.py
Async and sync database clients — PostgreSQL via SQLAlchemy + asyncpg/psycopg2,
MongoDB via Motor, Redis via redis-py async.
"""
from contextlib import contextmanager
from typing import AsyncGenerator, Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient

from core.config import get_settings

settings = get_settings()

# ── Async PostgreSQL engine (FastAPI routes) ──────────────────────────────────
engine = create_async_engine(
    settings.async_database_url,
    echo=settings.DEBUG,
    pool_size=20,
    max_overflow=40,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={"server_settings": {"application_name": "backend"}},
)

AsyncSessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


# ── Sync PostgreSQL engine (Celery tasks / Alembic) ───────────────────────────
sync_engine = create_engine(
    settings.database_url,
    echo=settings.DEBUG,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    connect_args={"application_name": "celery_worker"},
)

SyncSessionFactory = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy ORM models."""
    pass


# ── Async session dependency (FastAPI) ───────────────────────────────────────
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields an async DB session."""
    async with AsyncSessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Legacy alias
get_db_session = get_db


# ── Sync session context manager (Celery tasks) ───────────────────────────────
@contextmanager
def get_sync_db() -> Generator[Session, None, None]:
    """Context manager for sync DB access (Celery workers, scripts)."""
    session: Session = SyncSessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── MongoDB ───────────────────────────────────────────────────────────────────
_mongo_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = AsyncIOMotorClient(settings.mongodb_url)
    return _mongo_client


def get_mongo_db():
    return get_mongo_client()[settings.MONGO_DB]


# ── Redis ─────────────────────────────────────────────────────────────────────
_redis_client: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis_client


# ── Health checks ─────────────────────────────────────────────────────────────
async def check_postgres() -> tuple[bool, str | None]:
    try:
        from sqlalchemy import text
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        return False, str(e)


async def check_mongo() -> tuple[bool, str | None]:
    try:
        client = get_mongo_client()
        await client.admin.command("ping")
        return True, None
    except Exception as e:
        return False, str(e)


async def check_redis() -> tuple[bool, str | None]:
    try:
        r = await get_redis()
        await r.ping()
        return True, None
    except Exception as e:
        return False, str(e)


# ── Lifecycle helpers ─────────────────────────────────────────────────────────
async def init_db():
    """Create all tables (run on startup in dev; use Alembic migrations in prod)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Graceful shutdown."""
    await engine.dispose()
    if _mongo_client:
        _mongo_client.close()
    if _redis_client:
        await _redis_client.aclose()
