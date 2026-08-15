"""Engine and session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.db.models import Base

_connect_args = {}
_engine_kwargs: dict = {"pool_pre_ping": True}

if settings.database_url.startswith("sqlite"):
    # FastAPI serves requests on a threadpool; SQLite needs this to be shared.
    _connect_args["check_same_thread"] = False

    # NullPool opens a fresh connection per session and closes it after. Pooled
    # SQLite connections hold their WAL read snapshot between checkouts, so a
    # request served by a recycled connection could miss rows another request
    # had already committed — leads synced a moment earlier were invisible to
    # the very next query. Connecting is microseconds for a local file.
    _engine_kwargs = {"poolclass": NullPool}

engine: Engine = create_engine(
    settings.database_url,
    echo=settings.sql_echo,
    future=True,
    connect_args=_connect_args,
    **_engine_kwargs,
)


if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record) -> None:
        cursor = dbapi_connection.cursor()
        # WAL keeps reads from blocking during a sync; FKs are off by default.
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

        # Hand transaction control to SQLAlchemy. Left to itself, the pysqlite
        # driver opens implicit transactions it never closes, so a pooled
        # connection keeps an old WAL read snapshot and a request can miss
        # writes another request already committed — a lead synced a moment ago
        # would be absent from the queue until that connection was recycled.
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _sqlite_begin(connection) -> None:
        connection.exec_driver_sql("BEGIN")


SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables that do not exist yet."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts, jobs, and the MCP server."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
