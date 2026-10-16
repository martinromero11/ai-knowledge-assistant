import logging

from sqlalchemy import JSON, create_engine, inspect
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.types import TypeDecorator

from .config import settings

log = logging.getLogger("app.database")


def _normalize_url(url: str) -> str:
    # Use the psycopg 3 driver for plain postgres:// or postgresql:// URLs.
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


DATABASE_URL = _normalize_url(settings.database_url)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

# True once init_db() confirms the pgvector extension is available.
VECTOR_AVAILABLE = False


class Base(DeclarativeBase):
    pass


class EmbeddingType(TypeDecorator):
    """A pgvector `vector` column on PostgreSQL with pgvector; a JSON list of floats elsewhere."""

    impl = JSON
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql" and VECTOR_AVAILABLE:
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector(settings.embedding_dim))
        return dialect.type_descriptor(JSON(none_as_null=True))

    def process_bind_param(self, value, dialect):
        return None if value is None else [float(x) for x in value]


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def vector_backend() -> str:
    return "pgvector" if VECTOR_AVAILABLE else "python"


def init_db() -> None:
    """Create tables. On PostgreSQL, also set up full-text search and (if available) pgvector."""
    global VECTOR_AVAILABLE
    from . import models  # noqa: F401  (registers the tables)

    if engine.dialect.name == "postgresql":
        try:
            with engine.begin() as conn:
                conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            VECTOR_AVAILABLE = True
        except Exception as e:  # extension not installed on this server
            log.warning("pgvector unavailable, semantic search will run in Python: %s", e)
            VECTOR_AVAILABLE = False

    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        _add_missing_columns(conn)

    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector "
                "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
            )
            conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING GIN (tsv)")
            if VECTOR_AVAILABLE:
                conn.exec_driver_sql(
                    "CREATE INDEX IF NOT EXISTS ix_chunks_embedding "
                    "ON chunks USING hnsw (embedding vector_cosine_ops)"
                )


def _add_missing_columns(conn) -> None:
    """Add new nullable columns to tables created by an older version of the app.

    create_all() only creates missing tables, not missing columns. A real project
    would use a migration tool such as Alembic; this covers simple additive changes.
    """
    insp = inspect(conn)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        existing = {c["name"] for c in insp.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing or not column.nullable:
                continue
            column_type = column.type.compile(dialect=conn.dialect)
            conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {column.name} {column_type}")
            log.info("Added column %s.%s (%s)", table.name, column.name, column_type)
