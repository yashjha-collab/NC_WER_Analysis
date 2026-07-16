from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.models import Base

engine = create_async_engine(settings.nc_wer_database_url, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    def _migrate(connection) -> None:
        Base.metadata.create_all(connection)
        # SQLite: add human_url if an older DB was created without it.
        try:
            rows = connection.exec_driver_sql("PRAGMA table_info(call_records)").fetchall()
            cols = {row[1] for row in rows}
            if "human_url" not in cols:
                connection.exec_driver_sql(
                    "ALTER TABLE call_records ADD COLUMN human_url TEXT"
                )
        except Exception:
            pass

    async with engine.begin() as conn:
        await conn.run_sync(_migrate)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
