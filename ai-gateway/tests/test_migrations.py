import asyncio
import uuid

import pytest
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from app.settings import BASE_DIR, DATABASE_URL


def alembic_config(database_url: str | None = None) -> Config:
    config = Config(str(BASE_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BASE_DIR / "alembic"))
    config.attributes["configure_logger"] = False
    if database_url:
        config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    return config


def test_there_is_a_single_migration_head():
    """Two heads means two people created migrations from the same parent."""
    heads = ScriptDirectory.from_config(alembic_config()).get_heads()
    assert len(heads) == 1, heads


@pytest.mark.integration
def test_migrations_build_exactly_the_model_schema():
    """Run every migration on an empty scratch database, then compare it to the models.

    This fails when someone changes app/db/models.py without writing a migration.
    """
    base_url = make_url(DATABASE_URL)
    scratch_name = "autopilot_migtest_" + uuid.uuid4().hex[:8]
    admin_url = base_url.set(database="postgres")
    scratch_url = base_url.set(database=scratch_name)

    async def admin(sql: str):
        engine = create_async_engine(
            admin_url, isolation_level="AUTOCOMMIT", connect_args={"timeout": 3}
        )
        try:
            async with engine.connect() as conn:
                await conn.execute(text(sql))
        finally:
            await engine.dispose()

    try:
        asyncio.run(admin(f'CREATE DATABASE "{scratch_name}"'))
    except Exception as e:
        pytest.skip(f"Postgres not reachable ({type(e).__name__}); run docker compose up -d")

    try:
        config = alembic_config(scratch_url.render_as_string(hide_password=False))
        command.upgrade(config, "head")
        command.check(config)  # raises if the models differ from the migrated schema
    finally:
        asyncio.run(admin(f'DROP DATABASE IF EXISTS "{scratch_name}" WITH (FORCE)'))