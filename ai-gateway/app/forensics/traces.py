"""Reads the gateway's trace rows (by request id) for the forensics tool."""

from sqlalchemy import bindparam, text

from app.db.session import SessionLocal, engine


async def fetch_traces(request_ids: list[str]) -> dict:
    """request id -> trace row as a dict. Raises when the database cannot be reached."""
    if not request_ids:
        return {}
    query = text("select * from traces where request_id in :ids").bindparams(
        bindparam("ids", expanding=True)
    )
    try:
        async with SessionLocal() as session:
            rows = (await session.execute(query, {"ids": list(request_ids)})).mappings().all()
        return {row["request_id"]: dict(row) for row in rows}
    finally:
        await engine.dispose()