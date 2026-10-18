import asyncio

from sqlalchemy import text

from app.db.models import Base
from app.db.session import engine


async def main():
    async with engine.begin() as conn:
        version = (await conn.execute(text("select version()"))).scalar()
        print("Connected:", version)
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()
    print("Table 'traces' ready.")


if __name__ == "__main__":
    asyncio.run(main())