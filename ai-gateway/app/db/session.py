from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.settings import DATABASE_URL

# timeout chhota: Postgres na chale to request atki na rahe
engine = create_async_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    connect_args={"timeout": 3},
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False)