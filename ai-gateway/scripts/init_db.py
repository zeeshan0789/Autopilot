"""Create or upgrade the database schema by running all Alembic migrations.

If your database already has the traces table from before Alembic was added (created
by an older version of this script), run "alembic stamp 0001" once first.
"""

from alembic import command
from alembic.config import Config

from app.settings import BASE_DIR


def main():
    config = Config(str(BASE_DIR / "alembic.ini"))
    config.set_main_option("script_location", str(BASE_DIR / "alembic"))
    command.upgrade(config, "head")
    print("Database schema is up to date.")


if __name__ == "__main__":
    main()