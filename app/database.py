from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.config import DATABASE_URL
from app.models import Base


engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def test_connection():
    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT version();")
        )

        print("Connected to PostgreSQL!")
        print(result.scalar())


def create_tables():
    Base.metadata.create_all(bind=engine)

    print("Database tables created!")


if __name__ == "__main__":
    test_connection()
    create_tables()