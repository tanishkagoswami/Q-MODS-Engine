import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient

# MongoDB Client Setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# Database & Collection references
db = client["qmods_engine"]
snapshots_collection = db["clinical_snapshots"]

logger = logging.getLogger(__name__)

# Fallback to local SQLite if TimescaleDB/PostgreSQL URL is not provided
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./backend/data/processed/qmods_local.db"
)

logger.info(f"Connecting database engine using URL: {DATABASE_URL}")

# Establish connection pooling configuration for production stability
if DATABASE_URL.startswith("postgresql"):
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=1800
    )
else:
    # Standard SQLite configuration (for easy local fallback without complex setups)
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """
    Dependency injection generator to provide request-scoped database sessions.
    Ensures safe session closing post-execution.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()