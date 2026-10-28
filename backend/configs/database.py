import os
import logging
from sqlalchemy import create_engine

from pymongo import MongoClient

# MongoDB Client Setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)

# Database & Collection references
db = client["qmods_engine"]
snapshots_collection = db["clinical_snapshots"]

logger = logging.getLogger(__name__)

