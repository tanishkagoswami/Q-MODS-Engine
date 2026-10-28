import os
import logging
from pymongo import MongoClient

# MongoDB Client Setup
MONGO_URI = os.getenv("MONGO_URI")

# TASK 1.4 & 3.10 FIX: Fast fail if URI is missing, add 3-second timeout, and enforce UTC timezones
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000, tz_aware=True) if MONGO_URI else None

# Database & Collection references
db = client["qmods_engine"] if client else None
snapshots_collection = db["clinical_snapshots"] if db is not None else None

logger = logging.getLogger(__name__)