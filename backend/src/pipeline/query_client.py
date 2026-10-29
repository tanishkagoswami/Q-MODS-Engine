import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
from typing import Optional

# Configure logging for production monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIMICQueryClient:
    """
    Client for extracting patient data from MIMIC-IV PostgreSQL/TimescaleDB database.
    Uses SQLAlchemy for robust connection pooling and safe query execution.
    """
    
    def __init__(self, db_url: str):
        """
        Initializes the database connection engine.
        
        Args:
            db_url (str): PostgreSQL connection string 
                          (e.g., 'postgresql://user:password@localhost:5432/mimiciv')
        """
        # pool_pre_ping ensures connections are alive before using them
        try:
            self.engine = create_engine(db_url, pool_size=10, max_overflow=20, pool_pre_ping=True)
            logger.info("Successfully initialized PostgreSQL connection pool.")
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    def get_patient_labs(self, subject_id: int, hadm_id: Optional[int] = None) -> pd.DataFrame:
        """
        Extracts sparse lab events (e.g., Creatinine, Bilirubin) for a specific patient.
        These are the '12-hour snapshots' needed for the organ state vector.
        """
        # Parameterized query to prevent SQL injection and improve caching.
        # Note: SQLAlchemy uses :param_name syntax
        query = """
            SELECT subject_id, hadm_id, itemid, charttime, valuenum, valueuom
            FROM mimiciv_hosp.labevents
            WHERE subject_id = :subject_id
        """
        params = {"subject_id": subject_id}
        
        if hadm_id:
            # FIX (Task 5.3): Corrected parameter name from stay_id to match hadm_id column filtering
            query += " AND hadm_id = :hadm_id"
            params["hadm_id"] = hadm_id
            
        query += " ORDER BY charttime ASC;"
        
        logger.info(f"Fetching lab events for subject_id: {subject_id}")
        return self._execute_query(query, params)

    def get_patient_vitals(self, subject_id: int) -> pd.DataFrame:
        """
        Extracts charted vitals (heart rate, SpO2, etc.) from the ICU.
        """
        query = """
            SELECT subject_id, stay_id, charttime, itemid, valuenum, valueuom
            FROM mimiciv_icu.chartevents
            WHERE subject_id = :subject_id
            ORDER BY charttime ASC;
        """
        logger.info(f"Fetching charted vitals for subject_id: {subject_id}")
        return self._execute_query(query, {"subject_id": subject_id})

    def _execute_query(self, query: str, params: dict = None) -> pd.DataFrame:
        """
        Internal method to safely execute pandas SQL reads.
        """
        try:
            with self.engine.connect() as conn:
                # SQLAlchemy 2.0+ requires wrapping the raw SQL string in text() 
                # when executing parameterized queries with pandas.
                df = pd.read_sql(text(query), conn, params=params)
            return df
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise