from __future__ import annotations
import sqlite3
from typing import List, Tuple, Optional, Iterable, Dict, Any
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import numpy as np

from ..types import MemoryRecord

SCHEMA = '''
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    source_event_id TEXT NOT NULL,
    who_type TEXT NOT NULL,
    who_id TEXT NOT NULL,
    who_label TEXT,
    who_list TEXT,  -- JSON array of who entities
    what TEXT NOT NULL,
    when_ts TEXT NOT NULL,
    when_list TEXT,  -- JSON array of when expressions
    where_type TEXT NOT NULL,
    where_value TEXT NOT NULL,
    where_lat REAL,
    where_lon REAL,
    where_list TEXT,  -- JSON array of where locations
    why TEXT,
    how TEXT,
    raw_text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    embed_model TEXT NOT NULL,
    extra_json TEXT,
    created_at TEXT NOT NULL
);

-- FTS5 table removed - lexical search was broken and not needed

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_stats (
    memory_id TEXT PRIMARY KEY REFERENCES memories(memory_id) ON DELETE CASCADE,
    accesses INTEGER NOT NULL DEFAULT 0,
    last_access TEXT
);

-- Note: Advanced features (clustering, blocks, synapses, importance, drift) have been
-- removed as they were never implemented and had no data. If these features are needed
-- in the future, they can be re-added with proper implementation.
'''

class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_schema()

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _ensure_schema(self):
        with self.connect() as con:
            con.executescript(SCHEMA)

    def upsert_memory(self, rec: MemoryRecord, embedding: bytes, dim: int):
        with self.connect() as con:
            con.execute(
                """INSERT OR REPLACE INTO memories
                (memory_id, session_id, source_event_id, who_type, who_id, who_label, who_list, what, when_ts, when_list,
                 where_type, where_value, where_lat, where_lon, where_list, why, how, raw_text, token_count,
                 embed_model, extra_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                (
                    rec.memory_id, rec.session_id, rec.source_event_id, rec.who.type, rec.who.id, rec.who.label,
                    rec.who_list, rec.what, rec.when.isoformat(), rec.when_list, rec.where.type, rec.where.value, 
                    rec.where.lat, rec.where.lon, rec.where_list, rec.why, rec.how, rec.raw_text, rec.token_count, 
                    rec.embed_model, json.dumps(rec.extra), datetime.now(timezone.utc).astimezone().isoformat()
                )
            )
            # FTS5 table removed - lexical search not needed
            con.execute(
                """INSERT OR REPLACE INTO embeddings (memory_id, dim, vector) VALUES (?, ?, ?)""", 
                (rec.memory_id, dim, embedding)
            )
            con.execute(
                """INSERT OR IGNORE INTO usage_stats (memory_id, accesses, last_access) VALUES (?, 0, ?)""", 
                (rec.memory_id, datetime.now(timezone.utc).astimezone().isoformat())
            )

    def fetch_memories(self, ids: List[str]):
        qmarks = ','.join('?' * len(ids))
        with self.connect() as con:
            rows = con.execute(f"SELECT * FROM memories WHERE memory_id IN ({qmarks})", ids).fetchall()
        # Convert Row objects to dictionaries
        return [dict(row) for row in rows]

    def record_access(self, memory_ids: List[str]):
        if not memory_ids:
            return
        with self.connect() as con:
            now = datetime.now(timezone.utc).astimezone().isoformat()
            for mid in memory_ids:
                con.execute(
                    """INSERT INTO usage_stats (memory_id, accesses, last_access)
                           VALUES (?, 1, ?)
                           ON CONFLICT(memory_id) DO UPDATE SET 
                             accesses = accesses + 1,
                             last_access = excluded.last_access""",
                    (mid, now)
                )

    # Lexical search removed - FTS5 was broken and not needed

    def get_by_actor(self, actor_id: str, limit: int = 100) -> List[sqlite3.Row]:
        """Retrieve memories from a specific actor."""
        sql = """
            SELECT memory_id, who_id, raw_text, when_ts, token_count
            FROM memories
            WHERE who_id = ?
            ORDER BY when_ts DESC
            LIMIT ?
        """
        with self.connect() as con:
            rows = con.execute(sql, (actor_id, limit)).fetchall()
        return rows
    
    def get_by_location(self, location: str, limit: int = 100) -> List[sqlite3.Row]:
        """Retrieve memories from a specific location."""
        sql = """
            SELECT memory_id, where_value, raw_text, when_ts, token_count
            FROM memories
            WHERE where_value = ?
            ORDER BY when_ts DESC
            LIMIT ?
        """
        with self.connect() as con:
            rows = con.execute(sql, (location, limit)).fetchall()
        return rows
    
    # get_by_actor_and_text removed - FTS5 was removed
    
    def actor_exists(self, actor_id: str) -> bool:
        """Check if an actor exists in the database."""
        sql = "SELECT COUNT(*) FROM memories WHERE who_id = ? LIMIT 1"
        with self.connect() as con:
            count = con.execute(sql, (actor_id,)).fetchone()[0]
        return count > 0
    
    def get_by_date(self, date: str, limit: int = 200) -> List[sqlite3.Row]:
        """Retrieve memories from a specific date."""
        sql = """
            SELECT memory_id, when_ts, raw_text, token_count, who_id, what
            FROM memories
            WHERE DATE(when_ts) = ?
            ORDER BY when_ts DESC
            LIMIT ?
        """
        with self.connect() as con:
            rows = con.execute(sql, (date, limit)).fetchall()
        return rows
    
    def get_by_date_range(self, start_date: str, end_date: str, limit: int = 200) -> List[sqlite3.Row]:
        """Retrieve memories from a date range."""
        sql = """
            SELECT memory_id, when_ts, raw_text, token_count, who_id, what
            FROM memories
            WHERE DATE(when_ts) BETWEEN ? AND ?
            ORDER BY when_ts DESC
            LIMIT ?
        """
        with self.connect() as con:
            rows = con.execute(sql, (start_date, end_date, limit)).fetchall()
        return rows
    
    def get_by_relative_time(self, relative_spec: str, reference_date: Optional[datetime] = None) -> List[sqlite3.Row]:
        """Retrieve memories from relative time period.
        
        Supports: 'today', 'yesterday', 'last_week', 'last_month', 'last_year'
        """
        from datetime import timedelta
        
        if reference_date is None:
            reference_date = datetime.now(timezone.utc).replace(tzinfo=None)
        
        if relative_spec == "today":
            target_date = reference_date.date()
            return self.get_by_date(str(target_date), limit=200)
        elif relative_spec == "yesterday":
            target_date = (reference_date - timedelta(days=1)).date()
            return self.get_by_date(str(target_date), limit=200)
        elif relative_spec == "last_week":
            end_date = reference_date.date()
            start_date = (reference_date - timedelta(days=7)).date()
            return self.get_by_date_range(str(start_date), str(end_date), limit=300)
        elif relative_spec == "last_month":
            end_date = reference_date.date()
            start_date = (reference_date - timedelta(days=30)).date()
            return self.get_by_date_range(str(start_date), str(end_date), limit=500)
        elif relative_spec == "last_year":
            end_date = reference_date.date()
            start_date = (reference_date - timedelta(days=365)).date()
            return self.get_by_date_range(str(start_date), str(end_date), limit=1000)
        else:
            # Default to last week if unknown
            end_date = reference_date.date()
            start_date = (reference_date - timedelta(days=7)).date()
            return self.get_by_date_range(str(start_date), str(end_date), limit=300)

    # Note: Block-related methods removed as tables were dropped
    # def create_block() and get_block() were here
    
    def get_usage_stats(self, memory_ids: List[str]) -> Dict[str, Dict]:
        """Get usage statistics for multiple memories"""
        if not memory_ids:
            return {}
        
        qmarks = ','.join('?' * len(memory_ids))
        with self.connect() as con:
            rows = con.execute(
                f"SELECT memory_id, accesses, last_access FROM usage_stats WHERE memory_id IN ({qmarks})",
                memory_ids
            ).fetchall()
        
        return {row['memory_id']: dict(row) for row in rows}
    
    # Note: Synapse-related methods removed as tables were dropped
    # update_synapse() and get_synapses() were here
    
    # Note: Importance-related methods removed as tables were dropped
    # update_importance() and get_importance_scores() were here
    
    # Note: decay_synapses() removed as memory_synapses table was dropped
    
    # Note: Embedding drift methods removed as tables were dropped
    # store_embedding_drift() and get_embedding_drift() were here

    def get_sample_embedding(self) -> Optional[np.ndarray]:
        """Get a sample embedding to determine dimension"""
        with self.connect() as con:
            row = con.execute(
                "SELECT vector FROM embeddings LIMIT 1"
            ).fetchone()

        if row and row['vector']:
            return np.frombuffer(row['vector'], dtype=np.float32)
        return None

    def get_all_embeddings_for_rebuild(self, batch_size: int = 500) -> List[Tuple[str, bytes]]:
        """Get all embeddings for rebuilding the FAISS index"""
        embeddings = []
        with self.connect() as con:
            cursor = con.execute(
                "SELECT memory_id, vector FROM embeddings WHERE vector IS NOT NULL"
            )

            while True:
                batch = cursor.fetchmany(batch_size)
                if not batch:
                    break
                embeddings.extend([(row['memory_id'], row['vector']) for row in batch])

        return embeddings

    def get_random_memory_ids(self, count: int) -> List[str]:
        """Get random memory IDs for verification"""
        with self.connect() as con:
            rows = con.execute(
                "SELECT memory_id FROM embeddings ORDER BY RANDOM() LIMIT ?",
                (count,)
            ).fetchall()

        return [row['memory_id'] for row in rows]

    def get_embedding_by_memory_id(self, memory_id: str) -> Optional[bytes]:
        """Get embedding vector for a specific memory ID"""
        with self.connect() as con:
            row = con.execute(
                "SELECT vector FROM embeddings WHERE memory_id = ?",
                (memory_id,)
            ).fetchone()

        if row and row['vector']:
            return row['vector']
        return None
