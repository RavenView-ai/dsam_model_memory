from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.config import cfg

store = MemoryStore(cfg.db_path)
with store.connect() as con:
    # Check component_embeddings table
    cursor = con.execute("PRAGMA table_info(component_embeddings)")
    cols = cursor.fetchall()

    print("component_embeddings table columns:")
    for col in cols:
        print(f"  {col['name']} ({col['type']})")