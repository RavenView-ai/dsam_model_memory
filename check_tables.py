import sqlite3
import os

db_path = 'data/memory_store.db'
if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in data directory: {os.listdir('data') if os.path.exists('data') else 'data dir not found'}")
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print('Tables in database:')
    for table in tables:
        print(f'  - {table[0]}')
    conn.close()