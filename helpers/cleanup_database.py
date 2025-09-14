#!/usr/bin/env python
"""
Database cleanup script to remove unused tables and optimize the SQLite database.
This script will:
1. Backup the database
2. Drop unused tables
3. Clean orphaned records
4. Vacuum the database to reclaim space
"""

import sqlite3
import shutil
import os
from datetime import datetime
from pathlib import Path

def backup_database(db_path: str) -> str:
    """Create a backup of the database before cleanup."""
    backup_path = f"{db_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    return backup_path

def analyze_database(con: sqlite3.Connection) -> dict:
    """Analyze the current state of the database."""
    cursor = con.cursor()

    # Get initial size
    cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
    initial_size = cursor.fetchone()[0]

    # Count tables
    cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
    table_count = cursor.fetchone()[0]

    # Get row counts for important tables
    stats = {
        'size_bytes': initial_size,
        'table_count': table_count,
        'memories': 0,
        'embeddings': 0,
        'orphaned_embeddings': 0
    }

    try:
        cursor.execute("SELECT COUNT(*) FROM memories")
        stats['memories'] = cursor.fetchone()[0]
    except:
        pass

    try:
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        stats['embeddings'] = cursor.fetchone()[0]
    except:
        pass

    try:
        cursor.execute("""
            SELECT COUNT(*) FROM embeddings
            WHERE memory_id NOT IN (SELECT memory_id FROM memories)
        """)
        stats['orphaned_embeddings'] = cursor.fetchone()[0]
    except:
        pass

    return stats

def drop_unused_tables(con: sqlite3.Connection) -> list:
    """Drop unused tables from the database."""
    cursor = con.cursor()

    # Tables to drop
    tables_to_drop = [
        # Unused advanced features (all empty)
        'clusters',
        'cluster_membership',
        'blocks',
        'block_members',
        'memory_synapses',
        'memory_importance',
        'embedding_drift',
    ]

    dropped = []
    for table in tables_to_drop:
        try:
            # Check if table exists and get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            # Drop the table
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            dropped.append((table, count))
            print(f"  Dropped table: {table} ({count} rows)")
        except sqlite3.OperationalError:
            # Table doesn't exist
            pass

    return dropped

def clean_orphaned_records(con: sqlite3.Connection) -> int:
    """Clean orphaned records from the database."""
    cursor = con.cursor()

    # Clean orphaned embeddings
    cursor.execute("""
        DELETE FROM embeddings
        WHERE memory_id NOT IN (SELECT memory_id FROM memories)
    """)
    orphaned_cleaned = cursor.rowcount

    if orphaned_cleaned > 0:
        print(f"  Cleaned {orphaned_cleaned} orphaned embeddings")

    return orphaned_cleaned

def rebuild_fts(con: sqlite3.Connection) -> bool:
    """Optionally rebuild the FTS table."""
    cursor = con.cursor()

    try:
        # Check if FTS is actually being used for search
        cursor.execute("""
            SELECT COUNT(*) FROM mem_fts
        """)
        fts_count = cursor.fetchone()[0]

        if fts_count > 0:
            print(f"  FTS table has {fts_count} entries")
            print("  Note: FTS is only used for DELETE operations, not search")
            # We'll keep it for now since flask_app.py references it
            return False

    except sqlite3.OperationalError:
        pass

    return False

def vacuum_database(con: sqlite3.Connection):
    """Vacuum the database to reclaim space."""
    print("\nVacuuming database to reclaim space...")
    con.execute("VACUUM")
    print("  Vacuum complete")

def main():
    db_path = "data/amemory.sqlite3"

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return 1

    print("="*60)
    print("DATABASE CLEANUP SCRIPT")
    print("="*60)

    # Backup database
    print("\n1. BACKUP")
    print("-"*40)
    backup_path = backup_database(db_path)
    print(f"  Backup created: {backup_path}")

    # Connect to database
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")

    try:
        # Analyze before
        print("\n2. INITIAL ANALYSIS")
        print("-"*40)
        before_stats = analyze_database(con)
        print(f"  Database size: {before_stats['size_bytes'] / (1024*1024):.2f} MB")
        print(f"  Total tables: {before_stats['table_count']}")
        print(f"  Memories: {before_stats['memories']}")
        print(f"  Embeddings: {before_stats['embeddings']}")
        print(f"  Orphaned embeddings: {before_stats['orphaned_embeddings']}")

        # Drop unused tables
        print("\n3. DROP UNUSED TABLES")
        print("-"*40)
        dropped = drop_unused_tables(con)
        if not dropped:
            print("  No unused tables to drop")

        # Clean orphaned records
        print("\n4. CLEAN ORPHANED RECORDS")
        print("-"*40)
        orphaned = clean_orphaned_records(con)
        if orphaned == 0:
            print("  No orphaned records found")

        # Commit changes
        con.commit()

        # Vacuum
        vacuum_database(con)

        # Analyze after
        print("\n5. FINAL ANALYSIS")
        print("-"*40)
        after_stats = analyze_database(con)
        print(f"  Database size: {after_stats['size_bytes'] / (1024*1024):.2f} MB")
        print(f"  Total tables: {after_stats['table_count']}")
        print(f"  Memories: {after_stats['memories']}")
        print(f"  Embeddings: {after_stats['embeddings']}")

        # Summary
        print("\n6. SUMMARY")
        print("-"*40)
        size_saved = (before_stats['size_bytes'] - after_stats['size_bytes']) / (1024*1024)
        tables_removed = before_stats['table_count'] - after_stats['table_count']

        print(f"  Tables removed: {tables_removed}")
        print(f"  Space saved: {size_saved:.2f} MB")
        print(f"  Orphaned records cleaned: {orphaned}")

        if size_saved > 0:
            reduction = (1 - after_stats['size_bytes']/before_stats['size_bytes']) * 100
            print(f"  Size reduction: {reduction:.1f}%")

        print("\n" + "="*60)
        print("Cleanup completed successfully!")
        print(f"Backup saved at: {backup_path}")

    except Exception as e:
        print(f"\nError during cleanup: {e}")
        print("Rolling back changes...")
        con.rollback()
        return 1
    finally:
        con.close()

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())