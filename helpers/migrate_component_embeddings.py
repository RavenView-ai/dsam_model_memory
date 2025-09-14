#!/usr/bin/env python3
"""
Migration script to generate component embeddings for existing memories.
This adds WHO, WHERE, WHEN, WHY, and HOW embeddings to the FAISS index.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.embedding.component_embedder import get_component_embedder
from agentic_memory.embedding import get_llama_embedder

def check_existing_component_embeddings(index: FaissIndex) -> dict:
    """Check if component embeddings already exist in the index."""
    stats = {
        'who': 0,
        'where': 0,
        'when': 0,
        'why': 0,
        'how': 0,
        'total_memories': 0
    }

    # This is a simple heuristic - in production you might want to
    # track this more systematically
    print("Checking for existing component embeddings...")

    # We can't directly query FAISS for ID patterns, so we'll estimate
    # based on the migration status file if it exists
    migration_status_file = Path(cfg.db_path).parent / '.component_embeddings_migrated'
    if migration_status_file.exists():
        with open(migration_status_file, 'r') as f:
            stats = json.load(f)

    return stats

def migrate_memory_components(dry_run: bool = False, batch_size: int = 100):
    """
    Migrate existing memories to include component embeddings.

    Args:
        dry_run: If True, only analyze without making changes
        batch_size: Number of memories to process at once
    """
    print("=" * 60)
    print("COMPONENT EMBEDDINGS MIGRATION")
    print("=" * 60)

    # Initialize components
    store = MemoryStore(cfg.db_path)
    component_embedder = get_component_embedder()
    embedder = get_llama_embedder()
    # Get embedding dimension from the embedder
    embed_dim = embedder.embedding_dimension
    index = FaissIndex(embed_dim, cfg.index_path)

    # Check existing status
    existing_stats = check_existing_component_embeddings(index)
    if existing_stats.get('total_memories', 0) > 0:
        print(f"\nFound existing migration status:")
        print(f"  Total memories processed: {existing_stats['total_memories']}")
        print(f"  WHO embeddings: {existing_stats['who']}")
        print(f"  WHERE embeddings: {existing_stats['where']}")
        print(f"  WHEN embeddings: {existing_stats['when']}")
        print(f"  WHY embeddings: {existing_stats['why']}")
        print(f"  HOW embeddings: {existing_stats['how']}")

        response = input("\nMigration may have already been run. Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            return

    # Get all memories
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row

    count_query = "SELECT COUNT(*) as count FROM memories"
    total_count = con.execute(count_query).fetchone()['count']

    print(f"\nFound {total_count} memories to process")

    if dry_run:
        print("\n*** DRY RUN MODE - No changes will be made ***\n")

    # Process in batches
    offset = 0
    stats = {
        'who': 0,
        'where': 0,
        'when': 0,
        'why': 0,
        'how': 0,
        'errors': 0,
        'skipped': 0
    }

    # Create progress bar
    pbar = tqdm(total=total_count, desc="Processing memories")

    while offset < total_count:
        # Fetch batch of memories
        query = """
            SELECT memory_id, who_id, who_label, who_type, who_list,
                   what, when_ts, when_list,
                   where_value, where_type, where_lat, where_lon, where_list,
                   why, how
            FROM memories
            ORDER BY created_at
            LIMIT ? OFFSET ?
        """

        rows = con.execute(query, (batch_size, offset)).fetchall()

        for row in rows:
            memory_dict = dict(row)
            memory_id = memory_dict['memory_id']

            try:
                # Generate component embeddings
                component_embeddings = component_embedder.embed_all_components(memory_dict)

                if not dry_run:
                    # Store component embeddings in FAISS with prefixed IDs
                    if component_embeddings.get('who') is not None:
                        index.add(f"who:{memory_id}", component_embeddings['who'])
                        stats['who'] += 1

                    if component_embeddings.get('where') is not None:
                        index.add(f"where:{memory_id}", component_embeddings['where'])
                        stats['where'] += 1

                    if component_embeddings.get('when') is not None:
                        index.add(f"when:{memory_id}", component_embeddings['when'])
                        stats['when'] += 1

                    if component_embeddings.get('why') is not None:
                        index.add(f"why:{memory_id}", component_embeddings['why'])
                        stats['why'] += 1

                    if component_embeddings.get('how') is not None:
                        index.add(f"how:{memory_id}", component_embeddings['how'])
                        stats['how'] += 1
                else:
                    # In dry run, just count what would be added
                    if component_embeddings.get('who') is not None:
                        stats['who'] += 1
                    if component_embeddings.get('where') is not None:
                        stats['where'] += 1
                    if component_embeddings.get('when') is not None:
                        stats['when'] += 1
                    if component_embeddings.get('why') is not None:
                        stats['why'] += 1
                    if component_embeddings.get('how') is not None:
                        stats['how'] += 1

            except Exception as e:
                stats['errors'] += 1
                if stats['errors'] <= 5:  # Only show first 5 errors
                    print(f"\nError processing memory {memory_id}: {e}")

            pbar.update(1)

        # Save index periodically (every 10 batches)
        if not dry_run and (offset // batch_size) % 10 == 0:
            index.save()

        offset += batch_size

    pbar.close()
    con.close()

    # Final save
    if not dry_run:
        print("\nSaving FAISS index...")
        index.save()

        # Save migration status
        migration_status = {
            'total_memories': total_count,
            'who': stats['who'],
            'where': stats['where'],
            'when': stats['when'],
            'why': stats['why'],
            'how': stats['how'],
            'migration_date': datetime.now().isoformat(),
            'errors': stats['errors']
        }

        migration_status_file = Path(cfg.db_path).parent / '.component_embeddings_migrated'
        with open(migration_status_file, 'w') as f:
            json.dump(migration_status, f, indent=2)

        print(f"Migration status saved to {migration_status_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Total memories processed: {total_count}")
    print(f"WHO embeddings created: {stats['who']}")
    print(f"WHERE embeddings created: {stats['where']}")
    print(f"WHEN embeddings created: {stats['when']}")
    print(f"WHY embeddings created: {stats['why']}")
    print(f"HOW embeddings created: {stats['how']}")
    if stats['errors'] > 0:
        print(f"Errors encountered: {stats['errors']}")

    if dry_run:
        print("\n*** This was a dry run - no changes were made ***")
        print("Run without --dry-run to apply changes")
    else:
        print("\n✅ Migration completed successfully!")

def verify_migration():
    """Verify that component embeddings are working correctly."""
    print("\n" + "=" * 60)
    print("VERIFYING MIGRATION")
    print("=" * 60)

    component_embedder = get_component_embedder()
    embedder = get_llama_embedder()
    # Get embedding dimension from the embedder
    embed_dim = embedder.embedding_dimension
    index = FaissIndex(embed_dim, cfg.index_path)

    # Test searching for a common actor name
    test_queries = [
        ("who", "user"),
        ("who", "assistant"),
        ("where", "office"),
        ("where", "room")
    ]

    for component, query in test_queries:
        print(f"\nTesting {component.upper()} search for '{query}':")

        # Generate embedding for the test query
        if component == "who":
            embedding = component_embedder.embed_who(query)
        elif component == "where":
            embedding = component_embedder.embed_where(query)
        else:
            continue

        if embedding is None:
            print(f"  Failed to generate embedding for '{query}'")
            continue

        # Search for similar embeddings
        results = index.search(embedding, 5)

        # Filter to component-specific results
        component_results = [(id, score) for id, score in results if id.startswith(f"{component}:")]

        if component_results:
            print(f"  Found {len(component_results)} matches:")
            for id, score in component_results[:3]:
                print(f"    {id}: similarity {score:.3f}")
        else:
            print(f"  No {component} embeddings found matching '{query}'")

    print("\n✅ Verification complete!")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate existing memories to include component embeddings")
    parser.add_argument("--dry-run", action="store_true", help="Analyze without making changes")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of memories to process at once")
    parser.add_argument("--verify", action="store_true", help="Verify migration results")

    args = parser.parse_args()

    if args.verify:
        verify_migration()
    else:
        # Check if embedding server is running
        print("Checking embedding server...")
        embedder = get_llama_embedder()

        try:
            test_embedding = embedder.encode(["test"], normalize_embeddings=True)
            if test_embedding is None or test_embedding.shape[0] == 0:
                print("\n⚠️  WARNING: Embedding server is not responding properly.")
                print("Please ensure the embedding server is running on port 8002.")
                response = input("Continue anyway? (y/n): ")
                if response.lower() != 'y':
                    print("Migration cancelled.")
                    return
        except Exception as e:
            print(f"\n⚠️  WARNING: Could not connect to embedding server: {e}")
            print("Please ensure the embedding server is running on port 8002.")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                return

        migrate_memory_components(dry_run=args.dry_run, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
