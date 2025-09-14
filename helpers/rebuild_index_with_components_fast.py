#!/usr/bin/env python3
"""
Optimized rebuild of FAISS index with batch processing for component embeddings.
Much faster than the original version by batching embedding generation.
"""

import os
import sys
import json
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agentic_memory.config import cfg
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.embedding.component_embedder import ComponentEmbedder
from agentic_memory.embedding import get_llama_embedder

class BatchComponentEmbedder(ComponentEmbedder):
    """Optimized component embedder that batches requests."""

    def embed_components_batch(self, memory_batch: list) -> dict:
        """Generate embeddings for all components in a batch of memories.

        Returns dict mapping memory_id to component embeddings.
        """
        # Collect all texts to embed in one batch
        who_texts = []
        where_texts = []
        when_texts = []
        what_texts = []
        why_texts = []
        how_texts = []

        # Track which memories have which components
        who_indices = {}
        where_indices = {}
        when_indices = {}
        what_indices = {}
        why_indices = {}
        how_indices = {}

        for i, memory in enumerate(memory_batch):
            memory_id = memory['memory_id']

            # Extract WHO text
            who_data = self._extract_who_data(memory)
            if who_data:
                who_text = self._format_who_text(who_data)
                who_texts.append(who_text)
                who_indices[memory_id] = len(who_texts) - 1

            # Extract WHERE text
            where_data = self._extract_where_data(memory)
            if where_data:
                where_text = self._format_where_text(where_data)
                where_texts.append(where_text)
                where_indices[memory_id] = len(where_texts) - 1

            # Extract WHEN text
            when_data = memory.get('when_list') or memory.get('when_ts')
            if when_data:
                when_text = self._format_when_text(when_data)
                when_texts.append(when_text)
                when_indices[memory_id] = len(when_texts) - 1

            # Extract WHAT text
            what_data = memory.get('what')
            if what_data:
                what_texts.append(what_data)
                what_indices[memory_id] = len(what_texts) - 1

            # Extract WHY text
            why_data = memory.get('why')
            if why_data and why_data != 'unknown':
                why_texts.append(f"Reason: {why_data}")
                why_indices[memory_id] = len(why_texts) - 1

            # Extract HOW text
            how_data = memory.get('how')
            if how_data and how_data != 'unknown':
                how_texts.append(f"Method: {how_data}")
                how_indices[memory_id] = len(how_texts) - 1

        # Batch encode all texts at once
        all_embeddings = {}

        if who_texts:
            who_embeddings = self.embedder.encode(who_texts, batch_size=32, normalize_embeddings=True)
            for memory_id, idx in who_indices.items():
                if memory_id not in all_embeddings:
                    all_embeddings[memory_id] = {}
                all_embeddings[memory_id]['who'] = who_embeddings[idx]

        if where_texts:
            where_embeddings = self.embedder.encode(where_texts, batch_size=32, normalize_embeddings=True)
            for memory_id, idx in where_indices.items():
                if memory_id not in all_embeddings:
                    all_embeddings[memory_id] = {}
                all_embeddings[memory_id]['where'] = where_embeddings[idx]

        if when_texts:
            when_embeddings = self.embedder.encode(when_texts, batch_size=32, normalize_embeddings=True)
            for memory_id, idx in when_indices.items():
                if memory_id not in all_embeddings:
                    all_embeddings[memory_id] = {}
                all_embeddings[memory_id]['when'] = when_embeddings[idx]

        if what_texts:
            what_embeddings = self.embedder.encode(what_texts, batch_size=32, normalize_embeddings=True)
            for memory_id, idx in what_indices.items():
                if memory_id not in all_embeddings:
                    all_embeddings[memory_id] = {}
                all_embeddings[memory_id]['what'] = what_embeddings[idx]

        if why_texts:
            why_embeddings = self.embedder.encode(why_texts, batch_size=32, normalize_embeddings=True)
            for memory_id, idx in why_indices.items():
                if memory_id not in all_embeddings:
                    all_embeddings[memory_id] = {}
                all_embeddings[memory_id]['why'] = why_embeddings[idx]

        if how_texts:
            how_embeddings = self.embedder.encode(how_texts, batch_size=32, normalize_embeddings=True)
            for memory_id, idx in how_indices.items():
                if memory_id not in all_embeddings:
                    all_embeddings[memory_id] = {}
                all_embeddings[memory_id]['how'] = how_embeddings[idx]

        return all_embeddings

    def _format_who_text(self, who_data):
        """Format WHO data into text for embedding."""
        if isinstance(who_data, dict):
            parts = []
            if who_data.get('label'):
                parts.append(who_data['label'])
            if who_data.get('id') and who_data['id'] != who_data.get('label'):
                parts.append(who_data['id'])
            if who_data.get('type'):
                parts.append(f"type:{who_data['type']}")
            return f"Actor: {' '.join(parts)}"
        elif isinstance(who_data, list):
            return f"Actor: {' '.join(str(w) for w in who_data if w)}"
        else:
            return f"Actor: {who_data}"

    def _format_where_text(self, where_data):
        """Format WHERE data into text for embedding."""
        if isinstance(where_data, dict):
            parts = []
            if where_data.get('value') and where_data['value'] != 'unknown':
                parts.append(where_data['value'])
            if where_data.get('type'):
                parts.append(f"type:{where_data['type']}")
            return f"Location: {' '.join(parts)}"
        elif isinstance(where_data, list):
            return f"Location: {' '.join(str(w) for w in where_data if w and w != 'unknown')}"
        else:
            return f"Location: {where_data}"

    def _format_when_text(self, when_data):
        """Format WHEN data into text for embedding."""
        if isinstance(when_data, str):
            return f"Time: {self._parse_temporal_context(when_data)}"
        elif isinstance(when_data, list):
            contexts = [self._parse_temporal_context(str(ts)) for ts in when_data if ts]
            return f"Time: {' '.join(contexts)}"
        return f"Time: {when_data}"

def rebuild_index_fast(backup: bool = True, batch_size: int = 100):
    """
    Fast rebuild of FAISS index using batch processing.

    Args:
        backup: If True, backup the existing index before rebuilding
        batch_size: Number of memories to process at once
    """
    print("=" * 60)
    print("FAST REBUILD FAISS INDEX WITH COMPONENTS")
    print("=" * 60)

    start_time = time.time()

    # Backup existing index if requested
    if backup and os.path.exists(cfg.index_path):
        backup_path = f"{cfg.index_path}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\nBacking up existing index to: {backup_path}")
        shutil.copy2(cfg.index_path, backup_path)

    # Initialize components
    print("\nInitializing components...")
    store = MemoryStore(cfg.db_path)
    component_embedder = BatchComponentEmbedder()
    embedder = get_llama_embedder()

    # Create new index
    print("Creating new FAISS index...")
    embed_dim = embedder.embedding_dimension
    print(f"Using embedding dimension: {embed_dim}")
    index = FaissIndex(embed_dim, cfg.index_path)
    index.reset()

    # Get all memories
    con = sqlite3.connect(cfg.db_path)
    con.row_factory = sqlite3.Row

    # Ensure embeddings table exists
    con.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            memory_id TEXT PRIMARY KEY,
            dim INTEGER NOT NULL,
            vector BLOB NOT NULL
        )
    """)
    con.commit()

    count_query = "SELECT COUNT(*) as count FROM memories"
    total_count = con.execute(count_query).fetchone()['count']

    print(f"\nFound {total_count} memories to index")

    # Statistics
    stats = {
        'main_embeddings': 0,
        'who': 0,
        'where': 0,
        'when': 0,
        'what': 0,
        'why': 0,
        'how': 0,
        'errors': 0,
        'missing_embeddings': 0
    }

    # Process in batches
    offset = 0
    pbar = tqdm(total=total_count, desc="Processing memories")

    while offset < total_count:
        batch_start = time.time()

        # Fetch batch of memories
        query = """
            SELECT m.memory_id, e.vector as embedding, m.raw_text,
                   m.who_id, m.who_label, m.who_type, m.who_list,
                   m.what, m.when_ts, m.when_list,
                   m.where_value, m.where_type, m.where_lat, m.where_lon, m.where_list,
                   m.why, m.how
            FROM memories m
            LEFT JOIN embeddings e ON m.memory_id = e.memory_id
            ORDER BY m.created_at
            LIMIT ? OFFSET ?
        """

        rows = con.execute(query, (batch_size, offset)).fetchall()
        memory_batch = [dict(row) for row in rows]

        # Process main embeddings
        missing_embeddings = []
        for memory in memory_batch:
            memory_id = memory['memory_id']

            try:
                # Add main embedding
                if memory['embedding']:
                    embedding = np.frombuffer(memory['embedding'], dtype=np.float32)
                    index.add(memory_id, embedding)
                    stats['main_embeddings'] += 1
                elif memory['raw_text']:
                    # Track for batch generation
                    missing_embeddings.append(memory)
                else:
                    stats['missing_embeddings'] += 1
            except Exception as e:
                stats['errors'] += 1
                if stats['errors'] <= 5:
                    print(f"\nError processing main embedding for {memory_id}: {e}")

        # Batch generate missing main embeddings
        if missing_embeddings:
            texts = [m['raw_text'] for m in missing_embeddings]
            try:
                embeddings = embedder.encode(texts, batch_size=32, normalize_embeddings=True)
                for i, memory in enumerate(missing_embeddings):
                    if embeddings[i] is not None:
                        index.add(memory['memory_id'], embeddings[i])
                        stats['main_embeddings'] += 1
                        # Update database
                        con.execute(
                            "INSERT OR REPLACE INTO embeddings (memory_id, dim, vector) VALUES (?, ?, ?)",
                            (memory['memory_id'], len(embeddings[i]), embeddings[i].tobytes())
                        )
            except Exception as e:
                print(f"\nError batch generating main embeddings: {e}")
                stats['errors'] += len(missing_embeddings)

        # Batch generate component embeddings
        try:
            component_embeddings = component_embedder.embed_components_batch(memory_batch)

            # Add to index
            for memory_id, components in component_embeddings.items():
                if components.get('who') is not None:
                    index.add(f"who:{memory_id}", components['who'])
                    stats['who'] += 1

                if components.get('where') is not None:
                    index.add(f"where:{memory_id}", components['where'])
                    stats['where'] += 1

                if components.get('when') is not None:
                    index.add(f"when:{memory_id}", components['when'])
                    stats['when'] += 1

                if components.get('what') is not None:
                    index.add(f"what:{memory_id}", components['what'])
                    stats['what'] += 1

                if components.get('why') is not None:
                    index.add(f"why:{memory_id}", components['why'])
                    stats['why'] += 1

                if components.get('how') is not None:
                    index.add(f"how:{memory_id}", components['how'])
                    stats['how'] += 1

        except Exception as e:
            print(f"\nError batch generating component embeddings: {e}")
            stats['errors'] += len(memory_batch)

        # Update progress
        pbar.update(len(rows))

        # Show batch timing
        batch_time = time.time() - batch_start
        if offset % (batch_size * 10) == 0 and offset > 0:
            pbar.set_postfix({'batch_time': f'{batch_time:.2f}s', 'per_item': f'{batch_time/len(rows):.3f}s'})

        # Commit database changes and save index periodically
        if offset % (batch_size * 10) == 0:
            con.commit()
            index.save()

        offset += batch_size

    pbar.close()

    # Final commit and close
    con.commit()
    con.close()

    # Save the index
    print("\nSaving FAISS index...")
    index.save()

    # Calculate total time
    total_time = time.time() - start_time

    # Save rebuild status
    rebuild_status = {
        'rebuild_date': datetime.now().isoformat(),
        'total_memories': total_count,
        'main_embeddings': stats['main_embeddings'],
        'who': stats['who'],
        'where': stats['where'],
        'when': stats['when'],
        'what': stats['what'],
        'why': stats['why'],
        'how': stats['how'],
        'missing_embeddings': stats['missing_embeddings'],
        'errors': stats['errors'],
        'total_time_seconds': total_time,
        'memories_per_second': total_count / total_time if total_time > 0 else 0
    }

    status_file = Path(cfg.db_path).parent / '.index_rebuild_status.json'
    with open(status_file, 'w') as f:
        json.dump(rebuild_status, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("REBUILD SUMMARY")
    print("=" * 60)
    print(f"Total memories processed: {total_count}")
    print(f"Main embeddings indexed: {stats['main_embeddings']}")
    print(f"WHO embeddings indexed: {stats['who']}")
    print(f"WHERE embeddings indexed: {stats['where']}")
    print(f"WHEN embeddings indexed: {stats['when']}")
    print(f"WHAT embeddings indexed: {stats['what']}")
    print(f"WHY embeddings indexed: {stats['why']}")
    print(f"HOW embeddings indexed: {stats['how']}")

    if stats['missing_embeddings'] > 0:
        print(f"\n⚠️  Missing embeddings: {stats['missing_embeddings']}")
        print("   (These memories had no raw_text to generate embeddings from)")

    if stats['errors'] > 0:
        print(f"\n⚠️  Errors encountered: {stats['errors']}")

    print(f"\n⏱️  Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"📊 Processing speed: {total_count/total_time:.2f} memories/second")

    print("\n✅ Index rebuild completed successfully!")
    print(f"Index saved to: {cfg.index_path}")

    return stats

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fast rebuild of FAISS index with component embeddings")
    parser.add_argument("--no-backup", action="store_true", help="Skip backing up existing index")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of memories to process at once")

    args = parser.parse_args()

    # Check if embedding server is running
    print("Checking embedding server...")
    embedder = get_llama_embedder()

    try:
        test_embedding = embedder.encode(["test"], normalize_embeddings=True)
        if test_embedding is None or test_embedding.shape[0] == 0:
            print("\n⚠️  WARNING: Embedding server is not responding properly.")
            print("Please ensure the embedding server is running on port 8002.")
            print("You can start it with: python -m agentic_memory.cli server start --llm")
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Rebuild cancelled.")
                return
    except Exception as e:
        print(f"\n⚠️  WARNING: Could not connect to embedding server: {e}")
        print("Please ensure the embedding server is running on port 8002.")
        print("You can start it with: python -m agentic_memory.cli server start --llm")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Rebuild cancelled.")
            return

    # Confirm rebuild
    if not args.no_backup:
        print("\n⚠️  This will rebuild the entire FAISS index from scratch.")
        print("The existing index will be backed up first.")
    else:
        print("\n⚠️  WARNING: This will rebuild the entire FAISS index from scratch WITHOUT backup!")

    response = input("\nProceed with fast rebuild? (y/n): ")
    if response.lower() != 'y':
        print("Rebuild cancelled.")
        return

    rebuild_index_fast(backup=not args.no_backup, batch_size=args.batch_size)

if __name__ == "__main__":
    main()