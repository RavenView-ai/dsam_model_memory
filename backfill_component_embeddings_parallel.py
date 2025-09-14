"""
Parallel backfill script to generate component embeddings with batched encoding.

This version batches all embedding requests for a memory together for maximum efficiency.
Can also regenerate components from memory content using the LLM extractor.
"""

import sqlite3
import json
import numpy as np
import time
import sys
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    def tqdm(iterable, total=None, desc=None):
        if desc:
            print(f"{desc}: {total} items")
        return iterable

from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.embedding import get_llama_embedder
from agentic_memory.config import cfg
from agentic_memory.extraction.llm_extractor import UnifiedExtractor
from agentic_memory.types import RawEvent


def regenerate_components(memory: Dict[str, Any], extractor: UnifiedExtractor) -> Dict[str, Any]:
    """
    Regenerate 5W1H components from memory content using the LLM extractor.

    Args:
        memory: Memory record with content
        extractor: LLM extractor instance

    Returns:
        Updated memory dict with new components
    """
    # Create a RawEvent from the memory content (using correct RawEvent fields)
    # Handle timestamp carefully - it might be None or invalid
    timestamp_str = memory.get('created_at')
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    raw_event = RawEvent(
        content=memory.get('raw_text', ''),  # RawEvent uses 'content' field
        session_id=memory.get('session_id', 'regeneration'),  # Required field
        event_type='user_message',  # Required field - assume user message for regeneration
        actor=memory.get('who_id', 'user'),  # Required field
        timestamp=timestamp
    )

    # Extract components using the LLM
    try:
        records = extractor.extract_memories(raw_event, max_parts=1)

        if records:
            # Take the first record (we forced single extraction)
            record = records[0]

            # Update memory with new components (using correct column names and field access)
            # Add specific error handling for each field
            try:
                memory['who_id'] = record.who.id if record.who else None
            except Exception as e:
                print(f"    Error with who_id: {e}, type={type(record.who)}")
                raise

            try:
                memory['who_list'] = record.who_list if record.who_list else None
            except Exception as e:
                print(f"    Error with who_list: {e}, type={type(record.who_list)}")
                raise

            try:
                memory['what'] = record.what
            except Exception as e:
                print(f"    Error with what: {e}, type={type(record.what)}")
                raise

            try:
                memory['when_ts'] = record.when.isoformat() if record.when else memory.get('when_ts')
            except Exception as e:
                print(f"    Error with when_ts: {e}, type={type(record.when)}")
                raise

            try:
                memory['when_list'] = record.when_list if record.when_list else None
            except Exception as e:
                print(f"    Error with when_list: {e}, type={type(record.when_list)}")
                raise

            try:
                memory['where_value'] = record.where.value if record.where else None
            except Exception as e:
                print(f"    Error with where_value: {e}, type={type(record.where)}")
                raise

            try:
                memory['where_list'] = record.where_list if record.where_list else None
            except Exception as e:
                print(f"    Error with where_list: {e}, type={type(record.where_list)}")
                raise

            try:
                memory['why'] = record.why
            except Exception as e:
                print(f"    Error with why: {e}, type={type(record.why)}")
                raise

            try:
                memory['how'] = record.how
            except Exception as e:
                print(f"    Error with how: {e}, type={type(record.how)}")
                raise

            return memory
    except Exception as e:
        import traceback
        print(f"  Warning: Failed to regenerate components for memory {memory['memory_id']}: {e}")
        traceback.print_exc()

    return memory


def update_memory_components(memory: Dict[str, Any], store: MemoryStore):
    """
    Update the database with regenerated components.

    Args:
        memory: Memory dict with updated components
        store: Memory store instance
    """
    with store.connect() as con:
        con.execute("""
            UPDATE memories
            SET who_id = ?, who_list = ?, what = ?,
                when_ts = ?, when_list = ?,
                where_value = ?, where_list = ?,
                why = ?, how = ?
            WHERE memory_id = ?
        """, (
            memory.get('who_id'),
            memory.get('who_list'),
            memory.get('what'),
            memory.get('when_ts'),
            memory.get('when_list'),
            memory.get('where_value'),
            memory.get('where_list'),
            memory.get('why'),
            memory.get('how'),
            memory['memory_id']
        ))
        con.commit()


def process_memory_parallel(memory: Dict[str, Any], store: MemoryStore, embedder: Any, index: FaissIndex):
    """
    Process a single memory's component embeddings using parallel/batched encoding.

    This function collects all strings that need embedding, encodes them in a single
    batch call, then stores all the results.
    """
    memory_id = memory['memory_id']

    # Collect all strings to embed and track their metadata
    strings_to_embed = []
    embedding_metadata = []  # List of (component_type, component_value) tuples

    # Collect WHO embeddings
    if memory['who_id']:
        strings_to_embed.append(memory['who_id'])
        embedding_metadata.append(('who', memory['who_id'], True))  # True = add to FAISS

    if memory['who_list']:
        try:
            who_entities = json.loads(memory['who_list'])
            for entity in who_entities:
                if entity and entity != memory['who_id']:
                    strings_to_embed.append(entity)
                    embedding_metadata.append(('who', entity, False))
        except (json.JSONDecodeError, TypeError):
            pass

    # Collect WHERE embeddings
    if memory['where_value']:
        strings_to_embed.append(memory['where_value'])
        embedding_metadata.append(('where', memory['where_value'], True))

    if memory['where_list']:
        try:
            where_entities = json.loads(memory['where_list'])
            for entity in where_entities:
                if entity and entity != memory['where_value']:
                    strings_to_embed.append(entity)
                    embedding_metadata.append(('where', entity, False))
        except (json.JSONDecodeError, TypeError):
            pass

    # Collect WHAT embeddings
    if memory['what']:
        try:
            what_entities = json.loads(memory['what']) if memory['what'].startswith('[') else [memory['what']]
            for entity in what_entities[:3]:  # Limit to first 3
                if entity:
                    strings_to_embed.append(entity)
                    embedding_metadata.append(('what', entity, False))
        except (json.JSONDecodeError, TypeError):
            pass

    # Skip if nothing to embed
    if not strings_to_embed:
        return

    # Batch encode all strings at once
    embeddings = embedder.encode(strings_to_embed, normalize_embeddings=True)

    # Store all embeddings
    for i, (component_type, component_value, add_to_faiss) in enumerate(embedding_metadata):
        embedding_vec = embeddings[i]

        # Store in database
        store.store_component_embedding(
            memory_id, component_type, component_value,
            embedding_vec.astype('float32').tobytes(), embedding_vec.shape[0]
        )

        # Add to FAISS if needed (only for primary who/where)
        if add_to_faiss:
            index.add(f"{component_type}:{memory_id}", embedding_vec)


def backfill_parallel(skip_existing: bool = True, batch_memories: int = 10, regenerate: bool = False):
    """
    Backfill component embeddings using parallel/batched encoding.

    Args:
        skip_existing: If True, skip memories that already have component embeddings
        batch_memories: Number of memories to process before saving FAISS index
        regenerate: If True, regenerate 5W1H components from content before embedding
    """
    print("=" * 80)
    print("PARALLEL COMPONENT EMBEDDINGS BACKFILL")
    if regenerate:
        print("WITH COMPONENT REGENERATION")
    print("=" * 80)

    # Initialize components
    store = MemoryStore(cfg.db_path)
    index = FaissIndex(cfg.embed_dim, cfg.index_path)
    embedder = get_llama_embedder()
    extractor = UnifiedExtractor() if regenerate else None

    # Get total count of memories
    with store.connect() as con:
        total_count = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        print(f"\nTotal memories in database: {total_count}")

        if skip_existing and not regenerate:
            existing_count = con.execute(
                "SELECT COUNT(DISTINCT memory_id) FROM component_embeddings"
            ).fetchone()[0]
            print(f"Memories with existing component embeddings: {existing_count}")
            print(f"Memories to process: {total_count - existing_count}")

    # Fetch memories (include content if regenerating)
    print("\nFetching memories from database...")
    with store.connect() as con:
        if regenerate:
            # Need raw_text for regeneration
            rows = con.execute("""
                SELECT memory_id, session_id, raw_text, who_id, who_list, where_value, where_list, what,
                       when_ts, when_list, why, how, source_event_id as source, created_at
                FROM memories
                ORDER BY created_at DESC
            """).fetchall()
        else:
            rows = con.execute("""
                SELECT memory_id, who_id, who_list, where_value, where_list, what
            FROM memories
            ORDER BY created_at DESC
        """).fetchall()

    print(f"Fetched {len(rows)} memories")

    # Get existing memory IDs to skip
    existing_memory_ids = set()
    if skip_existing:
        with store.connect() as con:
            existing = con.execute(
                "SELECT DISTINCT memory_id FROM component_embeddings"
            ).fetchall()
            existing_memory_ids = {row['memory_id'] for row in existing}
        print(f"Found {len(existing_memory_ids)} memories with existing embeddings")

    # Process memories
    processed = 0
    skipped = 0
    errors = 0
    regenerated = 0
    save_counter = 0

    with tqdm(total=len(rows), desc="Processing memories") as pbar:
        for row in rows:
            memory_id = row['memory_id']
            memory_dict = dict(row)

            # Check if we should skip (only skip if not regenerating)
            if skip_existing and not regenerate and memory_id in existing_memory_ids:
                skipped += 1
                pbar.update(1)
                continue

            # Regenerate components if requested
            if regenerate and extractor:
                try:
                    memory_dict = regenerate_components(memory_dict, extractor)
                    update_memory_components(memory_dict, store)
                    regenerated += 1
                except Exception as e:
                    print(f"\nError regenerating components for {memory_id}: {e}")

            # Process memory with parallel encoding
            try:
                process_memory_parallel(memory_dict, store, embedder, index)
                processed += 1
                save_counter += 1

                # Periodically save FAISS index
                if save_counter >= batch_memories:
                    index.save()
                    save_counter = 0

            except Exception as e:
                print(f"\nError processing memory {memory_id}: {e}")
                errors += 1

            pbar.update(1)

    # Final save of FAISS index
    print("\nSaving FAISS index...")
    index.save()

    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE")
    print(f"Processed: {processed} memories")
    if regenerate:
        print(f"Regenerated: {regenerated} memory components")
    print(f"Skipped: {skipped} memories (already had embeddings)")
    print(f"Errors: {errors}")
    print("=" * 80)


def process_memories_super_batch(
    memories: List[Dict[str, Any]],
    store: MemoryStore,
    embedder: Any,
    index: FaissIndex
):
    """
    Process multiple memories in a super-batch for maximum efficiency.

    This collects ALL strings from ALL memories and encodes them in one go.
    """
    # Collect all strings and metadata from all memories
    all_strings = []
    all_metadata = []  # List of (memory_id, component_type, component_value, add_to_faiss)

    for memory in memories:
        memory_id = memory['memory_id']

        # WHO embeddings
        if memory['who_id']:
            all_strings.append(memory['who_id'])
            all_metadata.append((memory_id, 'who', memory['who_id'], True))

        if memory['who_list']:
            try:
                who_entities = json.loads(memory['who_list'])
                for entity in who_entities:
                    if entity and entity != memory['who_id']:
                        all_strings.append(entity)
                        all_metadata.append((memory_id, 'who', entity, False))
            except (json.JSONDecodeError, TypeError):
                pass

        # WHERE embeddings
        if memory['where_value']:
            all_strings.append(memory['where_value'])
            all_metadata.append((memory_id, 'where', memory['where_value'], True))

        if memory['where_list']:
            try:
                where_entities = json.loads(memory['where_list'])
                for entity in where_entities:
                    if entity and entity != memory['where_value']:
                        all_strings.append(entity)
                        all_metadata.append((memory_id, 'where', entity, False))
            except (json.JSONDecodeError, TypeError):
                pass

        # WHAT embeddings
        if memory['what']:
            try:
                what_entities = json.loads(memory['what']) if memory['what'].startswith('[') else [memory['what']]
                for entity in what_entities[:3]:
                    if entity:
                        all_strings.append(entity)
                        all_metadata.append((memory_id, 'what', entity, False))
            except (json.JSONDecodeError, TypeError):
                pass

    if not all_strings:
        return

    # Encode everything at once
    print(f"  Encoding {len(all_strings)} strings in a single batch...")
    embeddings = embedder.encode(all_strings, normalize_embeddings=True)

    # Store all embeddings
    for i, (memory_id, component_type, component_value, add_to_faiss) in enumerate(all_metadata):
        embedding_vec = embeddings[i]

        store.store_component_embedding(
            memory_id, component_type, component_value,
            embedding_vec.astype('float32').tobytes(), embedding_vec.shape[0]
        )

        if add_to_faiss:
            index.add(f"{component_type}:{memory_id}", embedding_vec)


def backfill_super_batch(skip_existing: bool = True, super_batch_size: int = 100, regenerate: bool = False):
    """
    Backfill using super-batching - process many memories' embeddings in one encode call.

    Args:
        skip_existing: If True, skip memories that already have component embeddings
        super_batch_size: Number of memories to collect before batch encoding
        regenerate: If True, regenerate 5W1H components from content before embedding
    """
    print("=" * 80)
    print("SUPER-BATCH COMPONENT EMBEDDINGS BACKFILL")
    if regenerate:
        print("WITH COMPONENT REGENERATION")
    print("=" * 80)

    # Initialize components
    store = MemoryStore(cfg.db_path)
    index = FaissIndex(cfg.embed_dim, cfg.index_path)
    embedder = get_llama_embedder()
    extractor = UnifiedExtractor() if regenerate else None

    # Get memories to process
    with store.connect() as con:
        total_count = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        print(f"\nTotal memories in database: {total_count}")

        if regenerate:
            rows = con.execute("""
                SELECT memory_id, session_id, raw_text, who_id, who_list, where_value, where_list, what,
                       when_ts, when_list, why, how, source_event_id as source, created_at
                FROM memories
                ORDER BY created_at DESC
            """).fetchall()
        else:
            rows = con.execute("""
                SELECT memory_id, who_id, who_list, where_value, where_list, what
                FROM memories
                ORDER BY created_at DESC
            """).fetchall()

    # Filter existing if needed
    existing_memory_ids = set()
    if skip_existing:
        with store.connect() as con:
            existing = con.execute(
                "SELECT DISTINCT memory_id FROM component_embeddings"
            ).fetchall()
            existing_memory_ids = {row['memory_id'] for row in existing}

    # Process in super-batches
    batch = []
    processed = 0
    skipped = 0
    regenerated = 0

    with tqdm(total=len(rows), desc="Processing memories") as pbar:
        for row in rows:
            memory_dict = dict(row)

            if skip_existing and not regenerate_components and row['memory_id'] in existing_memory_ids:
                skipped += 1
                pbar.update(1)
                continue

            # Regenerate components if requested
            if regenerate and extractor:
                try:
                    memory_dict = regenerate_components(memory_dict, extractor)
                    update_memory_components(memory_dict, store)
                    regenerated += 1
                except Exception as e:
                    import traceback
                    print(f"\nError regenerating components for {memory_dict['memory_id']}: {e}")
                    # Print just the last few frames to find the issue
                    tb_lines = traceback.format_exc().split('\n')
                    for line in tb_lines[-10:]:
                        if line.strip():
                            print(f"  {line}")
                    break  # Stop after first error for debugging

            batch.append(memory_dict)

            # Process batch when full
            if len(batch) >= super_batch_size:
                try:
                    process_memories_super_batch(batch, store, embedder, index)
                    processed += len(batch)
                except Exception as e:
                    print(f"\nError processing batch: {e}")

                batch = []
                pbar.update(super_batch_size)

        # Process remaining
        if batch:
            try:
                process_memories_super_batch(batch, store, embedder, index)
                processed += len(batch)
            except Exception as e:
                print(f"\nError processing final batch: {e}")
            pbar.update(len(batch))

    # Save FAISS index
    print("\nSaving FAISS index...")
    index.save()

    print("\n" + "=" * 80)
    print("BACKFILL COMPLETE")
    print(f"Processed: {processed} memories")
    if regenerate:
        print(f"Regenerated: {regenerated} memory components")
    print(f"Skipped: {skipped} memories")
    print("=" * 80)


def verify_backfill():
    """Verify the backfill worked correctly."""
    store = MemoryStore(cfg.db_path)

    with store.connect() as con:
        total_components = con.execute(
            "SELECT COUNT(*) FROM component_embeddings"
        ).fetchone()[0]

        type_counts = con.execute("""
            SELECT component_type, COUNT(*) as count
            FROM component_embeddings
            GROUP BY component_type
        """).fetchall()

        samples = con.execute("""
            SELECT memory_id, component_type, component_value
            FROM component_embeddings
            LIMIT 10
        """).fetchall()

    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print(f"Total component embeddings: {total_components}")
    print("\nBreakdown by type:")
    for row in type_counts:
        print(f"  {row['component_type']}: {row['count']}")

    print("\nSample component embeddings:")
    for row in samples:
        print(f"  [{row['component_type']}] {row['memory_id']}: {row['component_value']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Parallel backfill for component embeddings with batched encoding"
    )
    parser.add_argument(
        '--mode',
        choices=['parallel', 'super-batch'],
        default='parallel',
        help="Processing mode: 'parallel' for per-memory batching, 'super-batch' for multi-memory batching"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help="Batch size (memories per batch in super-batch mode, saves per batch in parallel mode)"
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help="Don't skip existing embeddings"
    )
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help="Regenerate 5W1H components from memory content before embedding"
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help="Only verify existing embeddings"
    )

    args = parser.parse_args()

    if args.verify:
        verify_backfill()
    else:
        print(f"Starting {args.mode} backfill")
        print(f"Batch size: {args.batch_size}")
        print(f"Skip existing: {not args.no_skip}")
        if args.regenerate:
            print("Component regeneration: ENABLED")

        start_time = time.time()

        if args.mode == 'super-batch':
            backfill_super_batch(
                skip_existing=not args.no_skip,
                super_batch_size=args.batch_size,
                regenerate=args.regenerate
            )
        else:
            backfill_parallel(
                skip_existing=not args.no_skip,
                batch_memories=args.batch_size,
                regenerate=args.regenerate
            )

        elapsed = time.time() - start_time
        print(f"\nTotal time: {elapsed:.2f} seconds")

        # Run verification
        verify_backfill()