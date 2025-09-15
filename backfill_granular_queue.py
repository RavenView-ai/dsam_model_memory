"""
Granular queue-based backfill that processes each component as a separate task.

This version creates individual queue tasks for:
- Each WHO entity embedding
- Each WHERE location embedding
- Each WHAT concept embedding

This maximizes parallelization and allows fine-grained error recovery.
"""

import time
import json
import random
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed")
    def tqdm(iterable, total=None, desc=None):
        if desc:
            print(f"{desc}: {total} items")
        return iterable

from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.config import cfg
from agentic_memory.extraction.llm_extractor import UnifiedExtractor
from agentic_memory.types import RawEvent
from agentic_memory.queue_handler import (
    MemoryQueueHandler,
    TaskPriority,
    get_queue_handler
)
from agentic_memory.embedding import get_llama_embedder


class GranularEmbeddingProcessor:
    """
    Processor that handles individual component embedding tasks.
    """

    def __init__(self, store: MemoryStore, index: FaissIndex):
        self.store = store
        self.index = index
        self.embedder = get_llama_embedder()
        self.queue = get_queue_handler()

        # Register processors for each component type
        self.queue.register_processor('embed_who', self._process_who_embedding)
        self.queue.register_processor('embed_where', self._process_where_embedding)
        self.queue.register_processor('embed_what', self._process_what_embedding)

    def _process_who_embedding(self, payload: Dict[str, Any]):
        """Process a single WHO component embedding."""
        memory_id = payload['memory_id']
        who_value = payload['value']
        is_primary = payload.get('is_primary', False)

        # Generate embedding
        embedding = self.embedder.encode([who_value], normalize_embeddings=True)[0]

        # Store in database
        self.store.store_component_embedding(
            memory_id, 'who', who_value,
            embedding.astype('float32').tobytes(),
            embedding.shape[0]
        )

        # Add to FAISS if primary
        if is_primary:
            self.index.add(f"who:{memory_id}", embedding)

        return {'status': 'success', 'component': 'who', 'value': who_value}

    def _process_where_embedding(self, payload: Dict[str, Any]):
        """Process a single WHERE component embedding."""
        memory_id = payload['memory_id']
        where_value = payload['value']
        is_primary = payload.get('is_primary', False)

        # Generate embedding
        embedding = self.embedder.encode([where_value], normalize_embeddings=True)[0]

        # Store in database
        self.store.store_component_embedding(
            memory_id, 'where', where_value,
            embedding.astype('float32').tobytes(),
            embedding.shape[0]
        )

        # Add to FAISS if primary
        if is_primary:
            self.index.add(f"where:{memory_id}", embedding)

        return {'status': 'success', 'component': 'where', 'value': where_value}

    def _process_what_embedding(self, payload: Dict[str, Any]):
        """Process a single WHAT component embedding."""
        memory_id = payload['memory_id']
        what_value = payload['value']

        # Generate embedding
        embedding = self.embedder.encode([what_value], normalize_embeddings=True)[0]

        # Store in database
        self.store.store_component_embedding(
            memory_id, 'what', what_value,
            embedding.astype('float32').tobytes(),
            embedding.shape[0]
        )

        return {'status': 'success', 'component': 'what', 'value': what_value}

    def queue_component(
        self,
        component_type: str,
        memory_id: str,
        value: str,
        is_primary: bool = False,
        priority: TaskPriority = TaskPriority.LOW
    ) -> str:
        """Queue a single component for embedding."""
        task_type = f'embed_{component_type}'
        payload = {
            'memory_id': memory_id,
            'value': value,
            'is_primary': is_primary
        }

        return self.queue.enqueue(
            task_type=task_type,
            payload=payload,
            priority=priority,
            task_id=f"{memory_id}_{component_type}_{value[:20]}"
        )


def extract_and_queue_granular(
    memory: Dict[str, Any],
    extractor: UnifiedExtractor,
    processor: GranularEmbeddingProcessor,
    store: MemoryStore,
    regenerate: bool = False,
    sample_rate: float = 0.01  # Sample 1% for verification
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """
    Extract components from a memory and queue each one individually.

    Returns:
        Tuple of (number of tasks queued, optional sample data for verification)
    """
    memory_id = memory['memory_id']
    tasks_queued = 0
    sample_data = None

    # Determine if this memory should be sampled
    should_sample = random.random() < sample_rate

    # Skip if not regenerating and already has components
    if not regenerate and memory.get('who_id'):
        # Just queue embeddings for existing components
        if memory.get('who_id'):
            processor.queue_component('who', memory_id, memory['who_id'], is_primary=True)
            tasks_queued += 1

        if memory.get('who_list'):
            try:
                who_entities = json.loads(memory['who_list'])
                for entity in who_entities:
                    if entity and entity != memory.get('who_id'):
                        processor.queue_component('who', memory_id, entity)
                        tasks_queued += 1
            except (json.JSONDecodeError, TypeError):
                pass

        if memory.get('where_value'):
            processor.queue_component('where', memory_id, memory['where_value'], is_primary=True)
            tasks_queued += 1

        if memory.get('where_list'):
            try:
                where_entities = json.loads(memory['where_list'])
                for entity in where_entities:
                    if entity and entity != memory.get('where_value'):
                        processor.queue_component('where', memory_id, entity)
                        tasks_queued += 1
            except (json.JSONDecodeError, TypeError):
                pass

        if memory.get('what'):
            try:
                what_entities = json.loads(memory['what']) if memory['what'].startswith('[') else [memory['what']]
                for entity in what_entities[:3]:  # Limit to 3
                    if entity:
                        processor.queue_component('what', memory_id, entity)
                        tasks_queued += 1
            except (json.JSONDecodeError, TypeError):
                pass

        # If sampling, verify existing components
        if should_sample:
            sample_data = verify_existing_components(memory, store)

        return tasks_queued, sample_data

    # Extract new components if regenerating
    timestamp_str = memory.get('created_at')
    if timestamp_str:
        try:
            timestamp = datetime.fromisoformat(timestamp_str)
        except (ValueError, TypeError):
            timestamp = datetime.now()
    else:
        timestamp = datetime.now()

    raw_event = RawEvent(
        content=memory.get('raw_text', ''),
        session_id=memory.get('session_id', 'regeneration'),
        event_type='user_message',
        actor=memory.get('who_id', 'user'),
        timestamp=timestamp
    )

    # Extract components using LLM
    try:
        records = extractor.extract_memories(raw_event, max_parts=1)

        if records:
            record = records[0]

            # Queue WHO components
            if record.who and record.who.id:
                processor.queue_component('who', memory_id, record.who.id, is_primary=True)
                tasks_queued += 1

            if record.who_list:
                try:
                    who_entities = json.loads(record.who_list)
                    for entity in who_entities:
                        if entity and entity != (record.who.id if record.who else None):
                            processor.queue_component('who', memory_id, entity)
                            tasks_queued += 1
                except (json.JSONDecodeError, TypeError):
                    pass

            # Queue WHERE components
            if record.where and record.where.value:
                processor.queue_component('where', memory_id, record.where.value, is_primary=True)
                tasks_queued += 1

            if record.where_list:
                try:
                    where_entities = json.loads(record.where_list)
                    for entity in where_entities:
                        if entity and entity != (record.where.value if record.where else None):
                            processor.queue_component('where', memory_id, entity)
                            tasks_queued += 1
                except (json.JSONDecodeError, TypeError):
                    pass

            # Queue WHAT components
            if record.what:
                try:
                    what_text = record.what
                    what_entities = json.loads(what_text) if what_text.startswith('[') else [what_text]
                    for entity in what_entities[:3]:
                        if entity:
                            processor.queue_component('what', memory_id, entity)
                            tasks_queued += 1
                except (json.JSONDecodeError, TypeError):
                    pass

            # Update database with extracted components
            update_memory_components(memory, record, store)

            # If sampling, create verification data
            if should_sample:
                sample_data = {
                    'memory_id': memory_id,
                    'raw_text': memory.get('raw_text', '')[:200],  # First 200 chars
                    'extraction': {
                        'who': record.who.id if record.who else None,
                        'who_list': record.who_list,
                        'where': record.where.value if record.where else None,
                        'where_list': record.where_list,
                        'what': record.what[:100] if record.what else None,  # First 100 chars
                        'when': record.when.isoformat() if record.when else None,
                        'why': record.why[:100] if record.why else None,
                        'how': record.how[:100] if record.how else None
                    },
                    'components_queued': tasks_queued
                }

    except Exception as e:
        print(f"Error extracting from {memory_id}: {e}")
        if should_sample:
            sample_data = {
                'memory_id': memory_id,
                'error': str(e),
                'components_queued': 0
            }

    return tasks_queued, sample_data


def verify_existing_components(memory: Dict[str, Any], store: MemoryStore) -> Dict[str, Any]:
    """Verify that existing components are properly stored."""
    memory_id = memory['memory_id']
    verification = {
        'memory_id': memory_id,
        'raw_text': memory.get('raw_text', '')[:200],
        'existing_components': {
            'who': memory.get('who_id'),
            'who_list': memory.get('who_list'),
            'where': memory.get('where_value'),
            'where_list': memory.get('where_list'),
            'what': memory.get('what')[:100] if memory.get('what') else None
        },
        'embeddings_found': {}
    }

    # Check for embeddings in database
    with store.connect() as con:
        embeddings = con.execute("""
            SELECT component_type, component_value, dim
            FROM component_embeddings
            WHERE memory_id = ?
        """, (memory_id,)).fetchall()

        for row in embeddings:
            comp_type = row['component_type']
            if comp_type not in verification['embeddings_found']:
                verification['embeddings_found'][comp_type] = []
            verification['embeddings_found'][comp_type].append({
                'value': row['component_value'][:50],  # First 50 chars
                'dim': row['dim']
            })

    return verification


def verify_embedding_generation(
    memory_id: str,
    component_type: str,
    component_value: str,
    store: MemoryStore
) -> bool:
    """Verify that an embedding was generated and stored."""
    with store.connect() as con:
        result = con.execute("""
            SELECT dim
            FROM component_embeddings
            WHERE memory_id = ? AND component_type = ? AND component_value = ?
        """, (memory_id, component_type, component_value)).fetchone()

        return result is not None and result['dim'] > 0


def update_memory_components(memory: Dict[str, Any], record, store: MemoryStore):
    """Update database with extracted components."""
    with store.connect() as con:
        con.execute("""
            UPDATE memories
            SET who_id = ?, who_list = ?, what = ?,
                when_ts = ?, when_list = ?,
                where_value = ?, where_list = ?,
                why = ?, how = ?
            WHERE memory_id = ?
        """, (
            record.who.id if record.who else None,
            record.who_list if record.who_list else None,
            record.what,
            record.when.isoformat() if record.when else memory.get('when_ts'),
            record.when_list if record.when_list else None,
            record.where.value if record.where else None,
            record.where_list if record.where_list else None,
            record.why,
            record.how,
            memory['memory_id']
        ))
        con.commit()


def save_checkpoint(checkpoint_file: str, data: Dict[str, Any]):
    """Save checkpoint data to file."""
    temp_file = checkpoint_file + '.tmp'
    with open(temp_file, 'wb') as f:
        pickle.dump(data, f)
    # Atomic rename
    os.replace(temp_file, checkpoint_file)
    print(f"Checkpoint saved: {data.get('processed_count', 0)} memories processed")


def load_checkpoint(checkpoint_file: str) -> Optional[Dict[str, Any]]:
    """Load checkpoint data from file if it exists."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            print(f"Checkpoint loaded: {data.get('processed_count', 0)} memories already processed")
            return data
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return None


def backfill_granular(
    skip_existing: bool = True,
    regenerate: bool = False,
    batch_size: int = 100,
    max_workers: int = 8,
    checkpoint_file: str = "backfill_checkpoint.pkl",
    resume: bool = True
):
    """
    Main backfill using granular component queuing with checkpointing.

    Each component (WHO, WHERE, WHAT) becomes its own queue task,
    allowing maximum parallelization. Supports resuming from checkpoint.

    Args:
        skip_existing: Skip memories that already have embeddings
        regenerate: Regenerate components from raw_text using LLM
        batch_size: Number of memories to process before saving checkpoint
        max_workers: Number of parallel workers
        checkpoint_file: Path to checkpoint file
        resume: Whether to resume from checkpoint if it exists
    """
    print("=" * 80)
    print("GRANULAR QUEUE-BASED COMPONENT BACKFILL")
    print("=" * 80)
    print(f"Workers: {max_workers}")
    print(f"Checkpoint file: {checkpoint_file}")
    print(f"Resume from checkpoint: {resume}")
    print(f"Each component will be processed as a separate task")
    print("=" * 80)

    # Initialize components
    store = MemoryStore(cfg.db_path)
    index = FaissIndex(cfg.embed_dim, cfg.index_path)
    extractor = UnifiedExtractor() if regenerate else None

    # Create processor
    processor = GranularEmbeddingProcessor(store, index)

    # Configure and start queue
    processor.queue.max_workers = max_workers
    processor.queue.start()

    # Load checkpoint if resuming
    checkpoint_data = None
    processed_memory_ids = set()

    if resume:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data:
            processed_memory_ids = set(checkpoint_data.get('processed_ids', []))
            print(f"Will skip {len(processed_memory_ids)} already processed memories")

    # Get memories
    print("\nFetching memories from database...")
    with store.connect() as con:
        rows = con.execute("""
            SELECT memory_id, session_id, raw_text, who_id, who_list,
                   where_value, where_list, what, when_ts, when_list,
                   why, how, created_at
            FROM memories
            ORDER BY created_at DESC
        """).fetchall()

    print(f"Fetched {len(rows)} memories")

    # Filter if needed
    memories_to_process = []
    if skip_existing and not regenerate:
        with store.connect() as con:
            existing = con.execute(
                "SELECT DISTINCT memory_id FROM component_embeddings"
            ).fetchall()
            existing_ids = {row['memory_id'] for row in existing}

            memories_to_process = [
                dict(row) for row in rows
                if row['memory_id'] not in existing_ids
                and row['memory_id'] not in processed_memory_ids  # Skip checkpoint memories
            ]
    else:
        memories_to_process = [
            dict(row) for row in rows
            if row['memory_id'] not in processed_memory_ids  # Skip checkpoint memories
        ]

    print(f"Memories to process: {len(memories_to_process)}")
    if processed_memory_ids:
        print(f"Skipping {len(processed_memory_ids)} memories from checkpoint")

    # Queue all component tasks with sampling and checkpointing
    total_queued = 0
    samples_collected = []
    sample_rate = 0.02  # 2% sampling rate
    processed_in_session = 0
    checkpoint_counter = 0

    print(f"\nQueueing component tasks (sampling {sample_rate*100:.1f}% for verification)...")
    print(f"Checkpoint will be saved every {batch_size} memories")

    try:
        with tqdm(total=len(memories_to_process), desc="Extracting & queueing") as pbar:
            for i in range(0, len(memories_to_process), batch_size):
                batch = memories_to_process[i:i+batch_size]
                batch_processed_ids = []

                for memory in batch:
                    tasks, sample = extract_and_queue_granular(
                        memory, extractor, processor, store, regenerate, sample_rate
                    )
                    total_queued += tasks
                    if sample:
                        samples_collected.append(sample)

                    batch_processed_ids.append(memory['memory_id'])
                    processed_memory_ids.add(memory['memory_id'])
                    processed_in_session += 1
                    pbar.update(1)

                # Save checkpoint after each batch
                checkpoint_counter += 1
                if checkpoint_counter % 1 == 0:  # Save every batch
                    checkpoint_data = {
                        'processed_ids': list(processed_memory_ids),
                        'processed_count': len(processed_memory_ids),
                        'total_queued': total_queued,
                        'samples_collected': samples_collected[-100:],  # Keep last 100 samples
                        'timestamp': datetime.now().isoformat()
                    }
                    save_checkpoint(checkpoint_file, checkpoint_data)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        checkpoint_data = {
            'processed_ids': list(processed_memory_ids),
            'processed_count': len(processed_memory_ids),
            'total_queued': total_queued,
            'samples_collected': samples_collected,
            'timestamp': datetime.now().isoformat()
        }
        save_checkpoint(checkpoint_file, checkpoint_data)
        print(f"Checkpoint saved with {len(processed_memory_ids)} processed memories")
        print("You can resume by running the script again with --resume")
        processor.queue.stop()
        return

    print(f"\nQueued {total_queued} component embedding tasks")
    if samples_collected:
        print(f"Collected {len(samples_collected)} samples for verification")
    print(f"Processed {processed_in_session} memories in this session")

    # Monitor progress
    if total_queued > 0:
        print("\nProcessing embeddings...")
        last_completed = 0

        with tqdm(total=total_queued, desc="Embedding components") as pbar:
            while True:
                status = processor.queue.get_status()

                # Update progress
                current_completed = status['completed_tasks']
                if current_completed > last_completed:
                    pbar.update(current_completed - last_completed)
                    last_completed = current_completed

                # Check if done
                if status['queue_size'] == 0 and status['active_tasks'] == 0:
                    if current_completed > last_completed:
                        pbar.update(current_completed - last_completed)
                    break

                time.sleep(1)

        # Save FAISS index
        print("\nSaving FAISS index...")
        index.save()

        # Verify samples if collected
        if samples_collected:
            print("\n" + "=" * 80)
            print("VERIFICATION SAMPLING RESULTS")
            print("=" * 80)
            verify_samples(samples_collected, store)

        # Print summary
        final_status = processor.queue.get_status()
        print("\n" + "=" * 80)
        print("BACKFILL COMPLETE")
        print("=" * 80)
        print(f"Total tasks processed: {final_status['completed_tasks']}")
        print(f"Failed tasks: {final_status['failed_tasks']}")
        print(f"Average time per task: {final_status['stats']['avg_processing_time']:.3f}s")

        # Estimate components per memory
        if len(memories_to_process) > 0:
            avg_components = total_queued / len(memories_to_process)
            print(f"Average components per memory: {avg_components:.1f}")

    # Stop queue
    processor.queue.stop()

    # Clean up checkpoint file on successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\nCheckpoint file removed (backfill completed successfully)")


def verify_samples(samples: List[Dict[str, Any]], store: MemoryStore):
    """Analyze and report on verification samples."""
    print(f"\nVerifying {len(samples)} sampled memories...")

    extraction_stats = {
        'total': len(samples),
        'with_who': 0,
        'with_where': 0,
        'with_what': 0,
        'with_errors': 0,
        'avg_components': []
    }

    embedding_stats = {
        'memories_checked': 0,
        'embeddings_verified': 0,
        'missing_embeddings': []
    }

    for sample in samples:
        if 'error' in sample:
            extraction_stats['with_errors'] += 1
            continue

        # Check extraction quality
        if 'extraction' in sample:
            ext = sample['extraction']
            if ext.get('who'):
                extraction_stats['with_who'] += 1
            if ext.get('where'):
                extraction_stats['with_where'] += 1
            if ext.get('what'):
                extraction_stats['with_what'] += 1

            # Count total components
            component_count = 0
            if ext.get('who'):
                component_count += 1
            if ext.get('who_list'):
                try:
                    component_count += len(json.loads(ext['who_list']))
                except:
                    pass
            if ext.get('where'):
                component_count += 1
            if ext.get('where_list'):
                try:
                    component_count += len(json.loads(ext['where_list']))
                except:
                    pass
            if ext.get('what'):
                try:
                    what_items = json.loads(ext['what']) if ext['what'].startswith('[') else [ext['what']]
                    component_count += min(len(what_items), 3)
                except:
                    component_count += 1

            extraction_stats['avg_components'].append(component_count)

        # Check embeddings (for existing components)
        if 'embeddings_found' in sample:
            embedding_stats['memories_checked'] += 1
            if sample['embeddings_found']:
                embedding_stats['embeddings_verified'] += 1
            else:
                embedding_stats['missing_embeddings'].append(sample['memory_id'])

    # Print extraction statistics
    print("\n--- Extraction Quality ---")
    if extraction_stats['total'] > 0:
        print(f"Total samples: {extraction_stats['total']}")
        print(f"With WHO: {extraction_stats['with_who']} ({extraction_stats['with_who']/extraction_stats['total']*100:.1f}%)")
        print(f"With WHERE: {extraction_stats['with_where']} ({extraction_stats['with_where']/extraction_stats['total']*100:.1f}%)")
        print(f"With WHAT: {extraction_stats['with_what']} ({extraction_stats['with_what']/extraction_stats['total']*100:.1f}%)")
        if extraction_stats['with_errors'] > 0:
            print(f"Extraction errors: {extraction_stats['with_errors']} ({extraction_stats['with_errors']/extraction_stats['total']*100:.1f}%)")

        if extraction_stats['avg_components']:
            avg_comp = np.mean(extraction_stats['avg_components'])
            print(f"Average components per memory: {avg_comp:.2f}")

    # Print embedding statistics
    print("\n--- Embedding Verification ---")
    if embedding_stats['memories_checked'] > 0:
        print(f"Memories with existing components checked: {embedding_stats['memories_checked']}")
        print(f"Memories with embeddings verified: {embedding_stats['embeddings_verified']}")
        if embedding_stats['missing_embeddings']:
            print(f"Missing embeddings for {len(embedding_stats['missing_embeddings'])} memories")
            if len(embedding_stats['missing_embeddings']) <= 5:
                print(f"  Memory IDs: {', '.join(embedding_stats['missing_embeddings'][:5])}")

    # Sample some extraction details
    print("\n--- Sample Extraction Details (first 3) ---")
    detail_count = 0
    for sample in samples[:3]:
        if 'extraction' in sample:
            detail_count += 1
            print(f"\nMemory {detail_count}: {sample['memory_id'][:8]}...")
            print(f"  Text: {sample['raw_text'][:80]}...")
            ext = sample['extraction']
            if ext.get('who'):
                print(f"  WHO: {ext['who']}")
            if ext.get('where'):
                print(f"  WHERE: {ext['where']}")
            if ext.get('what'):
                what_preview = ext['what'][:60] if ext['what'] else None
                print(f"  WHAT: {what_preview}..." if what_preview else "  WHAT: None")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Granular queue-based backfill"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help="Memories to process before updating progress"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help="Parallel workers (can be higher with granular tasks)"
    )
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help="Don't skip existing embeddings"
    )
    parser.add_argument(
        '--regenerate',
        action='store_true',
        help="Regenerate components from raw_text"
    )
    parser.add_argument(
        '--checkpoint-file',
        type=str,
        default='backfill_checkpoint.pkl',
        help="Path to checkpoint file (default: backfill_checkpoint.pkl)"
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help="Don't resume from checkpoint, start fresh"
    )
    parser.add_argument(
        '--clear-checkpoint',
        action='store_true',
        help="Clear existing checkpoint and start fresh"
    )

    args = parser.parse_args()

    # Clear checkpoint if requested
    if args.clear_checkpoint and os.path.exists(args.checkpoint_file):
        os.remove(args.checkpoint_file)
        print(f"Cleared checkpoint file: {args.checkpoint_file}")

    print(f"Starting granular queue-based backfill")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Skip existing: {not args.no_skip}")
    print(f"Regenerate: {args.regenerate}")
    print(f"Resume: {not args.no_resume}")

    start_time = time.time()

    backfill_granular(
        skip_existing=not args.no_skip,
        regenerate=args.regenerate,
        batch_size=args.batch_size,
        max_workers=args.workers,
        checkpoint_file=args.checkpoint_file,
        resume=not args.no_resume
    )

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f} seconds")
