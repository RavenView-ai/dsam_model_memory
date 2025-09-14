"""
Test script to verify component regeneration functionality.
"""

import json
from datetime import datetime
from agentic_memory.extraction.llm_extractor import UnifiedExtractor
from agentic_memory.types import RawEvent
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.config import cfg


def test_component_regeneration():
    """Test regenerating components from a sample memory."""
    print("=" * 80)
    print("TESTING COMPONENT REGENERATION")
    print("=" * 80)

    # Sample memory content
    sample_memory = {
        'memory_id': 'test_mem_001',
        'raw_text': 'John and Sarah met at the coffee shop on Main Street yesterday afternoon to discuss the project timeline and budget constraints.',
        'created_at': datetime.now().isoformat(),
        'source': 'test'
    }

    print(f"\nOriginal memory content:")
    print(f"  {sample_memory['raw_text']}")

    # Create extractor
    extractor = UnifiedExtractor()

    # Create RawEvent from memory
    raw_event = RawEvent(
        text=sample_memory['raw_text'],
        source=sample_memory.get('source', 'memory_regeneration'),
        ts_raw=sample_memory.get('created_at', datetime.now().isoformat())
    )

    # Extract components
    print("\nExtracting components...")
    try:
        records = extractor.extract_memories(raw_event, max_parts=1)

        if records:
            record = records[0]
            print("\nExtracted components:")

            if record.who:
                print(f"  WHO (id): {record.who.id}")
                if record.who.entities:
                    print(f"  WHO (entities): {record.who.entities}")

            if record.what:
                print(f"  WHAT: {record.what}")

            if record.when:
                print(f"  WHEN (value): {record.when.value}")
                if record.when.entities:
                    print(f"  WHEN (entities): {record.when.entities}")

            if record.where:
                print(f"  WHERE (value): {record.where.value}")
                if record.where.entities:
                    print(f"  WHERE (entities): {record.where.entities}")

            if record.why:
                print(f"  WHY: {record.why}")

            if record.how:
                print(f"  HOW: {record.how}")

            # Show how it would be stored
            print("\nDatabase format:")
            print(f"  who_id: {record.who.id if record.who else None}")
            print(f"  who_list: {json.dumps(record.who.entities) if record.who and record.who.entities else None}")
            print(f"  what: {json.dumps(record.what) if isinstance(record.what, list) else record.what}")
            print(f"  when_value: {record.when.value if record.when else None}")
            print(f"  when_list: {json.dumps(record.when.entities) if record.when and record.when.entities else None}")
            print(f"  where_value: {record.where.value if record.where else None}")
            print(f"  where_list: {json.dumps(record.where.entities) if record.where and record.where.entities else None}")

        else:
            print("No records extracted")

    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()


def test_batch_regeneration():
    """Test regenerating components for multiple memories."""
    print("\n" + "=" * 80)
    print("TESTING BATCH REGENERATION")
    print("=" * 80)

    # Get a few real memories from database
    store = MemoryStore(cfg.db_path)

    with store.connect() as con:
        rows = con.execute("""
            SELECT memory_id, raw_text, who_id, what, where_value
            FROM memories
            LIMIT 3
        """).fetchall()

    if not rows:
        print("No memories found in database")
        return

    print(f"\nFound {len(rows)} memories to test")

    extractor = UnifiedExtractor()

    for row in rows:
        print(f"\n--- Memory: {row['memory_id']} ---")
        print(f"Content: {row['raw_text'][:100]}...")
        print(f"Current WHO: {row['who_id']}")
        print(f"Current WHERE: {row['where_value']}")

        # Try regenerating
        raw_event = RawEvent(
            text=row['raw_text'],
            source='test_regeneration',
            ts_raw=datetime.now().isoformat()
        )

        try:
            records = extractor.extract_memories(raw_event, max_parts=1)
            if records:
                record = records[0]
                print(f"New WHO: {record.who.id if record.who else None}")
                print(f"New WHERE: {record.where.value if record.where else None}")
                if record.who and record.who.entities:
                    print(f"New WHO entities: {record.who.entities}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("COMPONENT REGENERATION TEST SUITE")
    print("=" * 80)

    # Test basic regeneration
    test_component_regeneration()

    # Test with real memories if available
    try:
        test_batch_regeneration()
    except Exception as e:
        print(f"\nSkipping batch test: {e}")

    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)