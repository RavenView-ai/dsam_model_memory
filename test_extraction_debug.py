"""
Debug script to identify the extraction issue.
"""

from datetime import datetime
from agentic_memory.extraction.llm_extractor import UnifiedExtractor
from agentic_memory.types import RawEvent
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.config import cfg
import traceback

def test_extraction():
    """Test the extraction with detailed debugging."""
    print("=" * 80)
    print("TESTING EXTRACTION")
    print("=" * 80)

    # Get a real memory from database
    store = MemoryStore(cfg.db_path)

    with store.connect() as con:
        row = con.execute("""
            SELECT memory_id, session_id, raw_text, who_id, created_at
            FROM memories
            LIMIT 1
        """).fetchone()

    if not row:
        print("No memories found")
        return

    print(f"\nMemory ID: {row['memory_id']}")
    print(f"Raw text: {row['raw_text'][:100]}...")
    print(f"Session ID: {row['session_id']}")
    print(f"Who ID: {row['who_id']}")
    print(f"Created at: {row['created_at']}")

    # Create RawEvent
    print("\n--- Creating RawEvent ---")
    try:
        raw_event = RawEvent(
            content=row['raw_text'],
            session_id=row['session_id'],
            event_type='user_message',
            actor=row['who_id'] or 'user',
            timestamp=datetime.fromisoformat(row['created_at'])
        )
        print("RawEvent created successfully")
        print(f"  content length: {len(raw_event.content)}")
        print(f"  session_id: {raw_event.session_id}")
        print(f"  event_type: {raw_event.event_type}")
        print(f"  actor: {raw_event.actor}")
        print(f"  timestamp: {raw_event.timestamp}")
    except Exception as e:
        print(f"ERROR creating RawEvent: {e}")
        traceback.print_exc()
        return

    # Create extractor
    print("\n--- Creating Extractor ---")
    try:
        extractor = UnifiedExtractor()
        print("Extractor created successfully")
    except Exception as e:
        print(f"ERROR creating extractor: {e}")
        traceback.print_exc()
        return

    # Try extraction
    print("\n--- Attempting Extraction ---")
    try:
        records = extractor.extract_memories(raw_event, max_parts=1)
        print(f"Extraction completed, got {len(records)} records")

        if records:
            record = records[0]
            print("\n--- Extracted Data ---")
            print(f"Type of record: {type(record)}")
            print(f"Record attributes: {dir(record)}")

            # Check each field
            print("\n--- Checking Fields ---")

            # WHO
            print(f"who type: {type(record.who)}")
            print(f"who value: {record.who}")
            if record.who:
                print(f"  who.id: {record.who.id}")
                print(f"  who.type: {record.who.type}")

            # WHO_LIST
            print(f"who_list type: {type(record.who_list)}")
            print(f"who_list value: {record.who_list}")

            # WHAT
            print(f"what type: {type(record.what)}")
            print(f"what value: {record.what}")

            # WHEN
            print(f"when type: {type(record.when)}")
            print(f"when value: {record.when}")
            if record.when:
                print(f"  when isoformat: {record.when.isoformat()}")

            # WHERE
            print(f"where type: {type(record.where)}")
            print(f"where value: {record.where}")
            if record.where:
                print(f"  where.value: {record.where.value}")
                print(f"  where.type: {record.where.type}")

            # WHY and HOW
            print(f"why type: {type(record.why)}")
            print(f"why value: {record.why}")
            print(f"how type: {type(record.how)}")
            print(f"how value: {record.how}")

    except Exception as e:
        print(f"ERROR during extraction: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

        # Try to identify the specific line
        import sys
        tb = sys.exc_info()[2]
        print("\nTraceback frames:")
        while tb:
            frame = tb.tb_frame
            print(f"  File: {frame.f_code.co_filename}")
            print(f"  Function: {frame.f_code.co_name}")
            print(f"  Line: {tb.tb_lineno}")
            if 'self' in frame.f_locals:
                print(f"  self type: {type(frame.f_locals['self'])}")
            print()
            tb = tb.tb_next

if __name__ == "__main__":
    test_extraction()