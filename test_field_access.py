"""
Test accessing fields on MemoryRecord to identify the callable issue.
"""

from datetime import datetime
from agentic_memory.extraction.llm_extractor import UnifiedExtractor
from agentic_memory.types import RawEvent
import traceback

def test_field_access():
    """Test accessing each field to find the problematic one."""

    # Create a simple RawEvent
    raw_event = RawEvent(
        content="Test memory content for debugging",
        session_id="test_session",
        event_type='user_message',
        actor='test_user',
        timestamp=datetime.now()
    )

    # Extract
    extractor = UnifiedExtractor()
    records = extractor.extract_memories(raw_event, max_parts=1)

    if not records:
        print("No records extracted")
        return

    record = records[0]
    print(f"Record type: {type(record)}")
    print(f"Record class: {record.__class__.__name__}")

    # Test each field access
    fields_to_test = [
        ('who', lambda r: r.who),
        ('who.id', lambda r: r.who.id if r.who else None),
        ('who_list', lambda r: r.who_list),
        ('what', lambda r: r.what),
        ('when', lambda r: r.when),
        ('when.isoformat', lambda r: r.when.isoformat() if r.when else None),
        ('when_list', lambda r: r.when_list),
        ('where', lambda r: r.where),
        ('where.value', lambda r: r.where.value if r.where else None),
        ('where_list', lambda r: r.where_list),
        ('why', lambda r: r.why),
        ('how', lambda r: r.how),
    ]

    for field_name, accessor in fields_to_test:
        try:
            value = accessor(record)
            print(f"OK {field_name}: {type(value)} = {value}")
        except Exception as e:
            print(f"ERROR {field_name}: ERROR - {e}")
            traceback.print_exc()

    # Check if any are callable
    print("\n--- Checking if fields are callable ---")
    for attr_name in ['who', 'who_list', 'what', 'when', 'when_list', 'where', 'where_list', 'why', 'how']:
        if hasattr(record, attr_name):
            attr = getattr(record, attr_name)
            print(f"{attr_name}: callable={callable(attr)}, type={type(attr)}")

            # If it's callable but shouldn't be, that's our problem
            if callable(attr) and attr_name not in ['when']:  # datetime methods are ok
                print(f"  WARNING: {attr_name} is unexpectedly callable!")
                # Try to see what it returns
                try:
                    result = attr()
                    print(f"  Calling {attr_name}() returns: {result}")
                except Exception as e:
                    print(f"  Calling {attr_name}() raises: {e}")

if __name__ == "__main__":
    test_field_access()