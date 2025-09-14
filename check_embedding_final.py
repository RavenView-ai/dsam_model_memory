"""Final script to investigate embedding mismatch issue"""
import sqlite3
import json
import numpy as np
import struct
from agentic_memory.embedding import get_llama_embedder

def check_memory_embedding():
    """Check how embeddings are generated and searched for Part 2 memories"""

    # Connect to database
    conn = sqlite3.connect('data/amemory.sqlite3')
    cursor = conn.cursor()

    # Check the embeddings table schema
    cursor.execute("PRAGMA table_info(embeddings)")
    columns = cursor.fetchall()
    print("Embeddings table schema:")
    embed_col_name = None
    for col in columns:
        print(f"  {col[1]}: {col[2]}")
        if 'embed' in col[1].lower() and col[1] != 'embed_model':
            embed_col_name = col[1]

    # Find the specific Part 2 memory we're investigating
    memory_id = 'mem_0d46c6196dd5'  # From the previous run

    cursor.execute("""
        SELECT memory_id, raw_text, what, why, how
        FROM memories
        WHERE memory_id = ?
    """, (memory_id,))

    result = cursor.fetchone()
    if not result:
        print(f"Memory {memory_id} not found!")
        return

    memory_id, raw_text, what, why, how = result

    print("=" * 80)
    print("MEMORY DETAILS:")
    print(f"Memory ID: {memory_id}")
    print(f"Raw text: {raw_text}")
    print(f"What field: {what}")
    print(f"Why field: {why}")
    print(f"How field: {how}")

    # Get the embedding - try different column names
    for col_name in ['vector', 'embedding_vector', 'embed_vector', 'data']:
        try:
            cursor.execute(f"""
                SELECT {col_name}, dimension
                FROM embeddings
                WHERE memory_id = ?
            """, (memory_id,))
            embed_result = cursor.fetchone()
            if embed_result:
                print(f"\nFound embedding in column '{col_name}'")
                break
        except sqlite3.OperationalError:
            continue
    else:
        print("\nNo embedding found in embeddings table!")
        # Try to get it from FAISS index directly
        print("Checking FAISS index instead...")

    # Reconstruct the embed_text that was used when creating the embedding
    # Based on base_extractor.py create_embed_text method
    reconstructed_embed_text = f"WHAT: {what}\nWHY: {why}\nHOW: {how}\nRAW: {raw_text[:500]}"

    print("\n" + "=" * 80)
    print("RECONSTRUCTED EMBED TEXT (what was actually embedded):")
    print(reconstructed_embed_text)

    # Initialize embedder
    embedder = get_llama_embedder()

    # Generate embedding for the reconstructed text
    reconstructed_embedding = embedder.encode([reconstructed_embed_text], normalize_embeddings=True)[0]

    # Test different search queries
    test_queries = [
        # Your exact search query
        "[Part 2] The AI assistant asks the user to provide more context or specify the time they are referring to regarding dinner prep.",
        # Without [Part 2] prefix
        "The AI assistant asks the user to provide more context or specify the time they are referring to regarding dinner prep.",
        # Shorter semantic match
        "AI asks user for more context about dinner preparation time",
        # Key terms only
        "provide context dinner prep time",
        # The actual raw_text from DB
        raw_text,
        # The what field
        what
    ]

    print("\n" + "=" * 80)
    print("TESTING SEARCH QUERIES:")
    print("(Comparing query embeddings with reconstructed embed_text embedding)")

    for i, query in enumerate(test_queries):
        print(f"\nQuery {i+1}: '{query[:80]}...'")

        # Generate query embedding
        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]

        # Calculate similarity with reconstructed embedding
        similarity = np.dot(reconstructed_embedding, query_embedding)
        print(f"  Cosine similarity: {similarity:.4f}")

    # Now check how the actual search works
    print("\n" + "=" * 80)
    print("HOW SEARCH ACTUALLY WORKS:")

    # Check retrieval.py to understand search
    print("\n1. In router.py search_memories():")
    print("   - Query text is directly embedded: embedder.encode([query])")
    print("\n2. In router.py ingest():")
    print("   - Memory embedding uses create_embed_text() format")
    print("   - Format: 'WHAT: {what}\\nWHY: {why}\\nHOW: {how}\\nRAW: {raw[:500]}'")

    # The core issue
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS:")
    print("\n❌ THE PROBLEM:")
    print("   When you search for the exact raw_text like:")
    print(f"   '{raw_text[:80]}...'")
    print("\n   It gets embedded AS-IS.")
    print("\n   But the stored embedding was created from:")
    print(f"   'WHAT: {what[:50]}...\\nWHY: {why[:50]}...\\nHOW: {how[:50]}...\\nRAW: ...'")
    print("\n   These are VERY different texts, so embeddings won't match well!")

    print("\n✅ SOLUTIONS:")
    print("\n   Option 1: Store dual embeddings")
    print("   - One for structured search (current WHAT/WHY/HOW format)")
    print("   - One for raw text search (just the raw_text)")
    print("\n   Option 2: Transform search queries")
    print("   - Extract 5W1H from search queries before embedding")
    print("   - Use same WHAT/WHY/HOW format for queries")
    print("\n   Option 3: Hybrid search")
    print("   - Use both semantic (embedding) and lexical (FTS) search")
    print("   - FTS would find exact text matches")

    conn.close()

if __name__ == "__main__":
    check_memory_embedding()
