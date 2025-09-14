"""
Test script for verifying instruction-enhanced search functionality.

This script tests whether the Qwen3-Embedding instruction prefixes improve
retrieval performance based on different weight configurations.
"""

import numpy as np
from agentic_memory.storage.sql_store import MemoryStore
from agentic_memory.storage.faiss_index import FaissIndex
from agentic_memory.router import MemoryRouter
from agentic_memory.embedding.llama_embedder import get_llama_embedder
from agentic_memory.embedding.instruction_generator import get_instruction_generator
from agentic_memory.config import cfg


def test_instruction_generation():
    """Test that instructions are generated correctly for different weight configs."""
    print("=" * 80)
    print("TESTING INSTRUCTION GENERATION")
    print("=" * 80)

    generator = get_instruction_generator()

    # Test 1: Semantic-dominant weights
    weights = {
        'semantic': 0.7,
        'recency': 0.1,
        'actor': 0.1,
        'spatial': 0.05,
        'temporal': 0.03,
        'usage': 0.02
    }
    instruction = generator.generate_instruction(weights)
    print(f"\nSemantic-dominant weights: {weights}")
    print(f"Generated instruction: {instruction}")

    # Test 2: Recency-focused weights
    weights = {
        'semantic': 0.3,
        'recency': 0.5,
        'actor': 0.1,
        'spatial': 0.05,
        'temporal': 0.03,
        'usage': 0.02
    }
    instruction = generator.generate_instruction(weights)
    print(f"\nRecency-focused weights: {weights}")
    print(f"Generated instruction: {instruction}")

    # Test 3: Multi-aspect weights
    weights = {
        'semantic': 0.35,
        'recency': 0.3,
        'actor': 0.25,
        'spatial': 0.05,
        'temporal': 0.03,
        'usage': 0.02
    }
    instruction = generator.generate_instruction(weights)
    print(f"\nMulti-aspect weights: {weights}")
    print(f"Generated instruction: {instruction}")

    # Test 4: Balanced weights
    weights = {
        'semantic': 0.17,
        'recency': 0.17,
        'actor': 0.17,
        'spatial': 0.17,
        'temporal': 0.16,
        'usage': 0.16
    }
    instruction = generator.generate_instruction(weights)
    print(f"\nBalanced weights: {weights}")
    print(f"Generated instruction: {instruction}")


def test_embedding_with_instructions():
    """Test that embeddings change with different instructions."""
    print("\n" + "=" * 80)
    print("TESTING EMBEDDING WITH INSTRUCTIONS")
    print("=" * 80)

    embedder = get_llama_embedder()
    test_query = "What did we discuss about the project yesterday?"

    # Generate embeddings with different instructions
    no_instruction = embedder.encode([test_query], normalize_embeddings=True)[0]

    semantic_instruction = "Retrieve passages with similar meaning and conceptual relevance"
    semantic_embed = embedder.encode([test_query], normalize_embeddings=True, instruction=semantic_instruction)[0]

    recency_instruction = "Find recent events and time-sensitive information"
    recency_embed = embedder.encode([test_query], normalize_embeddings=True, instruction=recency_instruction)[0]

    actor_instruction = "Search for memories involving specific people or entities"
    actor_embed = embedder.encode([test_query], normalize_embeddings=True, instruction=actor_instruction)[0]

    # Calculate cosine similarities to show differences
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f"\nQuery: '{test_query}'")
    print(f"\nCosine similarities between different instruction embeddings:")
    print(f"  No instruction vs Semantic: {cosine_similarity(no_instruction, semantic_embed):.4f}")
    print(f"  No instruction vs Recency:  {cosine_similarity(no_instruction, recency_embed):.4f}")
    print(f"  No instruction vs Actor:    {cosine_similarity(no_instruction, actor_embed):.4f}")
    print(f"  Semantic vs Recency:        {cosine_similarity(semantic_embed, recency_embed):.4f}")
    print(f"  Semantic vs Actor:          {cosine_similarity(semantic_embed, actor_embed):.4f}")
    print(f"  Recency vs Actor:           {cosine_similarity(recency_embed, actor_embed):.4f}")

    # Show embedding statistics
    print(f"\nEmbedding norms (should all be ~1.0 if normalized):")
    print(f"  No instruction: {np.linalg.norm(no_instruction):.4f}")
    print(f"  Semantic:       {np.linalg.norm(semantic_embed):.4f}")
    print(f"  Recency:        {np.linalg.norm(recency_embed):.4f}")
    print(f"  Actor:          {np.linalg.norm(actor_embed):.4f}")


def test_search_with_different_weights():
    """Test search functionality with different weight configurations."""
    print("\n" + "=" * 80)
    print("TESTING SEARCH WITH DIFFERENT WEIGHTS")
    print("=" * 80)

    # Initialize router
    store = MemoryStore(cfg.db_path)
    index = FaissIndex(cfg.embed_dim, cfg.index_path)
    router = MemoryRouter(store, index)

    test_query = "What happened in the kitchen yesterday?"

    # Test with semantic-focused weights
    semantic_weights = {
        'semantic': 0.8,
        'recency': 0.05,
        'actor': 0.05,
        'spatial': 0.05,
        'temporal': 0.03,
        'usage': 0.02
    }

    print(f"\nSearching with semantic-focused weights...")
    results = router.search_memories(
        query=test_query,
        weights=semantic_weights,
        initial_candidates=20,
        token_budget=2000
    )

    if results['memories']:
        print(f"Found {len(results['memories'])} memories")
        print(f"Top result: {results['memories'][0]['content'][:100]}...")

    # Test with spatial-focused weights
    spatial_weights = {
        'semantic': 0.2,
        'recency': 0.1,
        'actor': 0.1,
        'spatial': 0.5,  # Heavy spatial focus
        'temporal': 0.05,
        'usage': 0.05
    }

    print(f"\nSearching with spatial-focused weights...")
    results = router.search_memories(
        query=test_query,
        weights=spatial_weights,
        initial_candidates=20,
        token_budget=2000
    )

    if results['memories']:
        print(f"Found {len(results['memories'])} memories")
        print(f"Top result: {results['memories'][0]['content'][:100]}...")

    # Test with temporal-focused weights
    temporal_weights = {
        'semantic': 0.2,
        'recency': 0.3,
        'actor': 0.1,
        'spatial': 0.1,
        'temporal': 0.25,  # Focus on time
        'usage': 0.05
    }

    print(f"\nSearching with temporal-focused weights...")
    results = router.search_memories(
        query=test_query,
        weights=temporal_weights,
        initial_candidates=20,
        token_budget=2000
    )

    if results['memories']:
        print(f"Found {len(results['memories'])} memories")
        print(f"Top result: {results['memories'][0]['content'][:100]}...")


def test_component_instructions():
    """Test component-specific instruction generation."""
    print("\n" + "=" * 80)
    print("TESTING COMPONENT-SPECIFIC INSTRUCTIONS")
    print("=" * 80)

    generator = get_instruction_generator()

    # Test WHO component
    who_instruction = generator.generate_component_instruction('who', 'John Smith')
    print(f"\nWHO component for 'John Smith':")
    print(f"  Instruction: {who_instruction}")

    # Test WHERE component
    where_instruction = generator.generate_component_instruction('where', 'kitchen')
    print(f"\nWHERE component for 'kitchen':")
    print(f"  Instruction: {where_instruction}")

    # Test WHAT component
    what_instruction = generator.generate_component_instruction('what', 'cooking dinner')
    print(f"\nWHAT component for 'cooking dinner':")
    print(f"  Instruction: {what_instruction}")


if __name__ == "__main__":
    print("INSTRUCTION-ENHANCED SEARCH TEST SUITE")
    print("=" * 80)

    # Run all tests
    test_instruction_generation()
    test_embedding_with_instructions()
    test_component_instructions()

    # Only run search test if there are memories in the database
    store = MemoryStore(cfg.db_path)
    with store.connect() as con:
        count = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    if count > 0:
        print(f"\nFound {count} memories in database, running search tests...")
        test_search_with_different_weights()
    else:
        print(f"\nNo memories in database, skipping search tests")

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)