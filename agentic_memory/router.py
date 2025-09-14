from __future__ import annotations
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from datetime import datetime
import numpy as np

# Use llama.cpp embeddings
from .embedding import get_llama_embedder
from .embedding.component_embedder import get_component_embedder
from .embedding.instruction_generator import get_instruction_generator
_embedder = get_llama_embedder()
_component_embedder = get_component_embedder()
_instruction_generator = get_instruction_generator()

from .config import cfg
from .types import RawEvent, RetrievalQuery
from .extraction.llm_extractor import UnifiedExtractor
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .retrieval import HybridRetriever
from .block_builder import BlockBuilder
from .tokenization import TokenizerAdapter
# Try to import optional components
try:
    from .attention import AdaptiveEmbeddingSpace
    HAS_ATTENTION = True
except ImportError:
    HAS_ATTENTION = False
    AdaptiveEmbeddingSpace = None

try:
    from .cluster.concept_cluster import LiquidMemoryClusters
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False
    LiquidMemoryClusters = None

class MemoryRouter:
    def __init__(self, store: MemoryStore, index: FaissIndex):
        self.store = store
        self.index = index
        self.retriever = HybridRetriever(store, index)
        self.builder = BlockBuilder(store)
        # Use global llama.cpp embedder
        self.embedder = _embedder
        self.component_embedder = _component_embedder
        self.tok = TokenizerAdapter()
        self.extractor = UnifiedExtractor()
        
        # Initialize dynamic components if enabled and available
        if cfg.use_attention_retrieval and HAS_ATTENTION:
            self.adaptive_embeddings = AdaptiveEmbeddingSpace(
                base_dim=cfg.embed_dim,
                meta_dim=64
            )
        else:
            self.adaptive_embeddings = None
            
        if cfg.use_liquid_clustering and HAS_CLUSTERING:
            self.liquid_clusters = LiquidMemoryClusters(
                n_clusters=64,
                dim=cfg.embed_dim
            )
            self.liquid_clusters.flow_rate = cfg.cluster_flow_rate
            self.liquid_clusters.merge_threshold = cfg.cluster_merge_threshold
        else:
            self.liquid_clusters = None

    def ingest(self, raw_event: RawEvent, context_hint: str = "", use_multi_part: bool = True) -> str:
        """
        Ingest a raw event, optionally breaking it into multiple memories.

        Args:
            raw_event: The raw event to process
            context_hint: Additional context for extraction
            use_multi_part: If True, attempt to break complex content into multiple memories

        Returns:
            Comma-separated list of memory IDs created
        """
        # Extract memories (single or multi based on content and settings)
        max_parts = None if use_multi_part else 1
        memories = self.extractor.extract_memories(raw_event, context_hint, max_parts)

        memory_ids = []
        for rec in memories:
            vec = np.array(rec.extra.pop('embed_vector_np'), dtype='float32')

            # Apply adaptive embedding if enabled
            vec = self._apply_adaptive_embedding(vec, rec.memory_id)

            # Persist
            self.store.upsert_memory(rec, embedding=vec.tobytes(), dim=vec.shape[0])
            # Add to FAISS (normalized already)
            self.index.add(rec.memory_id, vec)

            # Generate and store component embeddings
            self._store_component_embeddings(rec)

            memory_ids.append(rec.memory_id)

        if memory_ids:
            self.index.save()

        return ','.join(memory_ids) if memory_ids else ''
    
    def ingest_batch(self, raw_events: List[RawEvent], context_hints: Optional[List[str]] = None) -> List[str]:
        """
        Batch ingest multiple raw events for better performance.

        Args:
            raw_events: List of raw events to process
            context_hints: Optional list of context hints (one per event)

        Returns:
            List of comma-separated memory IDs created for each event
        """
        # Use batch extraction for efficiency
        all_memories = self.extractor.extract_batch(raw_events, context_hints)

        result_ids = []
        all_vectors = []
        all_records = []

        # Process all memories from all events
        for event_memories in all_memories:
            event_memory_ids = []

            for rec in event_memories:
                vec = np.array(rec.extra.pop('embed_vector_np'), dtype='float32')

                # Apply adaptive embedding if enabled
                vec = self._apply_adaptive_embedding(vec, rec.memory_id)

                all_vectors.append(vec)
                all_records.append(rec)
                event_memory_ids.append(rec.memory_id)

            result_ids.append(','.join(event_memory_ids) if event_memory_ids else '')

        # Batch persist to database
        for rec, vec in zip(all_records, all_vectors):
            self.store.upsert_memory(rec, embedding=vec.tobytes(), dim=vec.shape[0])

        # Batch add to FAISS index with component embeddings
        for rec, vec in zip(all_records, all_vectors):
            self.index.add(rec.memory_id, vec)
            self._store_component_embeddings(rec)

        # Save index once after all additions
        if all_records:
            self.index.save()

        return result_ids

    def retrieve_block(self, session_id: str, context_messages: List[Dict[str, str]],
                       actor_hint: Optional[str] = None,
                       temporal_hint: Optional[Union[str, Tuple[str, str], Dict]] = None) -> Dict[str, Any]:
        # Construct query text from context (last N messages)
        ctx_text = "\n".join([f"{m.get('role','')}: {m.get('content','')}" for m in context_messages[-6:]])
        
        # Auto-detect temporal hint if not provided
        if temporal_hint is None and cfg.get('auto_detect_temporal', True):
            from .extraction.temporal_parser import TemporalParser
            parser = TemporalParser()
            temporal_hint, cleaned_text = parser.extract_and_clean_query(ctx_text)
            if temporal_hint:
                ctx_text = cleaned_text  # Use cleaned text for better semantic search
        
        rq = RetrievalQuery(
            session_id=session_id,
            actor_hint=actor_hint,
            temporal_hint=temporal_hint,
            text=ctx_text
        )

        # Generate instruction based on default weights
        weights = {
            'semantic': cfg.w_semantic,
            'recency': cfg.w_recency,
            'actor': cfg.w_actor,
            'spatial': cfg.w_spatial,
            'temporal': cfg.w_temporal,
            'usage': cfg.w_usage
        }
        instruction = _instruction_generator.generate_instruction(weights)

        # Encode with instruction prefix for better retrieval
        qvec = self.embedder.encode([ctx_text], normalize_embeddings=True, instruction=instruction)[0]
        
        # Apply adaptive query embedding if enabled
        if self.adaptive_embeddings:
            # Create synthetic usage stats for query
            query_stats = {'access_count': 1, 'recency_score': 1.0, 'diversity_score': 0.5}
            qvec = self.adaptive_embeddings.encode_with_context(qvec, query_stats, None)
        
        ranked = self.retriever.search(rq, qvec, topk_sem=60, topk_lex=60)
        
        # Update liquid clusters if enabled and we have results
        if self.liquid_clusters and ranked:
            retrieved_ids = [c.memory_id for c in ranked[:20]]
            
            # Get embeddings for clustering
            embeddings_dict = {}
            for cand in ranked[:20]:
                # Try to get embedding from index or store
                emb = self.store.get_embedding_drift(cand.memory_id)
                if emb is not None:
                    embeddings_dict[cand.memory_id] = emb
            
            if embeddings_dict:
                # Update co-access patterns
                self.liquid_clusters.update_co_access(retrieved_ids)
                
                # Update cluster energy
                self.liquid_clusters.update_cluster_energy(retrieved_ids)
                
                # Perform flow step
                self.liquid_clusters.flow_step(retrieved_ids, embeddings_dict)
                
                # Periodically merge similar clusters
                if np.random.random() < 0.05:  # 5% chance
                    self.liquid_clusters.merge_similar_clusters(embeddings_dict)
        
        overhead = sum([len(m.get('content','')) for m in context_messages]) // 3  # rough token overhead
        blocks = self.builder.build(rq, ranked, context_overhead=overhead)
        if not blocks:
            return {"block": None, "members": []}
        first = blocks[0]
        # Use the dynamically generated block directly instead of trying to retrieve from DB
        out = {
            'block': first.dict(),
            'members': first.member_ids
        }
        
        # Update embedding drift based on retrieval context
        if self.adaptive_embeddings and out.get('members'):
            member_ids = out['members'][:10]  # Top 10 members
            # Note: Embedding drift tracking removed (table was dropped)
            # Previously tracked embedding evolution for adaptive retrieval
        
        return out
    
    def search_memories(self, query: str, weights: Optional[Dict[str, float]] = None, 
                       token_budget: Optional[int] = None, initial_candidates: int = 100,
                       score_threshold: float = 0.0, top_k: Optional[int] = None,
                       excluded_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search memories using the same process as the analyzer with knapsack optimization.
        
        Args:
            query: Search query text
            weights: Optional custom weights (uses config defaults if not provided)
            token_budget: Optional token budget (defaults to context window - reserves)
            initial_candidates: Number of initial candidates to retrieve before knapsack
            score_threshold: Minimum score threshold for memories (default 0.0)
            top_k: Maximum number of results to return (None for unlimited)
            excluded_ids: List of memory IDs to exclude from results
            
        Returns:
            Dictionary containing selected memories and metadata
        """
        from .settings_manager import SettingsManager
        from .block_builder import greedy_knapsack
        
        # Get weights from settings if not provided
        if weights is None:
            settings_manager = SettingsManager()
            weights = settings_manager.get_weights()
        
        # Calculate token budget if not provided
        if token_budget is None:
            token_budget = cfg.context_window - cfg.reserve_output_tokens - cfg.reserve_system_tokens - 512
        token_budget = max(512, token_budget)
        
        # Decompose query
        decomposition = self.retriever.decompose_query(query)
        
        # Create retrieval query
        rq = RetrievalQuery(
            session_id='memory_search',
            text=decomposition.get('what', query),
            actor_hint=decomposition['who'].get('id') if decomposition.get('who') else None,
            temporal_hint=decomposition.get('when')
        )
        
        # Generate instruction based on provided weights
        instruction = _instruction_generator.generate_instruction(weights)

        # Get query embedding with instruction prefix
        qvec = self.embedder.encode([query], normalize_embeddings=True, instruction=instruction)[0]

        # Apply adaptive embedding if enabled
        if self.adaptive_embeddings:
            query_stats = {'access_count': 1, 'recency_score': 1.0, 'diversity_score': 0.5}
            qvec = self.adaptive_embeddings.encode_with_context(qvec, query_stats, None)
        
        # Search with weights - get many candidates for knapsack
        candidates = self.retriever.search_with_weights(
            rq, qvec, weights,
            topk_sem=initial_candidates,
            topk_lex=initial_candidates
        )
        
        # Apply knapsack algorithm
        selected_ids, tokens_used = greedy_knapsack(candidates, token_budget)
        
        # Filter to selected memories
        selected_candidates = [c for c in candidates if c.memory_id in selected_ids]
        
        # Apply additional filters
        if excluded_ids:
            excluded_set = set(excluded_ids)
            selected_candidates = [c for c in selected_candidates if c.memory_id not in excluded_set]
        
        if score_threshold > 0:
            selected_candidates = [c for c in selected_candidates if c.score >= score_threshold]
        
        if top_k is not None and top_k > 0:
            selected_candidates = selected_candidates[:top_k]
        
        # Get detailed scores
        detailed_scores = self.retriever.get_detailed_scores(selected_candidates)
        
        # Build response with memory details
        memories = []
        # Fetch all selected memories at once
        if selected_candidates:
            memory_ids = [c.memory_id for c in selected_candidates]
            fetched_memories = self.store.fetch_memories(memory_ids)
            memory_lookup = {m['memory_id']: m for m in fetched_memories}
            
        for candidate in selected_candidates:
            memory_dict = memory_lookup.get(candidate.memory_id)
            if memory_dict:
                # Find detailed score for this memory
                detail = next((d for d in detailed_scores if d['memory_id'] == candidate.memory_id), {})
                memories.append({
                    'memory_id': candidate.memory_id,
                    'total_score': round(candidate.score, 3),
                    'token_count': candidate.token_count,
                    'scores': detail.get('scores', {}),
                    'when': memory_dict.get('when_ts', memory_dict.get('created_at', '')),
                    'who': f"{memory_dict.get('who_type', '')}: {memory_dict.get('who_id', '')}",
                    'what': memory_dict.get('what', ''),
                    'where': f"{memory_dict.get('where_type', '')}: {memory_dict.get('where_value', '')}",
                    'why': memory_dict.get('why', ''),
                    'how': memory_dict.get('how', ''),
                    'raw_text': memory_dict.get('raw_text', '')
                })
        
        return {
            'query': query,
            'decomposition': decomposition,
            'weights': weights,
            'total_candidates': len(candidates),
            'selected_count': len(selected_candidates),
            'token_budget': token_budget,
            'tokens_used': tokens_used,
            'memories': memories
        }

    def _apply_adaptive_embedding(self, vec: np.ndarray, memory_id: str) -> np.ndarray:
        """Apply adaptive embedding transformation if enabled."""
        if self.adaptive_embeddings:
            usage_stats = self.store.get_usage_stats([memory_id])
            return self.adaptive_embeddings.encode_with_context(
                vec,
                usage_stats.get(memory_id, {}),
                None
            )
        return vec

    def _store_component_embeddings(self, rec: MemoryRecord) -> None:
        """Generate and store component embeddings for a memory record."""
        import json

        # Process WHO embeddings
        if rec.who and rec.who.id:
            who_vec = self.embedder.encode([rec.who.id], normalize_embeddings=True)[0]
            self.index.add(f"who:{rec.memory_id}", who_vec)
            self.store.store_component_embedding(
                rec.memory_id, 'who', rec.who.id,
                who_vec.astype('float32').tobytes(), who_vec.shape[0]
            )

        # Process WHO_LIST embeddings
        if rec.who_list:
            try:
                who_entities = json.loads(rec.who_list) if isinstance(rec.who_list, str) else rec.who_list
                for entity in who_entities:
                    if entity:
                        entity_vec = self.embedder.encode([entity], normalize_embeddings=True)[0]
                        self.store.store_component_embedding(
                            rec.memory_id, 'who', entity,
                            entity_vec.astype('float32').tobytes(), entity_vec.shape[0]
                        )
            except (json.JSONDecodeError, TypeError):
                pass

        # Process WHERE embeddings
        if rec.where and rec.where.value:
            where_vec = self.embedder.encode([rec.where.value], normalize_embeddings=True)[0]
            self.index.add(f"where:{rec.memory_id}", where_vec)
            self.store.store_component_embedding(
                rec.memory_id, 'where', rec.where.value,
                where_vec.astype('float32').tobytes(), where_vec.shape[0]
            )

        # Process WHERE_LIST embeddings
        if rec.where_list:
            try:
                where_entities = json.loads(rec.where_list) if isinstance(rec.where_list, str) else rec.where_list
                for entity in where_entities:
                    if entity:
                        entity_vec = self.embedder.encode([entity], normalize_embeddings=True)[0]
                        self.store.store_component_embedding(
                            rec.memory_id, 'where', entity,
                            entity_vec.astype('float32').tobytes(), entity_vec.shape[0]
                        )
            except (json.JSONDecodeError, TypeError):
                pass

        # Process key WHAT entities (optional - only the most important ones)
        if rec.what:
            try:
                what_entities = json.loads(rec.what) if isinstance(rec.what, str) else [rec.what]
                # Only embed first 3 most important entities to avoid explosion
                for entity in what_entities[:3]:
                    if entity:
                        entity_vec = self.embedder.encode([entity], normalize_embeddings=True)[0]
                        self.store.store_component_embedding(
                            rec.memory_id, 'what', entity,
                            entity_vec.astype('float32').tobytes(), entity_vec.shape[0]
                        )
            except (json.JSONDecodeError, TypeError):
                pass
