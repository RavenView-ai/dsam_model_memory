from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Union, Any
import numpy as np
import json
import re
from datetime import datetime, timezone
from .config import cfg
from .storage.sql_store import MemoryStore
from .storage.faiss_index import FaissIndex
from .types import RetrievalQuery, Candidate
from .embedding.component_embedder import get_component_embedder
from .embedding.llama_embedder import get_llama_embedder
# Attention mechanisms removed - using fixed weight comprehensive scoring only

def exp_recency(ts_iso: str, now: datetime, half_life_hours: float = 72.0) -> float:
    try:
        ts = datetime.fromisoformat(ts_iso.replace('Z','+00:00'))
        # Ensure both datetimes are timezone-aware or both are naive
        if ts.tzinfo is not None and now.tzinfo is None:
            ts = ts.replace(tzinfo=None)
        elif ts.tzinfo is None and now.tzinfo is not None:
            now = now.replace(tzinfo=None)
    except Exception:
        return 0.5
    dt = (now - ts).total_seconds() / 3600.0
    # Exponential decay, 0..1
    return max(0.0, min(1.0, 0.5 ** (dt / half_life_hours)))

class HybridRetriever:
    def __init__(self, store: MemoryStore, index: FaissIndex):
        self.store = store
        self.index = index
        self.component_embedder = get_component_embedder()
        self.embedder = get_llama_embedder()

        # No attention mechanisms - using comprehensive scoring with fixed weights

    def _semantic(self, qvec: np.ndarray, topk: int) -> List[Tuple[str, float]]:
        results = self.index.search(qvec, topk)
        # FAISS with METRIC_INNER_PRODUCT returns cosine similarity scores in [-1, 1]
        # where 1 is perfect match, 0 is orthogonal, -1 is opposite
        # We'll map this to [0, 1] range preserving the absolute similarity
        if not results:
            return []
        
        normalized = []
        for mid, score in results:
            # Map from [-1, 1] to [0, 1]
            # This preserves the actual similarity: 
            # - Perfect match (1.0) stays 1.0
            # - No similarity (0.0) becomes 0.5
            # - Opposite (-1.0) becomes 0.0
            norm_score = (score + 1.0) / 2.0
            # Clip to ensure we're in valid range (in case of numerical errors)
            norm_score = max(0.0, min(1.0, norm_score))
            normalized.append((mid, float(norm_score)))
        
        return normalized

    # Lexical search removed - FTS5 was broken and not needed with good semantic search
    
    def _actor_based(self, actor_hint: str, topk: int) -> List[Tuple[str, float]]:
        """Retrieve memories from specific actor using semantic similarity.

        This now uses embeddings to find semantically similar actors,
        not just exact matches.
        """
        # Generate embedding for the actor hint
        actor_embedding = self.component_embedder.embed_who(actor_hint)
        if actor_embedding is None:
            return []

        # Search for similar actor embeddings in FAISS
        # We store component embeddings with prefixed IDs
        # Limit to reasonable number to avoid performance issues
        search_limit = min(topk * 2, 100)  # Cap at 100 for performance
        results = self.index.search(actor_embedding, search_limit)

        # Filter to only 'who:' prefixed entries and extract memory IDs
        memory_scores = {}
        for item_id, score in results:
            if item_id.startswith('who:'):
                memory_id = item_id[4:]  # Remove 'who:' prefix
                # Normalize score to [0, 1]
                norm_score = (score + 1.0) / 2.0
                memory_scores[memory_id] = max(0.0, min(1.0, norm_score))

        # If no semantic matches found, fall back to exact match
        if not memory_scores:
            rows = self.store.get_by_actor(actor_hint, limit=topk)
            if rows:
                for row in rows:
                    memory_scores[row['memory_id']] = 0.5  # Medium confidence for exact match

        # Convert to list format
        results = list(memory_scores.items())[:topk]
        return results
    
    def _where_based(self, where_value: str, topk: int) -> List[Tuple[str, float]]:
        """Retrieve memories from specific WHERE location using semantic similarity.

        This now uses embeddings to find semantically similar locations,
        not just exact matches.
        """
        # Generate embedding for the location hint
        where_embedding = self.component_embedder.embed_where(where_value)
        if where_embedding is None:
            return []

        # Search for similar location embeddings in FAISS
        # Limit to reasonable number to avoid performance issues
        search_limit = min(topk * 2, 100)  # Cap at 100 for performance
        results = self.index.search(where_embedding, search_limit)

        # Filter to only 'where:' prefixed entries and extract memory IDs
        memory_scores = {}
        for item_id, score in results:
            if item_id.startswith('where:'):
                memory_id = item_id[6:]  # Remove 'where:' prefix
                # Normalize score to [0, 1]
                norm_score = (score + 1.0) / 2.0
                memory_scores[memory_id] = max(0.0, min(1.0, norm_score))

        # If no semantic matches found, fall back to exact match
        if not memory_scores:
            rows = self.store.get_by_location(where_value, limit=topk)
            if rows:
                for row in rows:
                    memory_scores[row['memory_id']] = 0.5  # Medium confidence for exact match

        # Convert to list format
        results = list(memory_scores.items())[:topk]
        return results
    
    def _temporal_based(self, temporal_hint: Union[str, Tuple[str, str], Dict], topk: int) -> List[Tuple[str, float]]:
        """Retrieve memories based on temporal hint.
        
        Supports:
        - Single date: "2024-01-15"
        - Date range: ("2024-01-10", "2024-01-20")
        - Relative time: {"relative": "yesterday"}
        """
        from typing import Union, Tuple, Dict
        
        # Parse temporal hint and retrieve memories
        if isinstance(temporal_hint, str):
            # Single date or timestamp - extract date part if needed
            if 'T' in temporal_hint:
                # Full timestamp like "2025-09-07T16:08:33.917577" - extract date
                date_part = temporal_hint.split('T')[0]
            else:
                date_part = temporal_hint
            rows = self.store.get_by_date(date_part, limit=topk)
        elif isinstance(temporal_hint, tuple) and len(temporal_hint) == 2:
            # Date range
            start, end = temporal_hint
            rows = self.store.get_by_date_range(start, end, limit=topk)
        elif isinstance(temporal_hint, dict):
            if "relative" in temporal_hint:
                # Relative time like "yesterday", "last_week"
                rows = self.store.get_by_relative_time(temporal_hint["relative"])
            elif "start" in temporal_hint and "end" in temporal_hint:
                # Timestamp range - extract dates
                start = temporal_hint["start"].split("T")[0] if "T" in temporal_hint["start"] else temporal_hint["start"]
                end = temporal_hint["end"].split("T")[0] if "T" in temporal_hint["end"] else temporal_hint["end"]
                rows = self.store.get_by_date_range(start, end, limit=topk)
            else:
                return []
        else:
            return []
        
        if not rows:
            return []
        
        # Score based on being in temporal window
        # All memories in the window get high base score (0.8)
        # with slight variation based on exact time for ranking
        results = []
        for i, row in enumerate(rows):
            # Higher score for earlier results (they're already sorted by time)
            score = 0.8 - (i * 0.001)  # Small decay for ranking
            results.append((row['memory_id'], score))
        
        return results

    def compute_all_scores(self,
                           memory_ids: List[str],
                           query_vec: np.ndarray,
                           query: RetrievalQuery,
                           sem_scores: Dict[str, float],
                           lex_scores: Dict[str, float],
                           actor_matches: Dict[str, float],
                           temporal_matches: Dict[str, float],
                           spatial_matches: Optional[Dict[str, float]] = None) -> List[Candidate]:
        """Compute comprehensive scores for all candidates across all dimensions."""
        
        # Fetch metadata for all candidates
        if not memory_ids:
            return []
        
        memories = self.store.fetch_memories(memory_ids)
        memory_dict = {m['memory_id']: m for m in memories}
        
        # Get usage stats for all candidates
        usage_stats = self.store.get_usage_stats(memory_ids)
        
        # Current time for recency calculation
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Group memories by topic/entity for smart recency application
        memory_groups = self._group_related_memories(memories, sem_scores, lex_scores)
        
        # Calculate candidates with all dimension scores
        candidates = []
        
        for memory_id in memory_ids:
            memory = memory_dict.get(memory_id)
            if not memory:
                continue
            
            # 1. Semantic similarity (already computed)
            semantic_score = sem_scores.get(memory_id, 0.0)
            
            # 2. Lexical match removed (FTS5 was broken)
            
            # 3. Smart recency score (only applies as tiebreaker for related memories)
            when_list = json.loads(memory.get('when_list', '[]')) if memory.get('when_list') else []
            when_ts = when_list[0] if when_list else memory.get('when_list', '[]')
            base_recency = exp_recency(when_ts, now, half_life_hours=168.0)
            recency_score = self._apply_smart_recency(memory_id, base_recency, memory_groups)
            
            # 4. Actor match (semantic similarity or from matches)
            actor_score = actor_matches.get(memory_id, 0.0)
            if not actor_score and query.actor_hint:
                # Generate semantic similarity score on the fly if needed
                # This is a fallback - ideally should be in actor_matches already
                who_data = self._extract_who_for_memory(memory)
                if who_data:
                    actor_hint_emb = self.component_embedder.embed_who(query.actor_hint)
                    who_emb = self.component_embedder.embed_who(who_data)
                    if actor_hint_emb is not None and who_emb is not None:
                        # Compute cosine similarity
                        similarity = np.dot(actor_hint_emb, who_emb)
                        # Already normalized, similarity is in [-1, 1]
                        actor_score = max(0.0, (similarity + 1.0) / 2.0)
            
            # 5. Temporal match (binary or from hint)
            temporal_score = temporal_matches.get(memory_id, 0.0)
            
            # 6. Spatial/location match (semantic similarity)
            spatial_score = 0.0
            if spatial_matches and memory_id in spatial_matches:
                spatial_score = spatial_matches[memory_id]
            else:
                where_value = memory.get('where_value')
                if where_value and where_value != 'unknown':
                    # Fallback: Check if location mentioned in query
                    if where_value.lower() in query.text.lower():
                        spatial_score = 0.7  # Good match but not semantic
            
            # Get usage data
            usage_data = usage_stats.get(memory_id, {})
            usage_score = min(1.0, usage_data.get('accesses', 0) / 50.0)  # Normalize by 50 accesses
            
            # Fixed weight combination 
            # Default weights (can be configured)
            # Note: recency is reduced since it's now a smart tiebreaker
            w_semantic = 0.68
            w_recency = 0.02  # Small weight - mainly acts as tiebreaker
            w_actor = 0.10
            w_temporal = 0.10
            w_spatial = 0.05
            w_usage = 0.05

            final_score = (
                w_semantic * semantic_score +
                w_recency * recency_score +
                w_actor * actor_score +
                w_temporal * temporal_score +
                w_spatial * spatial_score +
                w_usage * usage_score
            )
            
            candidate = Candidate(
                memory_id=memory_id,
                score=final_score,
                token_count=int(memory['token_count']),
                base_score=final_score,
                semantic_score=semantic_score,
                recency_score=recency_score,
                actor_score=actor_score,
                temporal_score=temporal_score,
                spatial_score=spatial_score,
                usage_score=usage_score
            )
            
            candidates.append(candidate)
        
        return candidates

    
    def _group_related_memories(self, memories: List[Dict], sem_scores: Dict[str, float], lex_scores: Dict[str, float]) -> Dict[str, List[str]]:
        """Group memories that are about the same topic/entity for smart recency application.
        
        Memories are considered related if they have:
        1. High semantic similarity to each other (>0.8)
        2. Removed (was lexical overlap)
        3. Same actor
        4. Overlapping key entities/topics
        """
        groups = {}  # group_id -> list of memory_ids
        memory_to_group = {}  # memory_id -> group_id
        
        # Sort memories by score to process highest scoring first
        sorted_memories = sorted(memories, key=lambda m: sem_scores.get(m['memory_id'], 0.0), reverse=True)
        
        for memory in sorted_memories:
            mid = memory['memory_id']
            assigned = False
            
            # Check if this memory is highly similar to any existing group
            for group_id, group_members in groups.items():
                # Check similarity with group representatives (first few members)
                for other_mid in group_members[:3]:  # Check against first 3 members
                    other_memory = next((m for m in memories if m['memory_id'] == other_mid), None)
                    if not other_memory:
                        continue
                    
                    # Check if memories are related
                    is_related = False
                    
                    # Get entities from the 'what' field (now stored as JSON array)
                    memory_entities = self._extract_memory_entities(memory)
                    other_entities = self._extract_memory_entities(other_memory)
                    
                    # Calculate overlap
                    entity_overlap = len(memory_entities & other_entities)
                    total_entities = len(memory_entities | other_entities)
                    
                    # Check different types of relatedness:
                    
                    # 1. Same actor discussing similar topic (updates/corrections)
                    if json.loads(memory.get('who_list', '[]'))[0] if json.loads(memory.get('who_list', '[]')) else '' == json.loads(other_memory.get('who_list', '[]'))[0] if json.loads(other_memory.get('who_list', '[]')) else '' and entity_overlap > 3:
                        is_related = True
                    
                    # 2. High overlap ratio (>40% of words in common) - likely same topic
                    elif total_entities > 0 and entity_overlap / total_entities > 0.4:
                        is_related = True
                    
                    # 3. Both memories highly relevant to query (top scorers discussing same thing)
                    elif (sem_scores.get(mid, 0) > 0.7 and sem_scores.get(other_mid, 0) > 0.7 and 
                          entity_overlap > 2):
                        is_related = True
                    
                    if is_related:
                        groups[group_id].append(mid)
                        memory_to_group[mid] = group_id
                        assigned = True
                        break
                
                if assigned:
                    break
            
            # If not assigned to any group, create new group
            if not assigned:
                group_id = f"group_{len(groups)}"
                groups[group_id] = [mid]
                memory_to_group[mid] = group_id
        
        return memory_to_group
    
    def _extract_memory_entities(self, memory: Dict) -> set:
        """Extract entities from a memory's 'what' field.
        
        The 'what' field can be:
        1. JSON array of entities (new format)
        2. JSON string containing array (stored format)
        3. Plain text (legacy format)
        """
        entities = set()
        
        if 'what' not in memory.keys() or not memory['what']:
            return entities
        
        what_field = memory['what']
        
        # Try to parse as JSON array
        try:
            # If it's already a list
            if isinstance(what_field, list):
                entities.update(str(e).lower() for e in what_field)
            # If it's a JSON string
            elif isinstance(what_field, str) and what_field.startswith('['):
                parsed = json.loads(what_field)
                if isinstance(parsed, list):
                    entities.update(str(e).lower() for e in parsed)
            # Legacy plain text format
            else:
                # Extract key words from text
                text = str(what_field).lower()
                # Split on common delimiters and filter short words
                words = re.split(r'[\s,;:.!?\'"()\[\]{}]+', text)
                entities.update(w for w in words if len(w) > 2)
        except (json.JSONDecodeError, ValueError):
            # Fallback to text extraction
            text = str(what_field).lower()
            words = re.split(r'[\s,;:.!?\'"()\[\]{}]+', text)
            entities.update(w for w in words if len(w) > 2)
        
        # Also add entities from 'why' field if it exists
        if 'why' in memory.keys() and memory['why']:
            why_text = str(memory['why']).lower()
            # Extract key terms from why
            words = re.split(r'[\s,;:.!?\'"()\[\]{}]+', why_text)
            entities.update(w for w in words if len(w) > 3)  # Slightly longer threshold for 'why'
        
        return entities
    
    def _apply_smart_recency(self, memory_id: str, base_recency: float, memory_groups: Dict[str, str]) -> float:
        """Apply recency as a tiebreaker only for memories in the same group.
        
        If a memory is in a group with other memories (same topic/entity),
        boost its recency score. Otherwise, use minimal recency impact.
        """
        group_id = memory_groups.get(memory_id)
        
        if not group_id:
            # Not in any group, minimal recency impact
            return base_recency * 0.1  # Heavily dampened
        
        # Count how many memories are in this group
        group_size = sum(1 for gid in memory_groups.values() if gid == group_id)
        
        if group_size <= 1:
            # Only memory in its group, minimal recency impact
            return base_recency * 0.1
        
        # Multiple memories about same topic - recency matters more
        # The more memories in the group, the more recency matters (competing information)
        recency_importance = min(1.0, group_size / 5.0)  # Max out at 5 related memories
        
        # Apply recency with importance factor
        return base_recency * (0.1 + 0.9 * recency_importance)
    
    def _fetch_meta(self, ids: List[str]):
        rows = self.store.fetch_memories(ids)
        by_id = {r['memory_id']: r for r in rows}
        return by_id

    # Old rerank method removed - using compute_all_scores instead
    def rerank_old(self, merged: Dict[str, float], rq: RetrievalQuery, 
               memory_embeddings: Optional[Dict[str, np.ndarray]] = None,
               temporal_candidate_ids: Optional[List[str]] = None,
               sem_scores: Optional[Dict[str, float]] = None,
               lex_scores: Optional[Dict[str, float]] = None,
               attention_scores: Optional[Dict[str, float]] = None) -> List[Candidate]:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        metas = self._fetch_meta(list(merged.keys()))
        cands = []
        
        # Get usage stats for adaptive scoring
        usage_stats = self.store.get_usage_stats(list(merged.keys())) if hasattr(self.store, 'get_usage_stats') else {}
        
        # Determine if we have hints to adjust weights
        has_actor_hint = bool(rq.actor_hint)
        has_temporal_hint = bool(rq.temporal_hint)
        
        # Check if this is a memory recall query
        is_recall_query = False
        recall_boost = 1.0
        query_lower = rq.text.lower()
        recall_indicators = ['remember', 'recall', 'memory', 'what do you know', 
                           'what did we discuss', 'find memories', 'is there any memory']
        for indicator in recall_indicators:
            if indicator in query_lower:
                is_recall_query = True
                recall_boost = 1.5  # Boost all matching memories for recall queries
                break
        
        for mid, base in merged.items():
            m = metas.get(mid)
            if not m:
                continue
                
            # Importance is now only used in attention reranking, not base scoring
            importance = 0.0  # Will be computed during attention phase if needed
            
            # Extra signals
            when_list_for_rec = json.loads(m.get('when_list', '[]'))
            rec_date = when_list_for_rec[0] if when_list_for_rec else ''
            rec = exp_recency(rec_date, now)
            who_list = json.loads(m.get('who_list', '[]'))
            first_who = who_list[0] if who_list else ''
            actor_match = 1.0 if (rq.actor_hint and first_who == rq.actor_hint) else 0.0
            
            # Check temporal match
            temporal_match = 0.0
            if has_temporal_hint:
                when_list = json.loads(m.get('when_list', '[]'))
                if when_list:
                    mem_date = when_list[0]
                    if 'T' in mem_date:
                        mem_date = mem_date.split('T')[0]
                else:
                    mem_date = ''
                
                if isinstance(rq.temporal_hint, str):
                    # Single date match
                    temporal_match = 1.0 if mem_date == rq.temporal_hint else 0.0
                elif isinstance(rq.temporal_hint, tuple) and len(rq.temporal_hint) == 2:
                    # Date range match
                    start, end = rq.temporal_hint
                    temporal_match = 1.0 if start <= mem_date <= end else 0.0
                elif isinstance(rq.temporal_hint, dict):
                    # For relative times, we'd need to compute the actual date range
                    # This is handled by the retrieval method, so memories retrieved
                    # via temporal_candidates already match
                    if temporal_candidate_ids and mid in temporal_candidate_ids:
                        temporal_match = 1.0
            
            # Get actual usage count if available
            usage_data = usage_stats.get(mid, {})
            usage_score = min(1.0, usage_data.get('accesses', 0) / 100.0) if usage_data else 0.0
            
            # SIMPLIFIED: Use minimal scoring - let attention handle the complexity
            # Just use base score (semantic/lexical) plus hint matches
            score = base
            
            # Add bonus for exact hint matches (but don't dominate the score)
            if actor_match > 0:
                score += 0.1  # Small boost for actor match
            if temporal_match > 0:
                score += 0.1  # Small boost for temporal match
            
            # Apply recall boost if this is a memory recall query
            if is_recall_query:
                score = score * recall_boost
                
            # Create candidate with component scores for debugging
            candidate = Candidate(
                memory_id=mid, 
                score=score, 
                token_count=int(m['token_count']),
                base_score=base,
                semantic_score=sem_scores.get(mid, 0.0) if sem_scores else None,
                recency_score=rec,
                importance_score=importance if self.use_attention else None,
                actor_score=actor_match,
                temporal_score=temporal_match,
                usage_score=usage_score,
                attention_score=attention_scores.get(mid, 0.0) if attention_scores else None
            )
            cands.append(candidate)
            
        cands.sort(key=lambda x: x.score, reverse=True)
        return cands

    def search(self, rq: RetrievalQuery, qvec: np.ndarray, topk_sem: int = 50, topk_lex: int = 50, enable_component_search: bool = True) -> List[Candidate]:
        # REDESIGNED: Retrieve large candidate set and score comprehensively
        # The topk parameters now only control the FINAL output size, not initial retrieval

        # Step 1: Get large candidate sets from each source (cast wide net)
        # Use topk_sem for retrieval size (will be 999999 from analyzer)
        initial_retrieval_size = topk_sem

        # Auto-extract actor hint from query if not provided
        if enable_component_search and not rq.actor_hint:
            rq.actor_hint = self._extract_actor_from_query(rq.text)

        # Get semantic candidates (vector similarity)
        sem = self._semantic(qvec, initial_retrieval_size)
        sem_dict = {mid: score for mid, score in sem}

        # Lexical search removed - using semantic only
        lex_dict = {}  # Empty dict for backward compatibility

        # Get actor-specific candidates if hint provided or extracted
        actor_candidates = []
        if enable_component_search and rq.actor_hint:
            # Limit actor search for performance
            actor_candidates = self._actor_based(rq.actor_hint, min(50, topk_sem // 10))
        actor_dict = {mid: score for mid, score in actor_candidates}  # Now using similarity scores
        
        # Get location-specific candidates if location detected in query
        spatial_candidates = []
        if enable_component_search:
            location_hint = self._extract_location_from_query(rq.text)
            if location_hint:
                # Limit location search for performance
                spatial_candidates = self._where_based(location_hint, min(50, topk_sem // 10))
        self._spatial_matches = {mid: score for mid, score in spatial_candidates}

        # Get temporal candidates if hint provided
        temporal_candidates = []
        if enable_component_search and rq.temporal_hint:
            # Limit temporal search for performance
            temporal_candidates = self._temporal_based(rq.temporal_hint, min(50, topk_sem // 10))
        temporal_dict = {mid: 1.0 for mid, _ in temporal_candidates}  # Binary match
        
        # Step 2: Combine all unique memory IDs
        all_memory_ids = set()
        all_memory_ids.update(sem_dict.keys())
        all_memory_ids.update(lex_dict.keys())
        all_memory_ids.update(actor_dict.keys())
        all_memory_ids.update(temporal_dict.keys())
        if hasattr(self, '_spatial_matches'):
            all_memory_ids.update(self._spatial_matches.keys())
        
        # Step 3: Compute comprehensive scores for all candidates
        candidates = self.compute_all_scores(
            memory_ids=list(all_memory_ids),
            query_vec=qvec,
            query=rq,
            sem_scores=sem_dict,
            lex_scores=lex_dict,
            actor_matches=actor_dict,
            temporal_matches=temporal_dict,
            spatial_matches=getattr(self, '_spatial_matches', {})
        )
        
        # Step 4: Apply final top-K selection
        candidates.sort(key=lambda x: x.score, reverse=True)
        final_k = min(topk_sem, len(candidates))
        top_k_candidates = candidates[:final_k]
        
        return top_k_candidates
    
    def get_current_weights(self) -> Dict[str, float]:
        """Return current weight configuration for UI display"""
        # Redistributed weights after removing lexical (was 0.25)
        return {
            'semantic': 0.68,
            'recency': 0.02,
            'actor': 0.10,
            'temporal': 0.10,
            'spatial': 0.05,
            'usage': 0.05
        }
    
    def update_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Validate and normalize weights, return normalized version"""
        # Ensure all weights are present
        default_weights = self.get_current_weights()
        for key in default_weights:
            if key not in weights:
                weights[key] = default_weights[key]
        
        # Normalize to sum to 1.0
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            weights = {k: v/total for k, v in weights.items()}
        
        return weights
    
    def get_detailed_scores(self, candidates: List[Candidate]) -> List[Dict]:
        """Return detailed breakdown for UI display with entity extraction"""
        if not candidates:
            return []
        
        # Fetch full memory details
        memory_ids = [c.memory_id for c in candidates]
        memories = self.store.fetch_memories(memory_ids)
        # Convert SQLite Row objects to dictionaries
        memory_dict = {dict(m)['memory_id']: dict(m) for m in memories}
        
        detailed = []
        for c in candidates:
            memory = memory_dict.get(c.memory_id)
            if not memory:
                continue
            
            # Extract entities from 'what' field
            entities = self.extract_entities_from_what(memory.get('what', ''))
            
            # Extract lists from JSON fields
            who_list = self.extract_list_from_json(memory.get('who_list', ''))
            when_list = self.extract_list_from_json(memory.get('when_list', ''))
            where_list = self.extract_list_from_json(memory.get('where_list', ''))
            
            detailed.append({
                'memory_id': c.memory_id,
                'raw_text': memory.get('raw_text', ''),
                'entities': entities,  # This is the 'what' list extracted/parsed
                'what': memory.get('what', ''),  # Raw what field from database
                'who': memory.get('who_list', '[]'),
                'who_id': memory.get('who_id', ''),
                'who_label': memory.get('who_label', ''),
                'who_type': memory.get('who_type', ''),
                'who_list': who_list,
                'when': memory.get('when_list', '[]'),
                'when_ts': memory.get('when_ts', ''),
                'when_list': when_list,
                'where': memory.get('where_value', ''),
                'where_type': memory.get('where_type', ''),
                'where_list': where_list,
                'why': memory.get('why', ''),
                'how': memory.get('how', ''),
                'scores': {
                    'total': c.score,
                    'semantic': c.semantic_score if c.semantic_score is not None else 0.0,
                    'recency': c.recency_score if c.recency_score is not None else 0.0,
                    'actor': c.actor_score if c.actor_score is not None else 0.0,
                    'temporal': c.temporal_score if c.temporal_score is not None else 0.0,
                    'spatial': c.spatial_score if hasattr(c, 'spatial_score') and c.spatial_score is not None else 0.0,
                    'usage': c.usage_score if c.usage_score is not None else 0.0
                },
                'token_count': c.token_count if c.token_count is not None else 0
            })
        
        return detailed
    
    def extract_list_from_json(self, json_field: str) -> List[str]:
        """Extract list from a JSON field, return empty list if None or invalid"""
        if not json_field:
            return []
        
        try:
            items = json.loads(json_field)
            if isinstance(items, list):
                return items
        except (json.JSONDecodeError, TypeError):
            pass
        
        return []
    
    def extract_entities_from_what(self, what_field: str) -> List[str]:
        """Extract entities from the 'what' field which may be JSON array or text"""
        if not what_field:
            return []
        
        # Try to parse as JSON array first
        try:
            entities = json.loads(what_field)
            if isinstance(entities, list):
                return entities
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback to simple entity extraction from text
        entities = []
        
        # Extract capitalized words (likely proper nouns)
        cap_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.finditer(cap_pattern, what_field):
            entity = match.group()
            if len(entity) > 2 and entity not in ['The', 'This', 'That', 'What', 'When', 'Where']:
                entities.append(entity)
        
        # Extract acronyms
        acronym_pattern = r'\b[A-Z]{2,}(?:-\d+)?\b'
        for match in re.finditer(acronym_pattern, what_field):
            entities.append(match.group())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)
        
        return unique_entities[:20]  # Limit to 20 entities
    
    def decompose_query(self, query: str) -> Dict[str, Any]:
        """Decompose query into 5W1H components using LLM extraction"""
        from .extraction.llm_extractor import extract_5w1h
        from .types import RawEvent
        
        # Create a raw event for the query
        raw_event = RawEvent(
            session_id='analyzer',
            event_type='user_message',
            actor='user:analyzer',
            content=query,
            metadata={}
        )
        
        try:
            # Extract 5W1H components
            extracted = extract_5w1h(raw_event)
            
            # Convert to dictionary format
            components = {
                'who': {
                    'type': extracted.who.type if hasattr(extracted, 'who') else None,
                    'id': extracted.who.id if hasattr(extracted, 'who') else None,
                    'label': extracted.who.label if hasattr(extracted, 'who') else None
                },
                'what': extracted.what if hasattr(extracted, 'what') else query,
                'when': str(extracted.when) if hasattr(extracted, 'when') else None,
                'where': {
                    'type': extracted.where.type if hasattr(extracted, 'where') else None,
                    'value': extracted.where.value if hasattr(extracted, 'where') else None
                },
                'why': extracted.why if hasattr(extracted, 'why') else None,
                'how': extracted.how if hasattr(extracted, 'how') else None
            }
            
            # Extract entities from the what field
            if components['what']:
                components['entities'] = self.extract_entities_from_what(components['what'])
            else:
                components['entities'] = []
                
        except Exception as e:
            # Fallback if extraction fails
            print(f"Query decomposition failed: {e}")
            components = {
                'who': {'type': None, 'id': None, 'label': None},
                'what': query,
                'when': None,
                'where': {'type': None, 'value': None},
                'why': None,
                'how': None,
                'entities': []
            }
        
        return components
    
    def search_with_weights(self, rq: RetrievalQuery, qvec: np.ndarray, weights: Dict[str, float],
                           topk_sem: int = 100, topk_lex: int = 100) -> List[Candidate]:
        """Search with custom weights provided by UI"""
        # Normalize weights
        weights = self.update_weights(weights)

        # Check if component searches are needed based on weights
        enable_components = (weights.get('actor', 0) > 0.01 or
                            weights.get('spatial', 0) > 0.01 or
                            weights.get('temporal', 0) > 0.01)

        # Use standard search with component control
        # topk_lex is ignored now since lexical search is removed
        candidates = self.search(rq, qvec, topk_sem=topk_sem, topk_lex=0,
                               enable_component_search=enable_components)

        # Re-score with custom weights
        for c in candidates:
            c.score = (
                weights['semantic'] * (c.semantic_score or 0.0) +
                weights['recency'] * (c.recency_score or 0.0) +
                weights['actor'] * (c.actor_score or 0.0) +
                weights['temporal'] * (c.temporal_score or 0.0) +
                weights['spatial'] * (getattr(c, 'spatial_score', 0.0) or 0.0) +
                weights['usage'] * (c.usage_score or 0.0)
            )

        # Re-sort and return all candidates (let knapsack algorithm handle selection)
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def _extract_who_for_memory(self, memory: Dict) -> Optional[Union[str, List, Dict]]:
        """Extract WHO data from a memory record."""
        # Try who_list first
        who_list = memory.get('who_list')
        if who_list:
            try:
                parsed = json.loads(who_list) if isinstance(who_list, str) else who_list
                if parsed:
                    return parsed
            except:
                pass

        # Try structured who fields
        if memory.get('who_label'):
            return {
                'id': memory.get('who_id'),
                'label': memory.get('who_label'),
                'type': memory.get('who_type')
            }

        # Fall back to who_id
        if memory.get('who_id'):
            return memory['who_id']

        return None

    def _extract_location_from_query(self, query_text: str) -> Optional[str]:
        """Extract potential location references from query text.

        This is a simple heuristic approach. Could be enhanced with NER.
        """
        # Common location indicators
        location_patterns = [
            r'\b(?:in|at|from|to|near|around|within|outside)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:office|building|room|city|country|state)',
            r'\b(?:location|place|venue|site)\s*[:"]\s*([^"]+)',
        ]

        for pattern in location_patterns:
            match = re.search(pattern, query_text)
            if match:
                location = match.group(1).strip()
                # Filter out common false positives
                if location and location.lower() not in ['the', 'this', 'that', 'what', 'when', 'where']:
                    return location

        # Look for known location keywords in the query
        words = query_text.split()
        for i, word in enumerate(words):
            # Check if word looks like a place name (capitalized)
            if word[0].isupper() and len(word) > 2:
                # Check context - is it preceded by location prepositions?
                if i > 0 and words[i-1].lower() in ['in', 'at', 'from', 'to', 'near']:
                    return word

        return None

    def _extract_actor_from_query(self, query_text: str) -> Optional[str]:
        """Extract potential actor/person references from query text.

        This is a simple heuristic approach. Could be enhanced with NER.
        """
        # Common patterns for identifying actors/people
        actor_patterns = [
            # Direct person references with possessive or action
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:'s|\s+(?:said|asked|mentioned|discussed|talked|wrote|did|was|is))\b",
            # Questions about specific people
            r"\b(?:what|when|where|how|why)\s+(?:did|does|was|is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            # Prepositions indicating people
            r"\b(?:by|from|with|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            # Direct "about X" pattern
            r"\babout\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            # Role-based references
            r"\b(?:the\s+)?(user|assistant|system|admin|developer|manager|team lead)\b",
        ]

        for pattern in actor_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                actor = match.group(1).strip()
                # Filter out common false positives
                if actor and actor.lower() not in ['the', 'this', 'that', 'what', 'when', 'where', 'how', 'why']:
                    # Check if it looks like a person name (capitalized words)
                    if actor[0].isupper() or actor.lower() in ['user', 'assistant', 'system', 'admin']:
                        return actor

        # Look for pronouns that might indicate user/assistant
        if any(word in query_text.lower() for word in ['i ', "i've", "i'm", 'my ', 'me ']):
            return 'user'
        if any(word in query_text.lower() for word in ['you ', "you've", "you're", 'your ']):
            return 'assistant'

        return None
