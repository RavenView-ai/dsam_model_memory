"""
Unified 5W1H extractor that handles both single and multi-part extraction.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from ..types import RawEvent, MemoryRecord
from .base_extractor import (
    BaseExtractor,
    FIELD_EXTRACTION_RULES,
    BASE_SCHEMA,
    MEMORY_RECALL_INSTRUCTIONS
)


# Single extraction prompt
SINGLE_EXTRACTION_PROMPT = f"""You are a structured-information extractor that converts an interaction into 5W1H fields.
Return ONLY valid JSON in the following schema:

{BASE_SCHEMA}

CRITICAL: All list fields (who_list, what, when_list, where_list) must be JSON arrays containing relevant items.

{FIELD_EXTRACTION_RULES}

Consider the content and metadata; be concise but unambiguous.
{MEMORY_RECALL_INSTRUCTIONS}
"""


# Multi-part extraction prompt
MULTI_PART_PROMPT = f"""You are a structured-information extractor that identifies DISTINCT pieces of information and converts EACH into separate 5W1H fields.

IMPORTANT: Break down complex or multi-part information into SEPARATE, atomic memories.

For example, if the input contains:
- Multiple facts about different topics
- A list of items or events
- Multiple actions or outcomes
- Different pieces of information about the same topic

Create a SEPARATE memory for each distinct piece of information.

Return a JSON array where each element follows this schema:
{BASE_SCHEMA}

{FIELD_EXTRACTION_RULES}

Guidelines:
1. Each memory should be self-contained and independently searchable
2. Don't combine unrelated facts into one memory
3. Break lists into individual items
4. Separate different attributes of the same subject into different memories
5. Keep each "what" field focused on a single fact or action

{MEMORY_RECALL_INSTRUCTIONS}

Return ONLY the JSON array, no other text.
"""


class UnifiedExtractor(BaseExtractor):
    """Unified extractor that handles both single and multi-part extraction."""

    def extract_memories(
        self,
        raw: RawEvent,
        context_hint: str = "",
        max_parts: Optional[int] = None
    ) -> List[MemoryRecord]:
        """
        Extract one or more 5W1H memory records from a raw event.

        Args:
            raw: The raw event to process
            context_hint: Additional context for extraction
            max_parts: Maximum number of parts to extract (1 for single, None for auto)

        Returns:
            List of MemoryRecord objects
        """
        # Prepare content for LLM
        content = self._prepare_content(raw, context_hint)

        # Determine extraction mode
        if max_parts == 1:
            # Force single extraction
            parsed_list = self._extract_single(content)
        else:
            # Attempt multi-part extraction if content is complex
            should_multi = self._should_use_multi_part(raw)
            if should_multi:
                parsed_list = self._extract_multi(content)
            else:
                parsed_list = self._extract_single(content)

        # Fallback if extraction failed
        if not parsed_list:
            parsed_list = [self.create_fallback_parsed(raw)]

        # Limit parts if specified
        if max_parts and len(parsed_list) > max_parts:
            parsed_list = parsed_list[:max_parts]

        # Create embeddings and memory records
        return self._create_memory_records(raw, parsed_list)

    def extract_batch(
        self,
        raw_events: List[RawEvent],
        context_hints: Optional[List[str]] = None,
        max_parts_per_event: Optional[int] = None
    ) -> List[List[MemoryRecord]]:
        """
        Batch extract memories from multiple events for better performance.

        Args:
            raw_events: List of raw events to process
            context_hints: Optional list of context hints (one per event)
            max_parts_per_event: Maximum parts per event

        Returns:
            List of memory lists (one list per raw event)
        """
        if not context_hints:
            context_hints = [''] * len(raw_events)

        # Collect all parsed data first
        all_parsed_data = []
        for raw, hint in zip(raw_events, context_hints):
            content = self._prepare_content(raw, hint)

            # Determine extraction mode
            if max_parts_per_event == 1:
                parsed_list = self._extract_single(content)
            else:
                should_multi = self._should_use_multi_part(raw)
                if should_multi:
                    parsed_list = self._extract_multi(content)
                else:
                    parsed_list = self._extract_single(content)

            # Fallback if needed
            if not parsed_list:
                parsed_list = [self.create_fallback_parsed(raw)]

            # Limit parts if specified
            if max_parts_per_event and len(parsed_list) > max_parts_per_event:
                parsed_list = parsed_list[:max_parts_per_event]

            all_parsed_data.append((raw, parsed_list))

        # Batch process embeddings for efficiency
        return self._batch_create_memory_records(all_parsed_data)

    def _prepare_content(self, raw: RawEvent, context_hint: str) -> str:
        """Prepare content string for LLM extraction."""
        return (
            f"EventType: {raw.event_type}\n"
            f"Actor: {raw.actor}\n"
            f"Timestamp: {raw.timestamp.isoformat()}\n"
            f"Content: {raw.content}\n"
            f"Metadata: {raw.metadata}\n"
            f"Context: {context_hint}"
        )

    def _should_use_multi_part(self, raw: RawEvent) -> bool:
        """Determine if multi-part extraction should be used."""
        from ..config import cfg
        return (
            cfg.use_multi_part_extraction and (
                len(raw.content) > cfg.multi_part_threshold or
                '\n\n' in raw.content or
                raw.content.count('\n') > 3 or
                raw.event_type in ['llm_message', 'tool_result']
            )
        )

    def _extract_single(self, content: str) -> List[Dict[str, Any]]:
        """Extract a single memory structure."""
        result = self.call_llm(SINGLE_EXTRACTION_PROMPT, content, max_tokens=1024)
        if result:
            # Wrap single result in list for uniform processing
            if isinstance(result, dict):
                return [result]
            elif isinstance(result, list):
                return result[:1]  # Take only first if array returned
        return []

    def _extract_multi(self, content: str) -> List[Dict[str, Any]]:
        """Extract multiple memory structures."""
        result = self.call_llm(MULTI_PART_PROMPT, content, max_tokens=2048)
        if result:
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return [result]  # Single result, wrap in list
        return []

    def _create_memory_records(
        self,
        raw: RawEvent,
        parsed_list: List[Dict[str, Any]]
    ) -> List[MemoryRecord]:
        """Create memory records from parsed data."""
        memories = []

        # Prepare embedding texts
        embed_texts = []
        for idx, parsed in enumerate(parsed_list):
            what = self.process_what_field(
                parsed.get('what', []),
                fallback=f"Part {idx+1} of {raw.content[:100]}"
            )
            why = parsed.get('why', 'unspecified')
            how = parsed.get('how', 'message')

            # Only include raw content in first part's embedding
            raw_for_embed = raw.content if idx == 0 else ""
            embed_text = self.create_embed_text(what, why, how, raw_for_embed)
            embed_texts.append(embed_text)

        # Batch create embeddings
        embeddings = self.batch_create_embeddings(embed_texts)

        # Create memory records
        for idx, (parsed, embed_text, vec) in enumerate(zip(parsed_list, embed_texts, embeddings)):
            try:
                part_suffix = f"_part{idx}" if len(parsed_list) > 1 else ""
                rec = self.create_memory_record(
                    raw, parsed, vec, embed_text,
                    part_suffix=part_suffix,
                    part_index=idx if len(parsed_list) > 1 else None,
                    total_parts=len(parsed_list) if len(parsed_list) > 1 else None
                )
                memories.append(rec)
            except Exception as e:
                print(f"Failed to create memory {idx}: {e}")
                continue

        return memories

    def _batch_create_memory_records(
        self,
        all_parsed_data: List[tuple[RawEvent, List[Dict[str, Any]]]]
    ) -> List[List[MemoryRecord]]:
        """Batch create memory records for multiple events."""
        # Collect all embed texts for batch processing
        all_embed_texts = []
        embed_text_mapping = []  # Track which texts belong to which event/memory

        for event_idx, (raw, parsed_list) in enumerate(all_parsed_data):
            for memory_idx, parsed in enumerate(parsed_list):
                what = self.process_what_field(
                    parsed.get('what', []),
                    fallback=f"Part {memory_idx+1} of {raw.content[:100]}"
                )
                why = parsed.get('why', 'unspecified')
                how = parsed.get('how', 'message')

                # Only include raw content in first part's embedding
                raw_for_embed = raw.content if memory_idx == 0 else ""
                embed_text = self.create_embed_text(what, why, how, raw_for_embed)
                all_embed_texts.append(embed_text)
                embed_text_mapping.append((event_idx, memory_idx))

        # Batch encode all embeddings at once
        all_embeddings = self.batch_create_embeddings(all_embed_texts)

        # Build memory records using batch embeddings
        results = []
        embedding_idx = 0
        for event_idx, (raw, parsed_list) in enumerate(all_parsed_data):
            memories = []
            for memory_idx, parsed in enumerate(parsed_list):
                try:
                    # Get the corresponding embedding
                    embed_text = all_embed_texts[embedding_idx]
                    vec = all_embeddings[embedding_idx]
                    embedding_idx += 1

                    part_suffix = f"_part{memory_idx}" if len(parsed_list) > 1 else ""
                    rec = self.create_memory_record(
                        raw, parsed, vec, embed_text,
                        part_suffix=part_suffix,
                        part_index=memory_idx if len(parsed_list) > 1 else None,
                        total_parts=len(parsed_list) if len(parsed_list) > 1 else None
                    )
                    rec.extra['batch_processed'] = True
                    memories.append(rec)

                except Exception as e:
                    print(f"Failed to create memory for event {event_idx}, memory {memory_idx}: {e}")
                    continue

            results.append(memories)

        return results


# Convenience functions for backward compatibility
def extract_5w1h(raw: RawEvent, context_hint: str = "") -> MemoryRecord:
    """
    Extract a single 5W1H memory record from a raw event.
    Backward compatibility wrapper.
    """
    extractor = UnifiedExtractor()
    memories = extractor.extract_memories(raw, context_hint, max_parts=1)
    return memories[0] if memories else None


def extract_multi_part_5w1h(raw: RawEvent, context_hint: str = "") -> List[MemoryRecord]:
    """
    Extract multiple 5W1H memory records from a raw event.
    Backward compatibility wrapper.
    """
    extractor = UnifiedExtractor()
    return extractor.extract_memories(raw, context_hint, max_parts=None)


def extract_batch_5w1h(
    raw_events: List[RawEvent],
    context_hints: Optional[List[str]] = None
) -> List[List[MemoryRecord]]:
    """
    Batch extract memories from multiple events.
    Backward compatibility wrapper.
    """
    extractor = UnifiedExtractor()
    return extractor.extract_batch(raw_events, context_hints)
