"""
Base extractor with shared logic for 5W1H extraction.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import threading
import numpy as np
from ..types import RawEvent, MemoryRecord, Who, Where
from ..tokenization import TokenizerAdapter
from ..config import cfg

# Use llama.cpp embeddings
from ..embedding import get_llama_embedder

# Initialize embedder once at module level to avoid reloading
_embedder = None
_embedder_lock = threading.Lock()

def get_embedder():
    """Get or initialize the global embedder instance."""
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = get_llama_embedder()
    return _embedder


# Shared field extraction rules
FIELD_EXTRACTION_RULES = """
CRITICAL: Extract WHO and WHERE from the CONTENT ITSELF, not from chat metadata!

Extract for WHO (primary actor/subject IN THE CONTENT):
- Look for the MAIN subject performing actions or being discussed in the content
- "The AI assistant asks the user..." → who: {type: "llm", id: "AI assistant"}
- "The user requested help..." → who: {type: "user", id: "user"}
- "John Smith analyzed the data..." → who: {type: "person", id: "John Smith"}
- ONLY use metadata (Actor field) if the content provides NO clear subject

Extract for WHO_LIST: ALL people, organizations, entities mentioned in content
- "The CEO told the marketing team" → ["CEO", "marketing team"]
- "Alice and Bob from engineering" → ["Alice", "Bob", "engineering"]
- "The AI assistant asks the user" → ["AI assistant", "user"]

Extract for WHAT: Key entities, concepts, and topics from the content
- Names of people, organizations, teams, products
- Technical terms, genes, proteins, chemicals
- Programming languages, frameworks, tools
- Concepts, theories, methodologies
- Specific objects, places, or things
- Numbers, dates, measurements when significant

Extract for WHEN_LIST: Time expressions and temporal references
- "yesterday at 3pm during the meeting" → ["yesterday", "3pm", "during the meeting"]
- "last week's sprint" → ["last week", "sprint"]
- "Q3 2024 planning" → ["Q3 2024", "planning period"]

Extract for WHERE (location/context FROM THE CONTENT):
- Look for actual locations or contexts mentioned IN the text
- "regarding dinner prep" → {type: "conceptual", value: "dinner preparation context"}
- "in the kitchen" → {type: "physical", value: "kitchen"}
- "on the login page" → {type: "digital", value: "login page"}
- ONLY use "local_ui" if no location/context is mentioned in content

Extract for WHERE_LIST: Locations, places, and contexts
- "in the conference room at headquarters" → ["conference room", "headquarters"]
- "on GitHub in the main repository" → ["GitHub", "main repository"]
- "Seattle office's lab" → ["Seattle office", "lab"]

Examples:
- "The AI assistant asks the user to provide more context about dinner prep"
  → who: {type: "llm", id: "AI assistant"}, who_list: ["AI assistant", "user"],
    where: {type: "conceptual", value: "dinner preparation"}, what: ["context", "dinner prep"]
- "Python script for data analysis" → what: ["Python", "script", "data analysis"]
- "Player X was traded from Team A to Team B" → who: {type: "person", id: "Player X"},
    who_list: ["Player X"], what: ["trade", "sports transaction"], where_list: ["Team A", "Team B"]
"""

# Base schema for 5W1H extraction
BASE_SCHEMA = """
{
  "who": {
    "type": "<PRIMARY subject/actor mentioned IN THE CONTENT (not chat metadata): person, llm, user, organization, system, etc>",
    "id": "<identifier of the subject AS MENTIONED IN THE CONTENT>",
    "label": "<optional descriptive label from content>"
  },
  "who_list": ["<ALL people/entities/actors mentioned in the content>"],
  "what": ["<key topics/entities/concepts from the content>"],
  "when": "<ISO 8601 timestamp>",
  "when_list": ["<time expressions mentioned in content>"],
  "where": {
    "type": "<location type FROM CONTENT: physical, digital, conceptual, social, etc>",
    "value": "<SPECIFIC location/context mentioned IN THE CONTENT (not 'local_ui' unless content says so)>"
  },
  "where_list": ["<ALL locations/places/contexts mentioned in content>"],
  "why": "<intent or reason extracted from content - for memory recall queries use 'memory_recall: <topic>'>",
  "how": "<method/procedure mentioned in content>"
}
"""

MEMORY_RECALL_INSTRUCTIONS = """
Special instructions for the 'why' field:
- If user asks "do you remember...", "what do you know about...", "recall memories about...", "find memories of...", "is there any memory about...", set why to "memory_recall: <topic>"
- If user asks "what did we discuss about...", "what was said about...", set why to "memory_recall: <topic>"
- If user asks for past information, history, or previous discussions, set why to "memory_recall: <topic>"
"""


class BaseExtractor:
    """Base class for 5W1H extractors with shared functionality."""

    def __init__(self):
        self.embedder = get_embedder()
        self.tokenizer = TokenizerAdapter()

    def call_llm(self, system_prompt: str, user_content: str, max_tokens: int = 1024) -> Optional[Any]:
        """Call LLM with given prompts and return parsed JSON."""
        import requests
        url = f"{cfg.llm_base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        body = {
            "model": cfg.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        try:
            r = requests.post(url, headers=headers, json=body, timeout=60)
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"]

            # Try to parse JSON (could be object or array)
            # First try array
            start_arr = out.find('[')
            end_arr = out.rfind(']')
            if start_arr >= 0 and end_arr > start_arr:
                try:
                    obj = json.loads(out[start_arr:end_arr+1])
                    if isinstance(obj, list):
                        return obj
                except:
                    pass

            # Then try object
            start_obj = out.find('{')
            end_obj = out.rfind('}')
            if start_obj >= 0 and end_obj > start_obj:
                obj = json.loads(out[start_obj:end_obj+1])
                return obj

        except Exception as e:
            print(f"LLM extraction failed: {e}")
            return None
        return None

    def process_list_field(self, field_value: Any) -> Optional[str]:
        """Convert list field to JSON string for storage."""
        if isinstance(field_value, list) and field_value:
            return json.dumps(field_value)
        return None

    def process_what_field(self, what_raw: Any, fallback: str = "") -> str:
        """Process the 'what' field which should be an array of entities."""
        if isinstance(what_raw, list):
            return json.dumps(what_raw) if what_raw else '[]'
        else:
            # Fallback if LLM returns string instead of array
            return str(what_raw).strip() or fallback

    def create_embed_text(self, what: str, why: str, how: str, raw_content: str = "") -> str:
        """Create embedding text from memory fields."""
        embed_parts = [f"WHAT: {what}", f"WHY: {why}", f"HOW: {how}"]
        if raw_content:
            embed_parts.append(f"RAW: {raw_content[:500]}")  # Limit raw content length
        return "\n".join(embed_parts)

    def batch_create_embeddings(self, embed_texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for multiple texts in batch."""
        if not embed_texts:
            return []
        return self.embedder.encode(
            embed_texts,
            normalize_embeddings=True,
            batch_size=min(64, len(embed_texts)),
            show_progress_bar=False
        )

    def create_memory_record(
        self,
        raw: RawEvent,
        parsed: Dict[str, Any],
        embed_vec: np.ndarray,
        embed_text: str,
        part_suffix: str = "",
        part_index: Optional[int] = None,
        total_parts: Optional[int] = None
    ) -> MemoryRecord:
        """Create a MemoryRecord from parsed data."""
        # Process who field - trust the LLM extraction or use basic fallback
        who_data = parsed.get('who')
        if not who_data:
            # Basic fallback based on event type only
            if raw.event_type == 'user_message':
                who_data = {'type': 'user', 'id': raw.actor, 'label': 'User'}
            elif raw.event_type == 'llm_message':
                who_data = {'type': 'llm', 'id': raw.actor, 'label': 'Assistant'}
            else:
                who_data = {'type': 'system', 'id': raw.actor}
        who = Who(**who_data)

        # Process list fields
        who_list = self.process_list_field(parsed.get('who_list', []))
        when_list = self.process_list_field(parsed.get('when_list', []))
        where_list = self.process_list_field(parsed.get('where_list', []))

        # Process what field
        what = self.process_what_field(
            parsed.get('what', []),
            fallback=raw.content[:160]
        )

        # Process where field - trust the LLM extraction or use basic fallback
        where_data = parsed.get('where')
        if not where_data:
            # Basic fallback - no content parsing
            if raw.metadata.get('location'):
                where_data = {'type': 'digital', 'value': raw.metadata.get('location')}
            else:
                where_data = {'type': 'digital', 'value': 'conversation'}
        where = Where(
            type=where_data.get('type', 'digital'),
            value=where_data.get('value', 'conversation')
        )

        # Process why and how
        why = parsed.get('why', 'unspecified')
        how = parsed.get('how', 'message')

        # Calculate token count
        token_count = self.tokenizer.count_tokens(embed_text)

        # Determine raw text
        if part_index is not None and part_index > 0:
            raw_text = f"[Part {part_index + 1}] {json.loads(what)[0] if what != '[]' else what}"
        else:
            raw_text = raw.content

        # Create record
        rec = MemoryRecord(
            session_id=raw.session_id,
            source_event_id=f"{raw.event_id}{part_suffix}",
            who=who,
            who_list=who_list,
            what=what,
            when=raw.timestamp,
            when_list=when_list,
            where=where,
            where_list=where_list,
            why=why,
            how=how,
            raw_text=raw_text,
            token_count=token_count,
            embed_text=embed_text,
            embed_model=cfg.embed_model_name
        )

        # Add embedding and metadata to extras
        rec.extra['embed_vector_np'] = embed_vec.astype('float32').tolist()
        if part_index is not None:
            rec.extra['part_index'] = part_index
            rec.extra['total_parts'] = total_parts

        return rec

    def create_fallback_parsed(self, raw: RawEvent) -> Dict[str, Any]:
        """Create fallback parsed data when LLM extraction fails."""
        # Only use basic event type mapping - no content parsing
        # The LLM should handle all content extraction
        who_type = 'tool' if raw.event_type in ('tool_call', 'tool_result') else \
                   ('user' if raw.event_type == 'user_message' else \
                   ('llm' if raw.event_type == 'llm_message' else 'system'))

        return {
            'who': {'type': who_type, 'id': raw.actor, 'label': None},
            'what': [raw.content[:160]],  # Just use first part of content
            'where': {'type': 'digital', 'value': raw.metadata.get('location', 'conversation')},
            'why': raw.metadata.get('intent', 'unspecified'),
            'how': raw.metadata.get('method', 'message')
        }