"""
Component embedder for 5W1H fields.
Generates separate embeddings for who, where, and other components to enable semantic search.
"""
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import json
import logging
from .llama_embedder import get_llama_embedder

logger = logging.getLogger(__name__)

class ComponentEmbedder:
    """Generates embeddings for 5W1H components."""

    def __init__(self):
        self.embedder = get_llama_embedder()

    def embed_who(self, who_data: Union[str, List[str], Dict]) -> Optional[np.ndarray]:
        """Generate embedding for WHO component.

        Args:
            who_data: Can be:
                - String: Single actor name/ID
                - List: Multiple actors
                - Dict: Structured who data with id, label, type

        Returns:
            Embedding vector or None if no valid data
        """
        text_parts = []

        if isinstance(who_data, str):
            if who_data:
                text_parts.append(who_data)
        elif isinstance(who_data, list):
            text_parts.extend([str(w) for w in who_data if w])
        elif isinstance(who_data, dict):
            # Handle structured who data
            if who_data.get('label'):
                text_parts.append(who_data['label'])
            if who_data.get('id') and who_data['id'] != who_data.get('label'):
                text_parts.append(who_data['id'])
            if who_data.get('type'):
                text_parts.append(f"type:{who_data['type']}")

        if not text_parts:
            return None

        # Create a descriptive text for embedding
        who_text = f"Actor: {' '.join(text_parts)}"
        embedding = self.embedder.encode([who_text], normalize_embeddings=True)[0]
        return embedding

    def embed_where(self, where_data: Union[str, List[str], Dict]) -> Optional[np.ndarray]:
        """Generate embedding for WHERE component.

        Args:
            where_data: Can be:
                - String: Location name
                - List: Multiple locations
                - Dict: Structured location with type, value, coordinates

        Returns:
            Embedding vector or None if no valid data
        """
        text_parts = []

        if isinstance(where_data, str):
            if where_data and where_data != "unknown":
                text_parts.append(where_data)
        elif isinstance(where_data, list):
            text_parts.extend([str(w) for w in where_data if w and w != "unknown"])
        elif isinstance(where_data, dict):
            # Handle structured location data
            if where_data.get('value') and where_data['value'] != "unknown":
                text_parts.append(where_data['value'])
            if where_data.get('type'):
                text_parts.append(f"type:{where_data['type']}")
            # Could add coordinate info if needed

        if not text_parts:
            return None

        # Create a descriptive text for embedding
        where_text = f"Location: {' '.join(text_parts)}"
        embedding = self.embedder.encode([where_text], normalize_embeddings=True)[0]
        return embedding

    def embed_when(self, when_data: Union[str, List[str]]) -> Optional[np.ndarray]:
        """Generate embedding for WHEN component.

        Args:
            when_data: Timestamp or list of timestamps

        Returns:
            Embedding vector or None if no valid data
        """
        text_parts = []

        if isinstance(when_data, str):
            if when_data:
                # Extract semantic time information
                text_parts.append(self._parse_temporal_context(when_data))
        elif isinstance(when_data, list):
            for ts in when_data:
                if ts:
                    text_parts.append(self._parse_temporal_context(str(ts)))

        if not text_parts:
            return None

        # Create temporal context embedding
        when_text = f"Time: {' '.join(text_parts)}"
        embedding = self.embedder.encode([when_text], normalize_embeddings=True)[0]
        return embedding

    def embed_what(self, what_data: Union[str, List[str]]) -> Optional[np.ndarray]:
        """Generate embedding for WHAT component.

        This is typically the main content embedding but focused on entities/actions.

        Args:
            what_data: Event description or list of entities

        Returns:
            Embedding vector or None if no valid data
        """
        text_parts = []

        if isinstance(what_data, str):
            if what_data:
                # Try to parse as JSON array first
                try:
                    entities = json.loads(what_data)
                    if isinstance(entities, list):
                        text_parts.extend([str(e) for e in entities])
                    else:
                        text_parts.append(what_data)
                except:
                    text_parts.append(what_data)
        elif isinstance(what_data, list):
            text_parts.extend([str(w) for w in what_data if w])

        if not text_parts:
            return None

        # Create entity-focused embedding
        what_text = ' '.join(text_parts)
        embedding = self.embedder.encode([what_text], normalize_embeddings=True)[0]
        return embedding

    def embed_why(self, why_data: str) -> Optional[np.ndarray]:
        """Generate embedding for WHY component (reason/purpose).

        Args:
            why_data: Reason or purpose text

        Returns:
            Embedding vector or None if no valid data
        """
        if not why_data or why_data == "unknown":
            return None

        why_text = f"Reason: {why_data}"
        embedding = self.embedder.encode([why_text], normalize_embeddings=True)[0]
        return embedding

    def embed_how(self, how_data: str) -> Optional[np.ndarray]:
        """Generate embedding for HOW component (method/process).

        Args:
            how_data: Method or process description

        Returns:
            Embedding vector or None if no valid data
        """
        if not how_data or how_data == "unknown":
            return None

        how_text = f"Method: {how_data}"
        embedding = self.embedder.encode([how_text], normalize_embeddings=True)[0]
        return embedding

    def embed_all_components(self, memory_data: Dict) -> Dict[str, Optional[np.ndarray]]:
        """Generate embeddings for all 5W1H components from memory data.

        Args:
            memory_data: Dictionary containing 5W1H fields

        Returns:
            Dictionary mapping component names to embeddings
        """
        embeddings = {}

        # Extract and embed WHO
        who_data = self._extract_who_data(memory_data)
        if who_data:
            embeddings['who'] = self.embed_who(who_data)
        else:
            embeddings['who'] = None

        # Extract and embed WHERE
        where_data = self._extract_where_data(memory_data)
        if where_data:
            embeddings['where'] = self.embed_where(where_data)
        else:
            embeddings['where'] = None

        # Extract and embed WHEN
        when_data = memory_data.get('when_list') or memory_data.get('when_ts')
        if when_data:
            embeddings['when'] = self.embed_when(when_data)
        else:
            embeddings['when'] = None

        # Extract and embed WHAT
        what_data = memory_data.get('what')
        if what_data:
            embeddings['what'] = self.embed_what(what_data)
        else:
            embeddings['what'] = None

        # Extract and embed WHY
        why_data = memory_data.get('why')
        if why_data:
            embeddings['why'] = self.embed_why(why_data)
        else:
            embeddings['why'] = None

        # Extract and embed HOW
        how_data = memory_data.get('how')
        if how_data:
            embeddings['how'] = self.embed_how(how_data)
        else:
            embeddings['how'] = None

        return embeddings

    def _extract_who_data(self, memory_data: Dict) -> Optional[Union[Dict, List, str]]:
        """Extract WHO data from memory, handling various formats."""
        # Try structured format first
        if 'who_label' in memory_data:
            who_dict = {
                'id': memory_data.get('who_id'),
                'label': memory_data.get('who_label'),
                'type': memory_data.get('who_type')
            }
            if any(who_dict.values()):
                return who_dict

        # Try list format
        if 'who_list' in memory_data:
            who_list = memory_data['who_list']
            if isinstance(who_list, str):
                try:
                    who_list = json.loads(who_list)
                except:
                    pass
            if who_list:
                return who_list

        # Fall back to who_id
        if 'who_id' in memory_data and memory_data['who_id']:
            return memory_data['who_id']

        return None

    def _extract_where_data(self, memory_data: Dict) -> Optional[Union[Dict, List, str]]:
        """Extract WHERE data from memory, handling various formats."""
        # Try structured format
        where_dict = {}
        if 'where_value' in memory_data and memory_data['where_value'] != 'unknown':
            where_dict['value'] = memory_data['where_value']
        if 'where_type' in memory_data:
            where_dict['type'] = memory_data['where_type']
        if 'where_lat' in memory_data and 'where_lon' in memory_data:
            where_dict['coordinates'] = (memory_data['where_lat'], memory_data['where_lon'])

        if where_dict.get('value'):
            return where_dict

        # Try list format
        if 'where_list' in memory_data:
            where_list = memory_data['where_list']
            if isinstance(where_list, str):
                try:
                    where_list = json.loads(where_list)
                except:
                    pass
            if where_list:
                return where_list

        return None

    def _parse_temporal_context(self, timestamp: str) -> str:
        """Parse timestamp into semantic temporal context.

        Converts timestamps into more semantic representations like
        'morning', 'evening', 'weekend', 'january', etc.
        """
        try:
            from datetime import datetime

            # Parse various timestamp formats
            if 'T' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # Assume date only
                dt = datetime.fromisoformat(timestamp + 'T00:00:00')

            # Extract semantic time features
            parts = []

            # Time of day
            hour = dt.hour
            if 5 <= hour < 12:
                parts.append("morning")
            elif 12 <= hour < 17:
                parts.append("afternoon")
            elif 17 <= hour < 21:
                parts.append("evening")
            else:
                parts.append("night")

            # Day of week
            weekday = dt.weekday()
            if weekday < 5:
                parts.append("weekday")
            else:
                parts.append("weekend")

            # Month
            parts.append(dt.strftime("%B").lower())

            # Year
            parts.append(str(dt.year))

            return ' '.join(parts)

        except Exception as e:
            logger.debug(f"Could not parse temporal context from {timestamp}: {e}")
            return timestamp

# Global instance
_component_embedder = None

def get_component_embedder() -> ComponentEmbedder:
    """Get or create the global ComponentEmbedder instance."""
    global _component_embedder
    if _component_embedder is None:
        _component_embedder = ComponentEmbedder()
    return _component_embedder