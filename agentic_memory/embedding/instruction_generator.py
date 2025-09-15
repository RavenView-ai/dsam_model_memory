"""
Dynamic instruction prefix generator for Qwen3-Embedding based on retrieval weights.

This module generates task-specific instructions that guide the embedding model
to focus on different aspects of retrieval based on the current weight configuration.
"""

from typing import Dict, Optional
import os


class InstructionGenerator:
    """Generates dynamic instruction prefixes based on retrieval weights."""

    def __init__(self):
        """Initialize the instruction generator with default templates."""
        # Base instruction templates for different retrieval focuses
        self.templates = {
            'semantic': "Retrieve passages with similar meaning and conceptual relevance",
            'recency': "Find recent events and time-sensitive information",
            'actor': "Search for memories involving specific people or entities",
            'spatial': "Locate memories related to specific places and locations",
            'temporal': "Find memories from specific time periods or dates",
            'usage': "Retrieve frequently accessed or important memories",
            'balanced': "Search for relevant memories considering multiple factors"
        }

        # Thresholds for determining primary focus
        self.threshold_dominant = 0.5  # Single weight > 50% means dominant
        self.threshold_significant = 0.25  # Weight > 25% is significant

    def generate_instruction(
        self,
        weights: Dict[str, float],
        query_context: Optional[str] = None,
        use_english: bool = True
    ) -> str:
        """
        Generate an instruction prefix based on retrieval weights.

        Args:
            weights: Dictionary of retrieval weights (semantic, recency, actor, etc.)
            query_context: Optional additional context about the query
            use_english: Whether to use English instructions (recommended by Qwen)

        Returns:
            Instruction string for the embedding model
        """
        # Normalize weights to ensure they sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}

        # Identify the primary focus based on weights
        primary_focus = self._identify_focus(weights)

        # Generate instruction based on focus
        instruction = self._build_instruction(primary_focus, weights, query_context)

        return instruction

    def _identify_focus(self, weights: Dict[str, float]) -> str:
        """
        Identify the primary retrieval focus based on weights.

        Args:
            weights: Normalized retrieval weights

        Returns:
            Primary focus type
        """
        # Check for dominant weight
        max_weight = max(weights.values())
        max_key = [k for k, v in weights.items() if v == max_weight][0]

        if max_weight > self.threshold_dominant:
            # Single dominant focus
            return max_key
        elif max_weight > self.threshold_significant:
            # Check for multi-aspect focus
            significant_aspects = [
                k for k, v in weights.items()
                if v > self.threshold_significant
            ]
            if len(significant_aspects) > 1:
                return 'multi_aspect'
            else:
                return max_key
        else:
            # Balanced search
            return 'balanced'

    def _build_instruction(
        self,
        focus: str,
        weights: Dict[str, float],
        query_context: Optional[str] = None
    ) -> str:
        """
        Build the instruction string based on focus and weights.

        Args:
            focus: Primary focus type
            weights: Retrieval weights
            query_context: Optional additional context

        Returns:
            Complete instruction string
        """
        if focus == 'multi_aspect':
            # Build multi-aspect instruction
            significant = [
                (k, v) for k, v in weights.items()
                if v > self.threshold_significant
            ]
            significant.sort(key=lambda x: x[1], reverse=True)

            aspects = []
            for aspect, weight in significant[:3]:  # Top 3 aspects
                if aspect == 'semantic':
                    aspects.append("conceptual similarity")
                elif aspect == 'recency':
                    aspects.append("recent timeframe")
                elif aspect == 'actor':
                    aspects.append("specific people involved")
                elif aspect == 'spatial':
                    aspects.append("location relevance")
                elif aspect == 'temporal':
                    aspects.append("time period")
                elif aspect == 'usage':
                    aspects.append("access frequency")

            instruction = f"Find memories with strong {' and '.join(aspects)}"

        elif focus in self.templates:
            instruction = self.templates[focus]

        else:
            # Default balanced instruction
            instruction = self.templates['balanced']

        # Add query context if provided
        if query_context:
            instruction = f"{instruction}, focusing on {query_context}"

        return instruction

    def generate_component_instruction(
        self,
        component_type: str,
        entity_value: str
    ) -> str:
        """
        Generate instruction for component-specific embedding.

        Args:
            component_type: Type of component (who, where, what)
            entity_value: The entity being embedded

        Returns:
            Instruction for component embedding
        """
        if component_type == 'who':
            return f"Match memories involving this person or entity: {entity_value}"
        elif component_type == 'where':
            return f"Find memories from this location: {entity_value}"
        elif component_type == 'what':
            return f"Retrieve memories about this topic or action: {entity_value}"
        else:
            return f"Find relevant memories for: {entity_value}"


# Singleton instance
_instruction_generator: Optional[InstructionGenerator] = None


def get_instruction_generator() -> InstructionGenerator:
    """Get or create the singleton instruction generator."""
    global _instruction_generator
    if _instruction_generator is None:
        _instruction_generator = InstructionGenerator()
    return _instruction_generator


def generate_query_instruction(weights: Dict[str, float]) -> str:
    """
    Quick helper to generate instruction from weights.

    Args:
        weights: Retrieval weights dictionary

    Returns:
        Instruction string
    """
    generator = get_instruction_generator()
    return generator.generate_instruction(weights)
