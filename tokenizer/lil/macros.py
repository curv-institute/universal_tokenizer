"""Macro registry for LIL.

Provides a registry of reversible macros for common interface patterns.
Macros are simple pattern replacements that reduce verbosity of common
language model interface constructs.

Key principles:
- All macros are reversible (pattern <-> ID is bijective)
- Macros are deterministic (no context-dependent expansion)
- Macros are streaming-compatible (no lookahead beyond pattern)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .interfaces import MacroDefinition


class MacroRegistry:
    """Registry of reversible macros.

    Each macro maps a verbose pattern to a compact ID. The registry
    ensures all IDs are unique and patterns are non-overlapping.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._macros_by_id: dict[int, MacroDefinition] = {}
        self._macros_by_pattern: dict[bytes, MacroDefinition] = {}
        self._sorted_cache: list[MacroDefinition] | None = None

    def register(self, macro: MacroDefinition) -> None:
        """Register a new macro.

        Args:
            macro: The macro definition to register

        Raises:
            ValueError: If ID or pattern already exists
        """
        if macro.id in self._macros_by_id:
            raise ValueError(f"Macro ID {macro.id} already registered")
        if macro.pattern in self._macros_by_pattern:
            raise ValueError(f"Macro pattern already registered: {macro.pattern!r}")

        self._macros_by_id[macro.id] = macro
        self._macros_by_pattern[macro.pattern] = macro
        self._sorted_cache = None

    def get_by_id(self, macro_id: int) -> MacroDefinition | None:
        """Get macro by ID."""
        return self._macros_by_id.get(macro_id)

    def get_by_pattern(self, pattern: bytes) -> MacroDefinition | None:
        """Get macro by pattern."""
        return self._macros_by_pattern.get(pattern)

    def get_all_macros(self) -> list[MacroDefinition]:
        """Get all registered macros."""
        return list(self._macros_by_id.values())

    def get_macros_by_length(self) -> list[MacroDefinition]:
        """Get macros sorted by pattern length (longest first).

        This ordering is important for correct macro application - longer
        patterns should be matched before shorter ones to avoid ambiguity.
        """
        if self._sorted_cache is None:
            self._sorted_cache = sorted(
                self._macros_by_id.values(),
                key=lambda m: len(m.pattern),
                reverse=True,
            )
        return self._sorted_cache

    def __len__(self) -> int:
        """Number of registered macros."""
        return len(self._macros_by_id)


def get_default_registry() -> MacroRegistry:
    """Get the default macro registry with common interface patterns.

    Returns:
        MacroRegistry populated with standard macros
    """
    registry = MacroRegistry()

    # Role headers (common in chat formats)
    registry.register(MacroDefinition(
        id=0x20,
        name="system_header",
        pattern=b"<|system|>\n",
        description="System role header (OpenAI/Anthropic style)",
    ))
    registry.register(MacroDefinition(
        id=0x21,
        name="user_header",
        pattern=b"<|user|>\n",
        description="User role header",
    ))
    registry.register(MacroDefinition(
        id=0x22,
        name="assistant_header",
        pattern=b"<|assistant|>\n",
        description="Assistant role header",
    ))
    registry.register(MacroDefinition(
        id=0x23,
        name="developer_header",
        pattern=b"<|developer|>\n",
        description="Developer role header",
    ))

    # Alternative role markers (Llama style)
    registry.register(MacroDefinition(
        id=0x24,
        name="system_tag_open",
        pattern=b"[INST] <<SYS>>\n",
        description="Llama-style system tag open",
    ))
    registry.register(MacroDefinition(
        id=0x25,
        name="system_tag_close",
        pattern=b"\n<</SYS>>\n\n",
        description="Llama-style system tag close",
    ))
    registry.register(MacroDefinition(
        id=0x26,
        name="inst_close",
        pattern=b" [/INST]",
        description="Llama-style instruction close",
    ))

    # Common instruction boilerplate
    registry.register(MacroDefinition(
        id=0x30,
        name="helpful_assistant",
        pattern=b"You are a helpful assistant.",
        description="Common system prompt opener",
    ))
    registry.register(MacroDefinition(
        id=0x31,
        name="helpful_honest_harmless",
        pattern=b"You are a helpful, harmless, and honest assistant.",
        description="HHH alignment boilerplate",
    ))
    registry.register(MacroDefinition(
        id=0x32,
        name="think_step_by_step",
        pattern=b"Let's think step by step.",
        description="Chain-of-thought prompt",
    ))
    registry.register(MacroDefinition(
        id=0x33,
        name="answer_json",
        pattern=b"Please respond with valid JSON.",
        description="JSON response instruction",
    ))

    # Code fence markers
    registry.register(MacroDefinition(
        id=0x40,
        name="code_fence_python",
        pattern=b"```python\n",
        description="Python code fence open",
    ))
    registry.register(MacroDefinition(
        id=0x41,
        name="code_fence_close",
        pattern=b"\n```",
        description="Code fence close",
    ))
    registry.register(MacroDefinition(
        id=0x42,
        name="code_fence_json",
        pattern=b"```json\n",
        description="JSON code fence open",
    ))
    registry.register(MacroDefinition(
        id=0x43,
        name="code_fence_typescript",
        pattern=b"```typescript\n",
        description="TypeScript code fence open",
    ))

    # JSON schema patterns
    registry.register(MacroDefinition(
        id=0x50,
        name="json_schema_header",
        pattern=b'{"$schema": "https://json-schema.org/draft/2020-12/schema",',
        description="JSON Schema header",
    ))
    registry.register(MacroDefinition(
        id=0x51,
        name="type_object",
        pattern=b'"type": "object"',
        description="JSON Schema type: object",
    ))
    registry.register(MacroDefinition(
        id=0x52,
        name="type_string",
        pattern=b'"type": "string"',
        description="JSON Schema type: string",
    ))
    registry.register(MacroDefinition(
        id=0x53,
        name="type_array",
        pattern=b'"type": "array"',
        description="JSON Schema type: array",
    ))
    registry.register(MacroDefinition(
        id=0x54,
        name="required_prop",
        pattern=b'"required": true',
        description="JSON Schema required property",
    ))

    # Safety/formatting scaffolds
    registry.register(MacroDefinition(
        id=0x60,
        name="safety_disclaimer",
        pattern=b"I cannot help with that request.",
        description="Safety refusal pattern",
    ))
    registry.register(MacroDefinition(
        id=0x61,
        name="format_markdown",
        pattern=b"Please format your response in Markdown.",
        description="Markdown formatting instruction",
    ))
    registry.register(MacroDefinition(
        id=0x62,
        name="no_explanation",
        pattern=b"Do not include any explanation.",
        description="No explanation instruction",
    ))

    return registry


# Singleton default registry
_default_registry: MacroRegistry | None = None


def get_registry() -> MacroRegistry:
    """Get the singleton default registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = get_default_registry()
    return _default_registry
