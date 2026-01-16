"""Language Interface Layer (LIL) implementation.

Provides reversible, deterministic transformation of structured prompts
into a canonical byte representation suitable for tokenization.
"""

from __future__ import annotations

import struct
from typing import Iterator

from .interfaces import (
    LILResult,
    LILTransformer,
    MacroDefinition,
    PromptSegment,
    Role,
    StructuredPrompt,
    TransformLog,
    WireFormat,
)
from .macros import MacroRegistry, get_default_registry


class LanguageInterfaceLayer:
    """Main LIL transformer implementation.

    This class provides reversible packing/unpacking of structured prompts
    and optional macro compaction for common interface patterns.

    Attributes:
        registry: The macro registry for compaction
        enable_macros: Whether to apply macro compaction
    """

    def __init__(
        self,
        registry: MacroRegistry | None = None,
        enable_macros: bool = True,
    ):
        """Initialize the LIL transformer.

        Args:
            registry: Optional custom macro registry (uses default if None)
            enable_macros: Whether to enable macro compaction
        """
        self.registry = registry or get_default_registry()
        self.enable_macros = enable_macros

    def _escape_content(self, content: bytes) -> bytes:
        """Escape special markers in content.

        Any occurrence of wire format markers in content must be escaped
        to ensure unambiguous parsing.

        Args:
            content: Raw content bytes

        Returns:
            Escaped content bytes
        """
        # Characters that need escaping
        special = set()
        special.add(WireFormat.ESCAPE[0])
        special.add(WireFormat.SEGMENT_START[0])
        special.add(WireFormat.SEGMENT_END[0])
        special.add(WireFormat.MACRO_START[0])
        special.add(WireFormat.MACRO_END[0])
        for marker in WireFormat.ROLE_MARKERS.values():
            special.add(marker[0])

        # Escape by prefixing with ESCAPE byte
        result = bytearray()
        for byte in content:
            if byte in special:
                result.append(WireFormat.ESCAPE[0])
            result.append(byte)
        return bytes(result)

    def _unescape_content(self, content: bytes) -> bytes:
        """Reverse the escape transformation.

        Args:
            content: Escaped content bytes

        Returns:
            Original content bytes
        """
        result = bytearray()
        i = 0
        while i < len(content):
            if content[i:i+1] == WireFormat.ESCAPE and i + 1 < len(content):
                # Skip escape, take next byte literally
                result.append(content[i + 1])
                i += 2
            else:
                result.append(content[i])
                i += 1
        return bytes(result)

    def pack(self, prompt: StructuredPrompt) -> LILResult:
        """Pack a structured prompt into canonical byte representation.

        Wire format:
        - MAGIC (4 bytes): "\x00LIL"
        - VERSION (1 byte)
        - NUM_SEGMENTS (2 bytes, big-endian)
        - For each segment:
          - SEGMENT_START (1 byte)
          - ROLE_MARKER (1 byte)
          - CONTENT_LENGTH (4 bytes, big-endian)
          - ESCAPED_CONTENT (variable)
          - SEGMENT_END (1 byte)

        Args:
            prompt: The structured prompt to pack

        Returns:
            LILResult with packed bytes and audit log
        """
        logs: list[TransformLog] = []
        original_size = prompt.total_bytes()

        # Build packed representation
        packed = bytearray()

        # Header
        packed.extend(WireFormat.MAGIC)
        packed.extend(WireFormat.VERSION)
        packed.extend(struct.pack(">H", len(prompt.segments)))

        # Segments
        for segment in prompt.segments:
            packed.extend(WireFormat.SEGMENT_START)
            packed.extend(WireFormat.ROLE_MARKERS[segment.role])

            escaped = self._escape_content(segment.content)
            packed.extend(struct.pack(">I", len(escaped)))
            packed.extend(escaped)
            packed.extend(WireFormat.SEGMENT_END)

        packed_bytes = bytes(packed)
        logs.append(TransformLog(
            operation="pack",
            input_bytes=original_size,
            output_bytes=len(packed_bytes),
            reversible=True,
        ))

        # Optionally apply macros
        if self.enable_macros:
            macro_result = self.apply_macros(packed_bytes)
            packed_bytes = macro_result.data
            logs.extend(macro_result.logs)

        return LILResult(
            data=packed_bytes,
            logs=logs,
            original_size=original_size,
            packed_size=len(packed_bytes),
        )

    def unpack(self, packed: bytes) -> StructuredPrompt:
        """Unpack bytes back to structured prompt.

        Args:
            packed: The packed byte representation

        Returns:
            The original structured prompt (exact reconstruction)

        Raises:
            ValueError: If the packed data is invalid
        """
        # Reverse macros if needed
        if self.enable_macros:
            packed = self.reverse_macros(packed)

        # Parse header
        if not packed.startswith(WireFormat.MAGIC):
            raise ValueError("Invalid LIL data: missing magic bytes")

        pos = len(WireFormat.MAGIC)

        version = packed[pos:pos + 1]
        if version != WireFormat.VERSION:
            raise ValueError(f"Unsupported LIL version: {version!r}")
        pos += 1

        num_segments = struct.unpack(">H", packed[pos:pos + 2])[0]
        pos += 2

        # Parse segments
        segments: list[PromptSegment] = []
        for _ in range(num_segments):
            # Segment start
            if packed[pos:pos + 1] != WireFormat.SEGMENT_START:
                raise ValueError(f"Expected segment start at position {pos}")
            pos += 1

            # Role marker
            role_marker = packed[pos:pos + 1]
            if role_marker not in WireFormat.MARKER_TO_ROLE:
                raise ValueError(f"Invalid role marker: {role_marker!r}")
            role = WireFormat.MARKER_TO_ROLE[role_marker]
            pos += 1

            # Content length
            content_len = struct.unpack(">I", packed[pos:pos + 4])[0]
            pos += 4

            # Escaped content
            escaped_content = packed[pos:pos + content_len]
            pos += content_len

            # Segment end
            if packed[pos:pos + 1] != WireFormat.SEGMENT_END:
                raise ValueError(f"Expected segment end at position {pos}")
            pos += 1

            # Unescape and create segment
            content = self._unescape_content(escaped_content)
            segments.append(PromptSegment(role=role, content=content))

        return StructuredPrompt(segments=segments)

    def apply_macros(self, data: bytes) -> LILResult:
        """Apply macro compaction to data.

        Scans for patterns in the registry and replaces them with
        compact macro references.

        Args:
            data: Input bytes

        Returns:
            LILResult with macro-compacted bytes
        """
        macros_applied: list[int] = []
        result = data

        # Apply macros in order of pattern length (longest first)
        for macro in self.registry.get_macros_by_length():
            if macro.pattern in result:
                # Replace pattern with macro reference
                macro_ref = WireFormat.MACRO_START + bytes([macro.id]) + WireFormat.MACRO_END
                result = result.replace(macro.pattern, macro_ref)
                macros_applied.append(macro.id)

        return LILResult(
            data=result,
            logs=[TransformLog(
                operation="apply_macros",
                input_bytes=len(data),
                output_bytes=len(result),
                macros_applied=macros_applied,
                reversible=True,
            )],
            original_size=len(data),
            packed_size=len(result),
        )

    def reverse_macros(self, data: bytes) -> bytes:
        """Reverse macro compaction.

        Args:
            data: Macro-compacted bytes

        Returns:
            Original bytes (exact reconstruction)
        """
        result = data

        # Find and replace all macro references
        for macro in self.registry.get_all_macros():
            macro_ref = WireFormat.MACRO_START + bytes([macro.id]) + WireFormat.MACRO_END
            result = result.replace(macro_ref, macro.pattern)

        return result

    def transform(self, text: str, roles: list[tuple[Role, str]] | None = None) -> LILResult:
        """Convenience method to transform text with optional role structure.

        Args:
            text: Plain text to transform
            roles: Optional list of (role, content) tuples. If None, treats
                   entire text as a single USER segment.

        Returns:
            LILResult with packed bytes
        """
        if roles is None:
            segments = [PromptSegment(role=Role.USER, content=text.encode("utf-8"))]
        else:
            segments = [
                PromptSegment(role=role, content=content.encode("utf-8"))
                for role, content in roles
            ]

        prompt = StructuredPrompt(segments=segments)
        return self.pack(prompt)

    def inverse_transform(self, packed: bytes) -> str:
        """Convenience method to inverse transform packed bytes to text.

        Args:
            packed: LIL-packed bytes

        Returns:
            Original text (exact reconstruction)
        """
        prompt = self.unpack(packed)
        return prompt.to_text()


# Module-level convenience instance
_default_lil: LanguageInterfaceLayer | None = None


def get_lil() -> LanguageInterfaceLayer:
    """Get the default LIL instance."""
    global _default_lil
    if _default_lil is None:
        _default_lil = LanguageInterfaceLayer()
    return _default_lil
