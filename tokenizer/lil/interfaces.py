"""Language Interface Layer (LIL) interfaces.

LIL is a reversible, deterministic transformation layer that operates above
the Universal Lossless Tokenizer. It optimizes language-model interface
structure (instructions, roles, schemas, prompts) rather than compression.

Key constraints:
- Reversible: Original text must be reconstructible exactly
- Deterministic: Same input -> same output
- Streaming-compatible: Bounded lookahead only
- Tokenizer-agnostic: Operates on bytes as pure transform
- Auditable: Every transformation step is explicit
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class Role(Enum):
    """Standard prompt roles."""
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class PromptSegment:
    """A single segment of a structured prompt.

    Attributes:
        role: The role of this segment (system, user, etc.)
        content: The raw content bytes
        metadata: Optional metadata for auditing/logging
    """
    role: Role
    content: bytes
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.content, str):
            self.content = self.content.encode("utf-8")


@dataclass
class StructuredPrompt:
    """A structured prompt with explicit role segments.

    Attributes:
        segments: List of prompt segments in order
        version: Schema version for forward compatibility
    """
    segments: list[PromptSegment]
    version: str = "1.0"

    def to_text(self) -> str:
        """Reconstruct plain text from segments."""
        parts = []
        for seg in self.segments:
            parts.append(seg.content.decode("utf-8"))
        return "\n".join(parts)

    def total_bytes(self) -> int:
        """Total content bytes across all segments."""
        return sum(len(seg.content) for seg in self.segments)


@dataclass
class MacroDefinition:
    """Definition of a reversible macro.

    Attributes:
        id: Unique identifier (0-255 for single-byte encoding)
        name: Human-readable name
        pattern: The pattern this macro represents (bytes)
        description: What this macro is for
    """
    id: int
    name: str
    pattern: bytes
    description: str = ""

    def __post_init__(self):
        if not 0 <= self.id <= 255:
            raise ValueError(f"Macro ID must be 0-255, got {self.id}")
        if isinstance(self.pattern, str):
            self.pattern = self.pattern.encode("utf-8")


@dataclass
class TransformLog:
    """Audit log for a transformation step.

    Attributes:
        operation: Name of the operation
        input_bytes: Input size
        output_bytes: Output size
        macros_applied: List of macro IDs applied
        reversible: Whether this transform is reversible
    """
    operation: str
    input_bytes: int
    output_bytes: int
    macros_applied: list[int] = field(default_factory=list)
    reversible: bool = True


@dataclass
class LILResult:
    """Result of a LIL transformation.

    Attributes:
        data: The transformed bytes
        logs: Audit trail of transformations
        original_size: Size before transformation
        packed_size: Size after transformation
    """
    data: bytes
    logs: list[TransformLog]
    original_size: int
    packed_size: int

    @property
    def size_ratio(self) -> float:
        """Packed size / original size."""
        return self.packed_size / self.original_size if self.original_size > 0 else 1.0


@runtime_checkable
class LILTransformer(Protocol):
    """Protocol for LIL transformers.

    All transforms must be reversible and deterministic.
    """

    def pack(self, prompt: StructuredPrompt) -> LILResult:
        """Pack a structured prompt into canonical byte representation.

        Args:
            prompt: The structured prompt to pack

        Returns:
            LILResult with packed bytes and audit log
        """
        ...

    def unpack(self, packed: bytes) -> StructuredPrompt:
        """Unpack bytes back to structured prompt.

        Args:
            packed: The packed byte representation

        Returns:
            The original structured prompt (exact reconstruction)
        """
        ...

    def apply_macros(self, data: bytes) -> LILResult:
        """Apply macro compaction to data.

        Args:
            data: Input bytes

        Returns:
            LILResult with macro-compacted bytes
        """
        ...

    def reverse_macros(self, data: bytes) -> bytes:
        """Reverse macro compaction.

        Args:
            data: Macro-compacted bytes

        Returns:
            Original bytes (exact reconstruction)
        """
        ...


# Wire format markers (chosen to be unlikely in normal text)
class WireFormat:
    """Constants for the LIL wire format."""

    # Magic bytes to identify LIL-packed data
    MAGIC = b"\x00LIL"
    VERSION = b"\x01"

    # Segment markers
    SEGMENT_START = b"\x02"
    SEGMENT_END = b"\x03"

    # Role markers (map to Role enum values)
    ROLE_MARKERS = {
        Role.SYSTEM: b"\x10",
        Role.DEVELOPER: b"\x11",
        Role.USER: b"\x12",
        Role.ASSISTANT: b"\x13",
    }
    MARKER_TO_ROLE = {v: k for k, v in ROLE_MARKERS.items()}

    # Macro marker (followed by macro ID byte)
    MACRO_START = b"\x04"
    MACRO_END = b"\x05"

    # Length encoding uses 4-byte big-endian for content length
    LENGTH_BYTES = 4

    # Escape byte for literal occurrences of markers in content
    ESCAPE = b"\x1B"  # ESC character
