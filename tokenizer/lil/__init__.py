"""Language Interface Layer (LIL) - Reversible interface optimization.

LIL is a reversible, deterministic transformation layer that operates above
the Universal Lossless Tokenizer. It optimizes language-model interface
structure (instructions, roles, schemas, prompts).

Example usage:
    from tokenizer.lil import LanguageInterfaceLayer, Role, StructuredPrompt, PromptSegment

    # Create a structured prompt
    prompt = StructuredPrompt(segments=[
        PromptSegment(role=Role.SYSTEM, content=b"You are a helpful assistant."),
        PromptSegment(role=Role.USER, content=b"What is 2+2?"),
    ])

    # Pack with LIL
    lil = LanguageInterfaceLayer()
    result = lil.pack(prompt)

    # Result is reversible
    unpacked = lil.unpack(result.data)
    assert unpacked.to_text() == prompt.to_text()
"""

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
from .lil import LanguageInterfaceLayer, get_lil
from .macros import MacroRegistry, get_default_registry, get_registry
from .metrics import (
    BoundaryMetrics,
    ChurnMetrics,
    LILEvalMetrics,
    SegmentMetrics,
    compute_boundary_curvature,
    compute_instruction_stability,
    compute_token_churn,
)

__all__ = [
    # Interfaces
    "LILResult",
    "LILTransformer",
    "MacroDefinition",
    "PromptSegment",
    "Role",
    "StructuredPrompt",
    "TransformLog",
    "WireFormat",
    # Implementation
    "LanguageInterfaceLayer",
    "get_lil",
    # Macros
    "MacroRegistry",
    "get_default_registry",
    "get_registry",
    # Metrics
    "BoundaryMetrics",
    "ChurnMetrics",
    "LILEvalMetrics",
    "SegmentMetrics",
    "compute_boundary_curvature",
    "compute_instruction_stability",
    "compute_token_churn",
]
