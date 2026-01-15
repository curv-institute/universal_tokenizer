"""Universal Lossless Tokenizer.

A RIFT-instantiated tokenizer for bit-exact reconstruction of byte streams.
"""

__version__ = "0.1.0"

from .interfaces import (
    Token,
    EncodeResult,
    DecodeResult,
    TokenizerConfig,
    TokenizerBundle,
)

__all__ = [
    "__version__",
    "Token",
    "EncodeResult",
    "DecodeResult",
    "TokenizerConfig",
    "TokenizerBundle",
]
