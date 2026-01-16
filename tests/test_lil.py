"""Tests for Language Interface Layer (LIL)."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.lil import (
    LanguageInterfaceLayer,
    MacroRegistry,
    PromptSegment,
    Role,
    StructuredPrompt,
    WireFormat,
    get_default_registry,
    compute_token_churn,
    compute_instruction_stability,
)


class TestStructuredPrompt:
    def test_create_simple_prompt(self):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=b"Hello world")
        ])
        assert len(prompt.segments) == 1
        assert prompt.segments[0].role == Role.USER
        assert prompt.segments[0].content == b"Hello world"

    def test_to_text_single_segment(self):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=b"test")
        ])
        text = prompt.to_text()
        assert "test" in text

    def test_to_text_multi_segment(self):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are helpful."),
            PromptSegment(role=Role.USER, content=b"Hi"),
        ])
        text = prompt.to_text()
        assert "You are helpful." in text
        assert "Hi" in text


class TestLILPackUnpack:
    @pytest.fixture
    def lil(self):
        return LanguageInterfaceLayer(enable_macros=False)

    @pytest.fixture
    def lil_with_macros(self):
        return LanguageInterfaceLayer(enable_macros=True)

    def test_pack_simple(self, lil):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=b"Hello world")
        ])
        result = lil.pack(prompt)
        assert result.data.startswith(WireFormat.MAGIC)
        assert result.original_size > 0
        assert result.packed_size > 0

    def test_unpack_simple(self, lil):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=b"Hello world")
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert len(unpacked.segments) == 1
        assert unpacked.segments[0].role == Role.USER
        assert unpacked.segments[0].content == b"Hello world"

    def test_roundtrip_single_segment(self, lil):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=b"What is 2+2?")
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert unpacked.to_text() == prompt.to_text()

    def test_roundtrip_multi_segment(self, lil):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are a helpful assistant."),
            PromptSegment(role=Role.USER, content=b"Hello"),
            PromptSegment(role=Role.ASSISTANT, content=b"Hi there!"),
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert unpacked.to_text() == prompt.to_text()

    def test_roundtrip_all_roles(self, lil):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"System message"),
            PromptSegment(role=Role.DEVELOPER, content=b"Developer message"),
            PromptSegment(role=Role.USER, content=b"User message"),
            PromptSegment(role=Role.ASSISTANT, content=b"Assistant message"),
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert unpacked.to_text() == prompt.to_text()
        assert len(unpacked.segments) == 4

    def test_roundtrip_binary_content(self, lil):
        # Content with all byte values
        content = bytes(range(256))
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=content)
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert unpacked.segments[0].content == content

    def test_roundtrip_special_markers(self, lil):
        # Content containing wire format markers
        content = WireFormat.MAGIC + WireFormat.SEGMENT_START + b"test"
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=content)
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert unpacked.segments[0].content == content

    def test_roundtrip_empty_content(self, lil):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=b"")
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert unpacked.segments[0].content == b""

    def test_roundtrip_unicode(self, lil):
        content = "Unicode: \u00e9\u00e0\u00f9 \u4e2d\u6587 \U0001f600".encode("utf-8")
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=content)
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert unpacked.segments[0].content == content


class TestMacroRegistry:
    def test_default_registry(self):
        registry = get_default_registry()
        assert len(registry) > 0

    def test_register_macro(self):
        from tokenizer.lil.interfaces import MacroDefinition
        registry = MacroRegistry()
        macro = MacroDefinition(
            id=0x99,
            name="test_macro",
            pattern=b"test pattern",
            description="Test"
        )
        registry.register(macro)
        assert registry.get_by_id(0x99) == macro
        assert registry.get_by_pattern(b"test pattern") == macro

    def test_duplicate_id_raises(self):
        from tokenizer.lil.interfaces import MacroDefinition
        registry = MacroRegistry()
        macro1 = MacroDefinition(id=0x99, name="m1", pattern=b"p1")
        macro2 = MacroDefinition(id=0x99, name="m2", pattern=b"p2")
        registry.register(macro1)
        with pytest.raises(ValueError):
            registry.register(macro2)

    def test_duplicate_pattern_raises(self):
        from tokenizer.lil.interfaces import MacroDefinition
        registry = MacroRegistry()
        macro1 = MacroDefinition(id=0x98, name="m1", pattern=b"same")
        macro2 = MacroDefinition(id=0x99, name="m2", pattern=b"same")
        registry.register(macro1)
        with pytest.raises(ValueError):
            registry.register(macro2)


class TestMacroApplication:
    @pytest.fixture
    def lil(self):
        return LanguageInterfaceLayer(enable_macros=True)

    def test_macro_application(self, lil):
        # Use a pattern that matches a default macro
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are a helpful assistant."),
            PromptSegment(role=Role.USER, content=b"Hello"),
        ])
        result = lil.pack(prompt)
        # Result should be smaller or equal (macros compact)
        assert result.packed_size <= result.original_size + 50  # Allow wire overhead

    def test_macro_reversal(self, lil):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are a helpful assistant."),
            PromptSegment(role=Role.USER, content=b"Hello"),
        ])
        result = lil.pack(prompt)
        unpacked = lil.unpack(result.data)
        assert unpacked.to_text() == prompt.to_text()

    def test_macro_logs(self, lil):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are a helpful assistant."),
        ])
        result = lil.pack(prompt)
        assert len(result.logs) > 0


class TestMetrics:
    def test_token_churn_identical(self):
        tokens = [1, 2, 3, 4, 5]
        churn = compute_token_churn(tokens, tokens)
        assert churn == 0.0

    def test_token_churn_different(self):
        tokens_a = [1, 2, 3, 4, 5]
        tokens_b = [6, 7, 8, 9, 10]
        churn = compute_token_churn(tokens_a, tokens_b)
        assert churn == 1.0

    def test_token_churn_partial(self):
        tokens_a = [1, 2, 3, 4, 5]
        tokens_b = [1, 2, 6, 4, 5]
        churn = compute_token_churn(tokens_a, tokens_b)
        assert 0 < churn < 1

    def test_token_churn_empty(self):
        churn = compute_token_churn([], [])
        assert churn == 0.0

    def test_instruction_stability(self):
        tokens_alone = [1, 2, 3, 4, 5]
        tokens_in_context = [1, 2, 3, 4, 5]
        churn, drift = compute_instruction_stability(tokens_alone, tokens_in_context)
        assert churn == 0.0


class TestLILIntegration:
    """Integration tests with the full tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        from tokenizer.config import load_config
        from tokenizer.bundle import UniversalTokenizer
        config = load_config(Path("configs/cpu_small.toml"))
        return UniversalTokenizer(config)

    @pytest.fixture
    def lil(self):
        return LanguageInterfaceLayer(enable_macros=True)

    def test_full_roundtrip(self, lil, tokenizer):
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.SYSTEM, content=b"You are a helpful assistant."),
            PromptSegment(role=Role.USER, content=b"What is 2+2?"),
        ])

        # Pack with LIL
        lil_result = lil.pack(prompt)

        # Encode with tokenizer
        enc = tokenizer.encode(lil_result.data)

        # Decode
        dec = tokenizer.decode(enc)

        # Unpack with LIL
        final = lil.unpack(dec.data)

        # Verify lossless
        assert final.to_text() == prompt.to_text()

    def test_full_roundtrip_code(self, lil, tokenizer):
        code_content = b"""```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
```"""
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=code_content)
        ])

        lil_result = lil.pack(prompt)
        enc = tokenizer.encode(lil_result.data)
        dec = tokenizer.decode(enc)
        final = lil.unpack(dec.data)

        assert final.to_text() == prompt.to_text()
