#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "numpy",
# ]
# ///
"""LIL encode script - Apply LIL then encode with ULT.

This script demonstrates the LIL pipeline:
1. Parse input into structured prompt
2. Apply LIL transformation (pack with macros)
3. Encode with Universal Lossless Tokenizer
4. Verify round-trip reversibility
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.lil import (
    LanguageInterfaceLayer,
    PromptSegment,
    Role,
    StructuredPrompt,
)
from tokenizer.config import load_config
from tokenizer.bundle import UniversalTokenizer


def parse_chat_format(text: str) -> StructuredPrompt:
    """Parse common chat formats into structured prompt.

    Recognizes formats like:
    - <|system|>...<|user|>...
    - [INST]...[/INST]
    - ### System:...\n### User:...

    Falls back to single USER segment if no format detected.
    """
    segments: list[PromptSegment] = []

    # Try to detect format
    if "<|system|>" in text or "<|user|>" in text:
        # OpenAI/Anthropic style
        parts = text.split("<|")
        for part in parts:
            if not part.strip():
                continue
            if part.startswith("system|>"):
                content = part[8:].strip()
                if content:
                    segments.append(PromptSegment(role=Role.SYSTEM, content=content.encode()))
            elif part.startswith("user|>"):
                content = part[6:].strip()
                if content:
                    segments.append(PromptSegment(role=Role.USER, content=content.encode()))
            elif part.startswith("assistant|>"):
                content = part[11:].strip()
                if content:
                    segments.append(PromptSegment(role=Role.ASSISTANT, content=content.encode()))
            elif part.startswith("developer|>"):
                content = part[11:].strip()
                if content:
                    segments.append(PromptSegment(role=Role.DEVELOPER, content=content.encode()))

    elif "### System:" in text or "### User:" in text:
        # Markdown-style headers
        lines = text.split("\n")
        current_role = None
        current_content: list[str] = []

        for line in lines:
            if line.startswith("### System:"):
                if current_role and current_content:
                    segments.append(PromptSegment(
                        role=current_role,
                        content="\n".join(current_content).strip().encode()
                    ))
                current_role = Role.SYSTEM
                current_content = [line[11:].strip()]
            elif line.startswith("### User:"):
                if current_role and current_content:
                    segments.append(PromptSegment(
                        role=current_role,
                        content="\n".join(current_content).strip().encode()
                    ))
                current_role = Role.USER
                current_content = [line[9:].strip()]
            elif line.startswith("### Assistant:"):
                if current_role and current_content:
                    segments.append(PromptSegment(
                        role=current_role,
                        content="\n".join(current_content).strip().encode()
                    ))
                current_role = Role.ASSISTANT
                current_content = [line[14:].strip()]
            else:
                current_content.append(line)

        if current_role and current_content:
            segments.append(PromptSegment(
                role=current_role,
                content="\n".join(current_content).strip().encode()
            ))

    # Fallback: treat as single user message
    if not segments:
        segments.append(PromptSegment(role=Role.USER, content=text.encode()))

    return StructuredPrompt(segments=segments)


def main():
    parser = argparse.ArgumentParser(
        description="Apply LIL transformation then encode with ULT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode text from stdin
  echo "Hello world" | lil_encode.py --config configs/default.toml

  # Encode a file
  lil_encode.py --input prompt.txt --config configs/default.toml --output encoded.bin

  # Encode with structured roles
  lil_encode.py --input chat.txt --config configs/default.toml --parse-chat
        """,
    )
    parser.add_argument("--input", "-i", type=Path, help="Input file (stdin if not specified)")
    parser.add_argument("--output", "-o", type=Path, help="Output file for encoded data")
    parser.add_argument("--config", "-c", type=Path, default=Path("configs/default.toml"),
                        help="Tokenizer config (default: configs/default.toml)")
    parser.add_argument("--parse-chat", action="store_true",
                        help="Parse input as chat format (detect roles)")
    parser.add_argument("--no-macros", action="store_true",
                        help="Disable macro compaction")
    parser.add_argument("--json-output", action="store_true",
                        help="Output JSON with metadata instead of raw bytes")
    parser.add_argument("--verify", action="store_true",
                        help="Verify round-trip reversibility")

    args = parser.parse_args()

    # Read input
    if args.input:
        text = args.input.read_text()
    else:
        text = sys.stdin.read()

    print(f"Input: {len(text)} bytes")

    # Parse into structured prompt
    if args.parse_chat:
        prompt = parse_chat_format(text)
        print(f"Parsed {len(prompt.segments)} segments:")
        for seg in prompt.segments:
            print(f"  {seg.role.value}: {len(seg.content)} bytes")
    else:
        prompt = StructuredPrompt(segments=[
            PromptSegment(role=Role.USER, content=text.encode())
        ])

    # Apply LIL transformation
    lil = LanguageInterfaceLayer(enable_macros=not args.no_macros)
    lil_result = lil.pack(prompt)

    print(f"\nLIL transformation:")
    print(f"  Original: {lil_result.original_size} bytes")
    print(f"  Packed: {lil_result.packed_size} bytes")
    print(f"  Ratio: {lil_result.size_ratio:.2%}")
    for log in lil_result.logs:
        print(f"  - {log.operation}: {log.input_bytes} -> {log.output_bytes}")
        if log.macros_applied:
            print(f"    Macros: {log.macros_applied}")

    # Verify LIL reversibility
    unpacked = lil.unpack(lil_result.data)
    lil_lossless = unpacked.to_text() == prompt.to_text()
    print(f"  LIL lossless: {lil_lossless}")

    if not lil_lossless:
        print("ERROR: LIL transformation is not lossless!")
        sys.exit(1)

    # Encode with ULT
    if args.config.exists():
        config = load_config(args.config)
        tokenizer = UniversalTokenizer(config)

        encode_result = tokenizer.encode(lil_result.data)
        print(f"\nULT encoding:")
        print(f"  Tokens: {len(encode_result.tokens)}")
        print(f"  Residual bytes: {len(encode_result.residual_data)}")

        # Verify ULT reversibility
        if args.verify:
            decode_result = tokenizer.decode(encode_result)
            ult_lossless = decode_result.data == lil_result.data
            print(f"  ULT lossless: {ult_lossless}")

            if ult_lossless:
                # Full round-trip
                final_prompt = lil.unpack(decode_result.data)
                full_lossless = final_prompt.to_text() == prompt.to_text()
                print(f"  Full round-trip lossless: {full_lossless}")
    else:
        print(f"\nSkipping ULT encoding (config not found: {args.config})")
        encode_result = None

    # Output
    if args.output:
        if args.json_output:
            output_data = {
                "input_bytes": len(text),
                "lil_packed_bytes": len(lil_result.data),
                "lil_ratio": lil_result.size_ratio,
                "num_segments": len(prompt.segments),
                "segments": [
                    {"role": s.role.value, "bytes": len(s.content)}
                    for s in prompt.segments
                ],
                "lil_lossless": lil_lossless,
                "packed_hex": lil_result.data.hex(),
            }
            if encode_result:
                output_data["tokens"] = encode_result.tokens
                output_data["num_tokens"] = len(encode_result.tokens)
            args.output.write_text(json.dumps(output_data, indent=2))
        else:
            args.output.write_bytes(lil_result.data)
        print(f"\nOutput written to {args.output}")


if __name__ == "__main__":
    main()
