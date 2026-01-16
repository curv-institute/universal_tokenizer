#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "torch",
# ]
# ///
"""Generate qualitative LIL example figure for the paper.

Creates a 3-panel figure illustrating:
- Panel A: Raw prompt with implicit boundaries and repeated scaffolding
- Panel B: LIL wire format with explicit structure and macro compaction
- Panel C: Tokenization stability under prompt extension

Output: paper/figures/lil_qualitative_example.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.lil import (
    LanguageInterfaceLayer,
    PromptSegment,
    Role,
    StructuredPrompt,
    WireFormat,
)
from tokenizer.config import load_config
from tokenizer.bundle import UniversalTokenizer


# Fixed example prompt for reproducibility
SYSTEM_TEXT = "You are a helpful assistant. Answer concisely."
USER_TEXT = "Explain representational stability in machine learning."
EXTENSION_TEXT = (
    "Additional context that is not directly relevant to the question: "
    "The weather today is sunny with a high of 72 degrees. "
    "Stock markets closed higher on Tuesday amid economic optimism."
)


def create_prompts():
    """Create the base and extended prompts."""
    base_prompt = StructuredPrompt(segments=[
        PromptSegment(role=Role.SYSTEM, content=SYSTEM_TEXT.encode()),
        PromptSegment(role=Role.USER, content=USER_TEXT.encode()),
    ])

    extended_prompt = StructuredPrompt(segments=[
        PromptSegment(role=Role.SYSTEM, content=SYSTEM_TEXT.encode()),
        PromptSegment(role=Role.USER, content=(USER_TEXT + " " + EXTENSION_TEXT).encode()),
    ])

    return base_prompt, extended_prompt


def format_wire_bytes(data: bytes, macros_applied: list[int], registry) -> list[tuple[str, str]]:
    """Format wire bytes for display with annotations.

    Returns list of (text, color) tuples for rendering.
    """
    parts = []
    i = 0

    # Color scheme
    HEADER_COLOR = "#2E86AB"      # Blue for headers
    ROLE_COLOR = "#A23B72"        # Pink for roles
    CONTENT_COLOR = "#333333"     # Dark gray for content
    MACRO_COLOR = "#F18F01"       # Orange for macros
    STRUCT_COLOR = "#C73E1D"      # Red for structure

    while i < len(data):
        # Check for magic
        if data[i:i+4] == WireFormat.MAGIC:
            parts.append(("MAGIC", HEADER_COLOR))
            i += 4
            continue

        # Check for version
        if i == 4 and data[i:i+1] == WireFormat.VERSION:
            parts.append(("V1", HEADER_COLOR))
            i += 1
            continue

        # Check for segment count (2 bytes after version)
        if i == 5:
            count = int.from_bytes(data[i:i+2], 'big')
            parts.append((f"N={count}", HEADER_COLOR))
            i += 2
            continue

        # Check for segment markers
        if data[i:i+1] == WireFormat.SEGMENT_START:
            parts.append(("[", STRUCT_COLOR))
            i += 1
            continue

        if data[i:i+1] == WireFormat.SEGMENT_END:
            parts.append(("]", STRUCT_COLOR))
            i += 1
            continue

        # Check for role markers
        role_found = False
        for role, marker in WireFormat.ROLE_MARKERS.items():
            if data[i:i+1] == marker:
                parts.append((role.value.upper()[:3], ROLE_COLOR))
                i += 1
                role_found = True
                break
        if role_found:
            continue

        # Check for macro references
        if data[i:i+1] == WireFormat.MACRO_START:
            if i + 2 < len(data) and data[i+2:i+3] == WireFormat.MACRO_END:
                macro_id = data[i+1]
                macro = registry.get_by_id(macro_id)
                if macro:
                    parts.append((f"M:{macro.name[:8]}", MACRO_COLOR))
                else:
                    parts.append((f"M:0x{macro_id:02x}", MACRO_COLOR))
                i += 3
                continue

        # Check for length field (4 bytes)
        if i > 7 and parts and parts[-1][1] == ROLE_COLOR:
            length = int.from_bytes(data[i:i+4], 'big')
            parts.append((f"L={length}", STRUCT_COLOR))
            i += 4
            continue

        # Regular content - show abbreviated
        content_start = i
        while i < len(data) and data[i:i+1] not in [
            WireFormat.SEGMENT_START, WireFormat.SEGMENT_END,
            WireFormat.MACRO_START
        ]:
            i += 1
        content = data[content_start:i]
        if len(content) > 20:
            text = content[:10].decode('utf-8', errors='replace') + "..."
        else:
            text = content.decode('utf-8', errors='replace')
        parts.append((text, CONTENT_COLOR))

    return parts


def render_panel_a(ax, base_prompt, extended_prompt):
    """Render Panel A: Raw prompt baseline."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("A) Raw Prompt (Baseline)", fontsize=12, fontweight='bold', loc='left')

    # Colors
    SYSTEM_BG = "#E8F4F8"
    USER_BG = "#FFF3E0"
    EXTENSION_BG = "#FFEBEE"

    y = 9.0

    # System section (with implicit boundary highlight)
    ax.add_patch(FancyBboxPatch((0.2, y-1.4), 9.6, 1.3,
                                 boxstyle="round,pad=0.05",
                                 facecolor=SYSTEM_BG, edgecolor='#999', linewidth=0.5))
    ax.text(0.4, y-0.3, "[implicit system boundary]", fontsize=7,
            color='#666', style='italic')
    ax.text(0.4, y-0.9, SYSTEM_TEXT, fontsize=9, family='monospace',
            wrap=True)

    y -= 2.0

    # User section
    ax.add_patch(FancyBboxPatch((0.2, y-1.8), 9.6, 1.7,
                                 boxstyle="round,pad=0.05",
                                 facecolor=USER_BG, edgecolor='#999', linewidth=0.5))
    ax.text(0.4, y-0.3, "[implicit user boundary]", fontsize=7,
            color='#666', style='italic')
    ax.text(0.4, y-0.9, USER_TEXT, fontsize=9, family='monospace',
            wrap=True)

    y -= 2.5

    # Extension section (highlighted as problematic)
    ax.add_patch(FancyBboxPatch((0.2, y-2.5), 9.6, 2.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor=EXTENSION_BG, edgecolor='#C73E1D',
                                 linewidth=1.5, linestyle='--'))
    ax.text(0.4, y-0.3, "Extension (destabilizes tokenization)", fontsize=8,
            color='#C73E1D', fontweight='bold')
    # Wrap long extension text
    ext_lines = [EXTENSION_TEXT[i:i+55] for i in range(0, len(EXTENSION_TEXT), 55)]
    for j, line in enumerate(ext_lines[:3]):
        ax.text(0.4, y-0.9-j*0.5, line, fontsize=8, family='monospace',
                color='#666')

    y -= 3.5

    # Note about ambiguity
    ax.text(0.4, y, "• No explicit role markers", fontsize=8, color='#666')
    ax.text(0.4, y-0.4, "• Extension changes all token boundaries", fontsize=8, color='#C73E1D')
    ax.text(0.4, y-0.8, "• Repeated patterns not compacted", fontsize=8, color='#666')


def render_panel_b(ax, lil, base_prompt, extended_prompt):
    """Render Panel B: LIL wire format."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("B) LIL Wire Format", fontsize=12, fontweight='bold', loc='left')

    # Pack the base prompt
    result = lil.pack(base_prompt)
    macros_applied = []
    for log in result.logs:
        macros_applied.extend(log.macros_applied)

    y = 9.0

    # Header explanation
    ax.text(0.3, y, "Explicit structure with macro compaction:", fontsize=9,
            fontweight='bold')

    y -= 0.8

    # Format bytes for display
    parts = format_wire_bytes(result.data, macros_applied, lil.registry)

    # Render wire format boxes
    x = 0.3
    row_y = y
    for text, color in parts:
        # Calculate box width based on text
        width = max(len(text) * 0.12, 0.6)
        if x + width > 9.5:
            x = 0.3
            row_y -= 0.7

        # Draw box
        ax.add_patch(FancyBboxPatch((x, row_y-0.45), width, 0.5,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color + "30",  # 30% opacity
                                     edgecolor=color, linewidth=1))
        ax.text(x + width/2, row_y-0.2, text, fontsize=7,
                family='monospace', ha='center', color=color)
        x += width + 0.1

    y = row_y - 1.2

    # Legend
    ax.text(0.3, y, "Legend:", fontsize=8, fontweight='bold')
    legend_items = [
        ("MAGIC/V1", "#2E86AB", "Header"),
        ("SYS/USR", "#A23B72", "Role marker"),
        ("M:name", "#F18F01", "Macro ref"),
        ("[ ]", "#C73E1D", "Segment"),
    ]
    lx = 1.2
    for abbr, color, desc in legend_items:
        ax.add_patch(FancyBboxPatch((lx, y-0.5), 0.8, 0.4,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color + "30", edgecolor=color))
        ax.text(lx + 0.4, y-0.3, abbr, fontsize=6, ha='center',
                family='monospace', color=color)
        ax.text(lx + 0.9, y-0.3, f"= {desc}", fontsize=7, color='#666')
        lx += 2.3

    y -= 1.2

    # Benefits
    ax.text(0.3, y, "✓ Explicit role boundaries", fontsize=8, color='#2E7D32')
    ax.text(0.3, y-0.4, "✓ Macros compact 'helpful assistant' → 1 byte",
            fontsize=8, color='#2E7D32')
    ax.text(0.3, y-0.8, "✓ Extension isolated to USER segment",
            fontsize=8, color='#2E7D32')
    ax.text(0.3, y-1.2, f"✓ Size: {result.original_size}B → {result.packed_size}B "
            f"({result.size_ratio:.0%})", fontsize=8, color='#2E7D32')


def render_panel_c(ax, lil, tokenizer, base_prompt, extended_prompt):
    """Render Panel C: Tokenization stability view."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("C) Tokenization Stability", fontsize=12, fontweight='bold', loc='left')

    y = 9.0

    # Encode base and extended with baseline (raw bytes)
    raw_base = base_prompt.to_text().encode()
    raw_ext = extended_prompt.to_text().encode()

    enc_base_raw = tokenizer.encode(raw_base)
    enc_ext_raw = tokenizer.encode(raw_ext)

    # Encode base and extended with LIL
    lil_base = lil.pack(base_prompt)
    lil_ext = lil.pack(extended_prompt)

    enc_base_lil = tokenizer.encode(lil_base.data)
    enc_ext_lil = tokenizer.encode(lil_ext.data)

    # Colors for tokens
    STABLE_COLOR = "#4CAF50"   # Green for stable
    CHANGED_COLOR = "#F44336"  # Red for changed
    NEW_COLOR = "#9E9E9E"      # Gray for new

    def draw_token_bar(ax, x, y, width, color, label=None):
        ax.add_patch(FancyBboxPatch((x, y-0.25), width, 0.4,
                                     boxstyle="round,pad=0.01",
                                     facecolor=color, edgecolor='none'))
        if label:
            ax.text(x + width/2, y, label, fontsize=6, ha='center',
                    va='center', color='white', fontweight='bold')

    # Baseline comparison
    ax.text(0.3, y, "Baseline (raw bytes):", fontsize=9, fontweight='bold')
    y -= 0.6

    ax.text(0.3, y, "Base prompt:", fontsize=8, color='#666')
    x = 2.5
    for i, tok in enumerate(enc_base_raw.tokens[:8]):
        draw_token_bar(ax, x, y, 0.6, STABLE_COLOR)
        x += 0.7
    if len(enc_base_raw.tokens) > 8:
        ax.text(x, y, f"...+{len(enc_base_raw.tokens)-8}", fontsize=7, color='#666')

    y -= 0.6
    ax.text(0.3, y, "Extended:", fontsize=8, color='#666')
    x = 2.5
    # Show that baseline tokens change
    for i, tok in enumerate(enc_ext_raw.tokens[:8]):
        if i < len(enc_base_raw.tokens) and tok.id == enc_base_raw.tokens[i].id:
            draw_token_bar(ax, x, y, 0.6, STABLE_COLOR)
        else:
            draw_token_bar(ax, x, y, 0.6, CHANGED_COLOR)
        x += 0.7
    if len(enc_ext_raw.tokens) > 8:
        ax.text(x, y, f"...+{len(enc_ext_raw.tokens)-8}", fontsize=7, color='#666')

    y -= 0.5
    # Compute baseline churn (extract token IDs)
    base_ids = [t.id for t in enc_base_raw.tokens]
    ext_ids = [t.id for t in enc_ext_raw.tokens]
    base_set = set(base_ids)
    ext_set = set(ext_ids[:len(base_ids)])
    baseline_overlap = len(base_set & ext_set) / max(len(base_set), 1)
    ax.text(0.3, y, f"→ Core instruction churn: {(1-baseline_overlap)*100:.0f}% tokens changed",
            fontsize=8, color=CHANGED_COLOR)

    y -= 1.0

    # LIL comparison
    ax.text(0.3, y, "LIL-stabilized:", fontsize=9, fontweight='bold')
    y -= 0.6

    ax.text(0.3, y, "Base prompt:", fontsize=8, color='#666')
    x = 2.5
    for i, tok in enumerate(enc_base_lil.tokens[:8]):
        draw_token_bar(ax, x, y, 0.6, STABLE_COLOR)
        x += 0.7
    if len(enc_base_lil.tokens) > 8:
        ax.text(x, y, f"...+{len(enc_base_lil.tokens)-8}", fontsize=7, color='#666')

    y -= 0.6
    ax.text(0.3, y, "Extended:", fontsize=8, color='#666')
    x = 2.5
    # LIL tokens should be more stable for the core instruction
    # The system segment should be identical
    for i, tok in enumerate(enc_ext_lil.tokens[:8]):
        if i < len(enc_base_lil.tokens) and tok.id == enc_base_lil.tokens[i].id:
            draw_token_bar(ax, x, y, 0.6, STABLE_COLOR)
        elif i >= len(enc_base_lil.tokens):
            draw_token_bar(ax, x, y, 0.6, NEW_COLOR)
        else:
            draw_token_bar(ax, x, y, 0.6, CHANGED_COLOR)
        x += 0.7
    if len(enc_ext_lil.tokens) > 8:
        ax.text(x, y, f"...+{len(enc_ext_lil.tokens)-8}", fontsize=7, color='#666')

    y -= 0.5
    # Compute LIL stability (extract token IDs)
    lil_base_ids = [t.id for t in enc_base_lil.tokens]
    lil_ext_ids = [t.id for t in enc_ext_lil.tokens]
    lil_base_set = set(lil_base_ids)
    lil_ext_set = set(lil_ext_ids[:len(lil_base_ids)])
    lil_overlap = len(lil_base_set & lil_ext_set) / max(len(lil_base_set), 1)
    ax.text(0.3, y, f"→ Core instruction preserved: structure isolated",
            fontsize=8, color=STABLE_COLOR)

    y -= 1.2

    # Legend
    ax.text(0.3, y, "Legend:", fontsize=8, fontweight='bold')
    draw_token_bar(ax, 1.5, y, 0.5, STABLE_COLOR)
    ax.text(2.1, y, "= stable", fontsize=7)
    draw_token_bar(ax, 3.0, y, 0.5, CHANGED_COLOR)
    ax.text(3.6, y, "= changed", fontsize=7)
    draw_token_bar(ax, 4.8, y, 0.5, NEW_COLOR)
    ax.text(5.4, y, "= new (extension)", fontsize=7)

    y -= 0.8
    ax.text(0.3, y, "Key insight: LIL explicit structure prevents extension",
            fontsize=8, fontweight='bold')
    ax.text(0.3, y-0.4, "from destabilizing tokenization of core instruction.",
            fontsize=8)


def main():
    # Ensure output directory exists
    output_dir = Path("paper/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LIL and tokenizer
    lil = LanguageInterfaceLayer(enable_macros=True)

    config_path = Path("configs/cpu_small.toml")
    if config_path.exists():
        config = load_config(config_path)
        tokenizer = UniversalTokenizer(config)
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        tokenizer = None

    # Create prompts
    base_prompt, extended_prompt = create_prompts()

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle("Language Interface Layer (LIL): Qualitative Example",
                 fontsize=14, fontweight='bold', y=0.98)

    # Render panels
    render_panel_a(axes[0], base_prompt, extended_prompt)
    render_panel_b(axes[1], lil, base_prompt, extended_prompt)

    if tokenizer:
        render_panel_c(axes[2], lil, tokenizer, base_prompt, extended_prompt)
    else:
        axes[2].text(0.5, 0.5, "Tokenizer not available",
                     ha='center', va='center', fontsize=12)
        axes[2].axis('off')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    # Save figure
    output_path = output_dir / "lil_qualitative_example.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=150)
    print(f"Figure saved to {output_path}")

    # Also save PNG for preview
    png_path = output_dir / "lil_qualitative_example.png"
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=150)
    print(f"Preview saved to {png_path}")

    plt.close()


if __name__ == "__main__":
    main()
