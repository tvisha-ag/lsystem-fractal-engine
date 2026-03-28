"""
exporter.py — PNG and SVG Export Module
========================================
Saves the current fractal state in two formats:

  PNG Export:
    Reads the OpenGL framebuffer directly via glReadPixels and saves
    with Pillow (PIL). Captures exactly what is on screen including
    shaders, glow effects, and the HUD (if visible).

  SVG Export:
    Reconstructs the fractal geometry as a pure mathematical SVG,
    independent of screen resolution. The SVG:
      - Uses viewport scaling so the fractal fills the document
      - Encodes depth as stroke color (same palette as shader)
      - Adds metadata comments with L-System parameters
      - Is valid SVG 1.1 compatible with Inkscape/Illustrator/browsers
      - Includes a neon-style dark background + glow filter

    SVG colour mapping mirrors the GLSL fragment shader palette
    so the exported file looks visually consistent with the live render.

Export filenames include a timestamp for versioned output:
    exports/fractal_20240315_143022.png
    exports/fractal_20240315_143022.svg
"""

from __future__ import annotations
import os
import time
import math
from datetime import datetime
from typing import List, Optional, Tuple

from graph import LSystemGraph, Edge

EXPORTS_DIR = "exports"


def _ensure_exports_dir() -> str:
    """Create exports/ directory if it doesn't exist."""
    path = os.path.join(os.path.dirname(__file__), EXPORTS_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ── Colour palette (mirrors GLSL shaders) ────────────────────────────────────

def _cyberpunk_color(t: float) -> Tuple[int, int, int]:
    """
    Python re-implementation of the GLSL cyberpunk palette.
    t ∈ [0, 1] → (R, G, B) in 0-255 range.
    """
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        # Magenta → Cyan
        s = t / 0.5
        r = int(255 * (1.0 - s))
        g = int(255 * s)
        b = 255
    else:
        # Cyan → Yellow
        s = (t - 0.5) / 0.5
        r = int(255 * s)
        g = 255
        b = int(255 * (1.0 - s))
    return (r, g, b)


def _fire_color(t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t / 0.5
        return (int(200 * (0.8 + 0.2 * s)), int(128 * s), 0)
    else:
        s = (t - 0.5) / 0.5
        return (255, int(128 + 127 * s), int(77 * s))


def _ice_color(t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t / 0.5
        return (int(0 + 0 * s), int(26 + 204 * s), int(153 + 102 * s))
    else:
        s = (t - 0.5) / 0.5
        return (int(230 * s), 255, 255)


def _matrix_color(t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    if t < 0.5:
        s = t / 0.5
        return (0, int(128 * s), int(26 * s))
    else:
        s = (t - 0.5) / 0.5
        return (int(51 * s), int(128 + 127 * s), int(26 + 50 * s))


PALETTES = {
    0: _cyberpunk_color,
    1: _fire_color,
    2: _ice_color,
    3: _matrix_color,
}


def _to_hex(rgb: Tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


# ── PNG export ────────────────────────────────────────────────────────────────

def export_png(width: int, height: int,
               filename: Optional[str] = None) -> str:
    """
    Read the current OpenGL framebuffer and save as PNG.

    Args:
        width:    Framebuffer width in pixels
        height:   Framebuffer height in pixels
        filename: Override output path (auto-generated if None)

    Returns:
        Path to saved file.
    """
    try:
        from OpenGL.GL import glReadPixels, GL_RGB, GL_UNSIGNED_BYTE
        import pygame
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"[Exporter] PNG export requires PyOpenGL + Pillow: {e}")
        return ""

    # Read raw RGBA bytes from GPU
    raw = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

    # OpenGL origin is bottom-left → flip vertically for PIL
    arr = arr[::-1, :, :]

    img = Image.fromarray(arr, mode="RGB")

    exports_dir = _ensure_exports_dir()
    if filename is None:
        filename = os.path.join(exports_dir, f"fractal_{_timestamp()}.png")

    img.save(filename, "PNG")
    print(f"[Exporter] PNG saved: {filename} ({width}×{height})")
    return filename


# ── SVG export ────────────────────────────────────────────────────────────────

def export_svg(graph: LSystemGraph,
               visible_edges: Optional[List[Edge]] = None,
               color_scheme: int = 0,
               filename: Optional[str] = None,
               meta: Optional[dict] = None,
               svg_width: int = 1200,
               svg_height: int = 900) -> str:
    """
    Export the fractal as a scalable SVG with neon styling.

    Args:
        graph:         The L-System graph to export
        visible_edges: If provided, only export these edges (partial growth)
                       If None, export all edges.
        color_scheme:  0=cyberpunk, 1=fire, 2=ice, 3=matrix
        filename:      Output path (auto-generated if None)
        meta:          Dict with L-System metadata for SVG comments
        svg_width:     SVG document width in px
        svg_height:    SVG document height in px

    Returns:
        Path to saved file.
    """
    if not graph.nodes:
        print("[Exporter] Graph is empty — nothing to export.")
        return ""

    palette_fn = PALETTES.get(color_scheme, _cyberpunk_color)
    edges_to_draw = visible_edges if visible_edges is not None else graph.edges

    if not edges_to_draw:
        print("[Exporter] No edges to export.")
        return ""

    # ── Compute coordinate transform: world → SVG space ──────────────────────
    min_x, min_y, max_x, max_y = graph.bounds
    world_w = max(max_x - min_x, 1.0)
    world_h = max(max_y - min_y, 1.0)

    margin = 60
    usable_w = svg_width  - 2 * margin
    usable_h = svg_height - 2 * margin

    scale = min(usable_w / world_w, usable_h / world_h)

    def to_svg(wx: float, wy: float) -> Tuple[float, float]:
        """World coordinates → SVG document coordinates."""
        sx = margin + (wx - min_x) * scale
        # Flip Y: SVG origin is top-left, our turtle origin is bottom-left
        sy = svg_height - margin - (wy - min_y) * scale
        return (sx, sy)

    # ── Compute stroke widths ─────────────────────────────────────────────────
    # Root edges are thicker; tips are thinner (organic taper)
    max_d = max(graph.max_depth, 1)

    def stroke_width(depth: int) -> float:
        taper = 1.0 - (depth / max_d) * 0.75
        return max(0.4, 2.5 * taper)

    # ── Build SVG string ──────────────────────────────────────────────────────
    parts = []

    # Header + metadata
    meta = meta or {}
    parts.append(f"""<?xml version="1.0" encoding="UTF-8"?>
<!--
  L-System Fractal Growth Engine — SVG Export
  Generated: {datetime.now().isoformat()}
  Preset    : {meta.get('preset_name', 'unknown')}
  Iterations: {meta.get('iterations', '?')}
  Angle     : {meta.get('angle', '?')}°
  Nodes     : {graph.node_count()}
  Edges     : {len(edges_to_draw)}
-->
<svg xmlns="http://www.w3.org/2000/svg"
     width="{svg_width}" height="{svg_height}"
     viewBox="0 0 {svg_width} {svg_height}"
     version="1.1">
  <title>L-System Fractal — {meta.get('preset_name', 'fractal')}</title>
""")

    # Dark background
    parts.append(f'  <rect width="{svg_width}" height="{svg_height}" fill="#000010"/>\n')

    # Glow filter definition (simulates neon bloom in SVG)
    parts.append("""  <defs>
    <filter id="neon-glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    <filter id="strong-glow" x="-100%" y="-100%" width="300%" height="300%">
      <feGaussianBlur stdDeviation="5" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
""")

    # Group all edges
    parts.append('  <g id="fractal" filter="url(#neon-glow)">\n')

    # Sort edges depth-first so deep branches render on top
    edges_sorted = sorted(edges_to_draw, key=lambda e: e.depth)

    for edge in edges_sorted:
        src = graph.nodes.get(edge.source_id)
        tgt = graph.nodes.get(edge.target_id)
        if src is None or tgt is None:
            continue

        x1, y1 = to_svg(src.x, src.y)
        x2, y2 = to_svg(tgt.x, tgt.y)

        rgb = palette_fn(edge.color_t)
        hex_color = _to_hex(rgb)
        opacity = 0.7 + 0.3 * (1.0 - edge.color_t)  # Root edges more opaque
        sw = stroke_width(edge.depth)

        parts.append(
            f'    <line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{hex_color}" stroke-width="{sw:.2f}" '
            f'stroke-opacity="{opacity:.2f}" stroke-linecap="round"/>\n'
        )

    parts.append('  </g>\n')

    # Watermark / credit text
    parts.append(
        f'  <text x="{svg_width - 10}" y="{svg_height - 10}" '
        f'font-family="monospace" font-size="10" fill="#334" '
        f'text-anchor="end">L-System Fractal Engine</text>\n'
    )

    parts.append("</svg>\n")

    svg_string = "".join(parts)

    # Write to disk
    exports_dir = _ensure_exports_dir()
    if filename is None:
        filename = os.path.join(exports_dir, f"fractal_{_timestamp()}.svg")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_string)

    size_kb = len(svg_string) / 1024
    print(f"[Exporter] SVG saved: {filename} ({size_kb:.1f} KB, "
          f"{len(edges_to_draw)} edges)")
    return filename
