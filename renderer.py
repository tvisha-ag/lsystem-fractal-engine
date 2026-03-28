"""
renderer.py — OpenGL Rendering Engine
======================================
Handles all GPU-side work:
  - OpenGL context setup via Pygame
  - GLSL shader program compilation and linking
  - VBO (Vertex Buffer Object) upload and draw calls
  - Pan/zoom transform management
  - Background scanline grid (cyberpunk floor effect)

Architecture: one VBO per draw call, rebuilt only when the edge list
changes. Per-frame cost is a single glDrawArrays call — scales to
hundreds of thousands of edges at 60 fps on any modern GPU.

OpenGL pipeline used:
  CPU (Python) → VBO → vertex shader → rasterisation → fragment shader → screen

Attribute layout inside the VBO (one record per vertex = one endpoint):
  [x: f32][y: f32][color_t: f32][depth: f32]   → stride = 4 × 4 = 16 bytes

Each LINE_STRIP pair contributes 2 vertices, so an edge list of N edges
generates 2N vertices uploaded once per graph rebuild.
"""

from __future__ import annotations
import os
import ctypes
import math
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False
    print("[Renderer] WARNING: pygame not installed. Using fallback mode.")

try:
    from OpenGL.GL import (
        glClear, glClearColor, glEnable, glBlendFunc,
        glLineWidth, glViewport, glUseProgram, glGetUniformLocation,
        glUniform2f, glUniform1f, glUniform1i, glDrawArrays,
        glGenVertexArrays, glBindVertexArray,
        glGenBuffers, glBindBuffer, glBufferData,
        glEnableVertexAttribArray, glVertexAttribPointer,
        glDeleteBuffers, glCreateShader, glShaderSource,
        glCompileShader, glGetShaderiv, glGetShaderInfoLog,
        glCreateProgram, glAttachShader, glLinkProgram,
        glGetProgramiv, glGetProgramInfoLog,
        GL_COLOR_BUFFER_BIT, GL_BLEND, GL_SRC_ALPHA,
        GL_ONE,  # Additive blending = neon glow
        GL_LINES, GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW,
        GL_FLOAT, GL_FALSE, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER,
        GL_COMPILE_STATUS, GL_LINK_STATUS, GL_LINE_SMOOTH,
        GL_LINE_SMOOTH_HINT, GL_NICEST,
    )
    from OpenGL.GL import glHint
    OPENGL_OK = True
except ImportError:
    OPENGL_OK = False
    print("[Renderer] WARNING: PyOpenGL not installed.")

from graph import LSystemGraph, Edge


WINDOW_W = 1280
WINDOW_H = 800
TITLE    = "L-System Fractal Growth Engine  |  Cyberpunk Shader Rendering"


class ShaderProgram:
    """
    Compiles, links, and provides uniform access for a GLSL program.
    """

    def __init__(self, vert_src: str, frag_src: str):
        self.program_id = self._build(vert_src, frag_src)
        self._uniform_cache: dict[str, int] = {}

    def use(self) -> None:
        glUseProgram(self.program_id)

    def set_float(self, name: str, v: float) -> None:
        glUniform1f(self._loc(name), v)

    def set_int(self, name: str, v: int) -> None:
        glUniform1i(self._loc(name), v)

    def set_vec2(self, name: str, x: float, y: float) -> None:
        glUniform2f(self._loc(name), x, y)

    def _loc(self, name: str) -> int:
        if name not in self._uniform_cache:
            self._uniform_cache[name] = glGetUniformLocation(self.program_id, name)
        return self._uniform_cache[name]

    @staticmethod
    def _compile_shader(source: str, shader_type: int) -> int:
        shader = glCreateShader(shader_type)
        glShaderSource(shader, source)
        glCompileShader(shader)
        status = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not status:
            log = glGetShaderInfoLog(shader).decode()
            raise RuntimeError(f"Shader compile error:\n{log}")
        return shader

    def _build(self, vert_src: str, frag_src: str) -> int:
        vert = self._compile_shader(vert_src, GL_VERTEX_SHADER)
        frag = self._compile_shader(frag_src, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vert)
        glAttachShader(prog, frag)
        glLinkProgram(prog)
        status = glGetProgramiv(prog, GL_LINK_STATUS)
        if not status:
            log = glGetProgramInfoLog(prog).decode()
            raise RuntimeError(f"Shader link error:\n{log}")
        return prog


class FractalVBO:
    """
    GPU-side vertex buffer for fractal geometry.
    Holds (x, y, color_t, depth) per vertex.
    Two vertices per edge → packed as a GL_LINES primitive.
    """

    STRIDE   = 4 * 4          # 4 floats × 4 bytes
    FLOATS_PER_VERTEX = 4

    def __init__(self):
        self.vbo_id      = glGenBuffers(1)
        self.vertex_count = 0

    def upload(self, graph: LSystemGraph, edges: List[Edge]) -> None:
        """
        Pack edge geometry into a flat float32 array and upload to GPU.
        Called once per graph rebuild — NOT every frame.
        """
        if not edges:
            self.vertex_count = 0
            return

        # Pre-allocate: 2 vertices × 4 floats each × N edges
        data = np.zeros((len(edges) * 2, self.FLOATS_PER_VERTEX), dtype=np.float32)

        row = 0
        for edge in edges:
            src = graph.nodes.get(edge.source_id)
            tgt = graph.nodes.get(edge.target_id)
            if src is None or tgt is None:
                row += 2
                continue

            t     = edge.color_t
            depth = float(edge.depth)

            data[row]     = [src.x, src.y, t, depth]
            data[row + 1] = [tgt.x, tgt.y, t, depth]
            row += 2

        self.vertex_count = row

        flat = data[:row].flatten()
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def draw(self, shader: ShaderProgram, first: int, count: int) -> None:
        """
        Draw [first, first+count) vertices as GL_LINES.
        `count` is in *vertices* (2 per edge).
        """
        if count <= 0:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)

        # Layout: [x f32][y f32][color_t f32][depth f32]
        stride = self.STRIDE

        # a_position (attrib 0)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))

        # a_color_t (attrib 1)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))

        # a_depth (attrib 2)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))

        # Clamp to what we actually uploaded
        safe_count = min(count, self.vertex_count - first)
        if safe_count > 0:
            glDrawArrays(GL_LINES, first, safe_count)

        glBindBuffer(GL_ARRAY_BUFFER, 0)


class CameraTransform:
    """
    Manages pan + zoom state for the fractal viewport.
    Converts between world space and screen space.
    """

    def __init__(self, width: int, height: int):
        self.width   = width
        self.height  = height
        self.offset  = [width / 2.0, height * 0.8]   # Start looking at tree base
        self.scale   = 1.0
        self.pan_speed = 15.0
        self.zoom_step = 0.1

    def zoom_in(self)  -> None: self.scale = min(20.0, self.scale * (1 + self.zoom_step))
    def zoom_out(self) -> None: self.scale = max(0.02, self.scale / (1 + self.zoom_step))
    def pan(self, dx: float, dy: float) -> None:
        self.offset[0] += dx / self.scale
        self.offset[1] += dy / self.scale
    def reset(self, width: int, height: int) -> None:
        self.offset = [width / 2.0, height * 0.8]
        self.scale  = 1.0

    def fit_to_graph(self, graph: LSystemGraph) -> None:
        """Auto-zoom so the fractal fills ~80% of the viewport."""
        if not graph.nodes:
            return
        min_x, min_y, max_x, max_y = graph.bounds
        w = max(max_x - min_x, 1.0)
        h = max(max_y - min_y, 1.0)
        scale_x = (self.width  * 0.8) / w
        scale_y = (self.height * 0.8) / h
        self.scale = min(scale_x, scale_y)
        cx = (min_x + max_x) / 2.0
        cy = (min_y + max_y) / 2.0
        self.offset[0] = self.width  / 2.0 / self.scale - cx
        self.offset[1] = self.height / 2.0 / self.scale - cy


class FractalRenderer:
    """
    Top-level renderer — owns the OpenGL context and render loop.

    Usage (called by main.py):
        renderer = FractalRenderer()
        renderer.init()
        renderer.load_graph(graph)
        while running:
            renderer.render(visible_edges, time_sec)
        renderer.shutdown()
    """

    def __init__(self, width: int = WINDOW_W, height: int = WINDOW_H):
        self.width    = width
        self.height   = height
        self._shader: Optional[ShaderProgram] = None
        self._vbo:    Optional[FractalVBO]    = None
        self._camera  = CameraTransform(width, height)
        self._graph:  Optional[LSystemGraph]  = None

        # Rendering parameters (user-adjustable via controls)
        self.glow_intensity     = 1.2
        self.chromatic_strength = 0.012
        self.color_scheme       = 0        # 0=cyberpunk, 1=fire, 2=ice, 3=matrix
        self.line_width         = 1.5
        self._start_time        = time.perf_counter()

        # VBO cursor: how many vertices to draw this frame
        self._draw_vertex_count = 0
        self._total_vertex_count = 0

    def init(self) -> None:
        """
        Initialise Pygame window and OpenGL context.
        Load and compile shaders.
        """
        if not PYGAME_OK or not OPENGL_OK:
            raise RuntimeError(
                "pygame and PyOpenGL must be installed.\n"
                "Run: pip install pygame PyOpenGL PyOpenGL_accelerate"
            )

        pygame.init()
        pygame.display.set_mode((self.width, self.height),
                                  DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption(TITLE)

        # OpenGL state
        glClearColor(0.0, 0.0, 0.05, 1.0)   # Near-black dark blue background
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)     # Additive blending = neon glow
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)

        # Load shaders from disk
        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        vert_path  = os.path.join(shader_dir, "vertex.glsl")
        frag_path  = os.path.join(shader_dir, "fragment.glsl")

        with open(vert_path)  as f: vert_src = f.read()
        with open(frag_path)  as f: frag_src = f.read()

        self._shader = ShaderProgram(vert_src, frag_src)
        self._vbo    = FractalVBO()

        print("[Renderer] OpenGL context initialised.")
        print(f"[Renderer] Window: {self.width}×{self.height}")

    def load_graph(self, graph: LSystemGraph) -> None:
        """
        Upload a new graph to the GPU.
        Uploads ALL edges once. Animated reveal is done by limiting draw count.
        """
        self._graph = graph
        edges_sorted = graph.get_edges_sorted_by_depth()
        self._vbo.upload(graph, edges_sorted)
        self._total_vertex_count = len(edges_sorted) * 2
        self._draw_vertex_count  = 0
        self._camera.fit_to_graph(graph)
        print(f"[Renderer] Loaded graph: {graph.node_count()} nodes, "
              f"{graph.edge_count()} edges → {self._total_vertex_count} vertices on GPU")

    def set_visible_count(self, edge_count: int) -> None:
        """Set how many edges to reveal (from growth engine)."""
        self._draw_vertex_count = min(edge_count * 2, self._total_vertex_count)

    def render(self) -> None:
        """
        Execute one full render pass.
        Called once per frame from the main loop.
        """
        t = time.perf_counter() - self._start_time

        glClear(GL_COLOR_BUFFER_BIT)
        glLineWidth(self.line_width)
        glViewport(0, 0, self.width, self.height)

        if self._shader and self._vbo and self._graph:
            self._shader.use()

            # Upload frame-level uniforms
            self._shader.set_vec2("u_resolution", float(self.width), float(self.height))
            self._shader.set_vec2("u_offset",
                                   self._camera.offset[0], self._camera.offset[1])
            self._shader.set_float("u_scale",              self._camera.scale)
            self._shader.set_float("u_time",               t)
            self._shader.set_float("u_max_depth",          float(max(self._graph.max_depth, 1)))
            self._shader.set_float("u_glow_intensity",     self.glow_intensity)
            self._shader.set_float("u_chromatic_strength", self.chromatic_strength)
            self._shader.set_int("u_color_scheme",         self.color_scheme)

            # Draw only the currently-visible portion
            self._vbo.draw(self._shader, 0, self._draw_vertex_count)

        pygame.display.flip()

    def handle_resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height
        glViewport(0, 0, width, height)
        self._camera.width  = width
        self._camera.height = height

    def camera(self) -> CameraTransform:
        return self._camera

    def cycle_color_scheme(self) -> int:
        self.color_scheme = (self.color_scheme + 1) % 4
        names = ["Cyberpunk", "Fire", "Ice", "Matrix"]
        print(f"[Renderer] Color scheme: {names[self.color_scheme]}")
        return self.color_scheme

    def adjust_glow(self, delta: float) -> None:
        self.glow_intensity = max(0.0, min(3.0, self.glow_intensity + delta))

    def adjust_chromatic(self, delta: float) -> None:
        self.chromatic_strength = max(0.0, min(0.05, self.chromatic_strength + delta))

    def shutdown(self) -> None:
        if PYGAME_OK:
            pygame.quit()
