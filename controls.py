"""
controls.py — Keyboard/Mouse Input Handler and HUD Overlay
==========================================================
Processes Pygame events and maps them to engine actions.
Also renders the on-screen HUD (heads-up display) with:
  - Current preset name and iteration count
  - Growth progress bar
  - FPS counter
  - Active keybindings

Design principle: this module never touches OpenGL directly.
It uses Pygame's 2D surface overlay rendered AFTER the OpenGL frame,
drawn onto a temporary surface and blitted to screen.
This keeps the HUD pixel-perfect regardless of camera zoom.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Tuple
from enum import Enum, auto

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False


class Action(Enum):
    """All possible actions the user can trigger."""
    # Growth control
    TOGGLE_PAUSE     = auto()
    RESTART          = auto()
    REVEAL_ALL       = auto()
    SPEED_UP         = auto()
    SPEED_DOWN       = auto()
    MODE_DFS         = auto()
    MODE_BFS         = auto()
    MODE_RANDOM      = auto()

    # L-System parameters
    ITER_UP          = auto()
    ITER_DOWN        = auto()
    ANGLE_UP         = auto()
    ANGLE_DOWN       = auto()
    NEXT_PRESET      = auto()
    PREV_PRESET      = auto()

    # Camera
    ZOOM_IN          = auto()
    ZOOM_OUT         = auto()
    RESET_CAMERA     = auto()
    FIT_CAMERA       = auto()

    # Renderer
    CYCLE_COLOR      = auto()
    GLOW_UP          = auto()
    GLOW_DOWN        = auto()
    CHROMATIC_UP     = auto()
    CHROMATIC_DOWN   = auto()
    TOGGLE_HUD       = auto()

    # Export
    EXPORT_PNG       = auto()
    EXPORT_SVG       = auto()

    # System
    QUIT             = auto()


# ── Default keybindings ────────────────────────────────────────────────────────

DEFAULT_BINDINGS: Dict[int, Action] = {
    # Growth
    pygame.K_SPACE   if PYGAME_OK else 32:  Action.TOGGLE_PAUSE,
    pygame.K_r       if PYGAME_OK else 114: Action.RESTART,
    pygame.K_END     if PYGAME_OK else 279: Action.REVEAL_ALL,
    pygame.K_UP      if PYGAME_OK else 273: Action.SPEED_UP,
    pygame.K_DOWN    if PYGAME_OK else 274: Action.SPEED_DOWN,
    pygame.K_1       if PYGAME_OK else 49:  Action.MODE_DFS,
    pygame.K_2       if PYGAME_OK else 50:  Action.MODE_BFS,
    pygame.K_3       if PYGAME_OK else 51:  Action.MODE_RANDOM,

    # L-System
    pygame.K_EQUALS  if PYGAME_OK else 61:  Action.ITER_UP,
    pygame.K_MINUS   if PYGAME_OK else 45:  Action.ITER_DOWN,
    pygame.K_PERIOD  if PYGAME_OK else 46:  Action.ANGLE_UP,
    pygame.K_COMMA   if PYGAME_OK else 44:  Action.ANGLE_DOWN,
    pygame.K_n       if PYGAME_OK else 110: Action.NEXT_PRESET,
    pygame.K_p       if PYGAME_OK else 112: Action.PREV_PRESET,

    # Camera
    pygame.K_z       if PYGAME_OK else 122: Action.ZOOM_IN,
    pygame.K_x       if PYGAME_OK else 120: Action.ZOOM_OUT,
    pygame.K_F5      if PYGAME_OK else 290: Action.RESET_CAMERA,
    pygame.K_f       if PYGAME_OK else 102: Action.FIT_CAMERA,

    # Renderer
    pygame.K_c       if PYGAME_OK else 99:  Action.CYCLE_COLOR,
    pygame.K_g       if PYGAME_OK else 103: Action.GLOW_UP,
    pygame.K_v       if PYGAME_OK else 118: Action.GLOW_DOWN,
    pygame.K_h       if PYGAME_OK else 104: Action.TOGGLE_HUD,

    # Export
    pygame.K_F1      if PYGAME_OK else 282: Action.EXPORT_PNG,
    pygame.K_F2      if PYGAME_OK else 283: Action.EXPORT_SVG,

    # System
    pygame.K_ESCAPE  if PYGAME_OK else 27:  Action.QUIT,
    pygame.K_q       if PYGAME_OK else 113: Action.QUIT,
}

HUD_LINES: List[Tuple[str, str]] = [
    ("SPACE",   "Pause / Resume"),
    ("R",       "Restart growth"),
    ("END",     "Reveal all"),
    ("↑ / ↓",  "Speed up / down"),
    ("1/2/3",   "DFS / BFS / Random"),
    ("= / -",   "More / fewer iterations"),
    (". / ,",   "Angle +5° / -5°"),
    ("N / P",   "Next / prev preset"),
    ("Z / X",   "Zoom in / out"),
    ("F",       "Fit to screen"),
    ("F5",      "Reset camera"),
    ("C",       "Cycle colour scheme"),
    ("G / V",   "Glow +/-"),
    ("H",       "Toggle this HUD"),
    ("F1",      "Save PNG"),
    ("F2",      "Save SVG"),
    ("Q / ESC", "Quit"),
]


class InputHandler:
    """
    Processes Pygame events and dispatches Actions to registered callbacks.

    Mouse support:
      - Left-drag:  pan the camera
      - Scroll up:  zoom in
      - Scroll down: zoom out
    """

    def __init__(self):
        self._bindings: Dict[int, Action] = dict(DEFAULT_BINDINGS)
        self._callbacks: Dict[Action, List[Callable]] = {a: [] for a in Action}
        self._mouse_drag = False
        self._last_mouse: Optional[Tuple[int, int]] = None

    def register(self, action: Action, callback: Callable) -> None:
        """Register a callback for an action. Multiple callbacks per action OK."""
        self._callbacks[action].append(callback)

    def process_events(self) -> bool:
        """
        Poll Pygame event queue and dispatch actions.
        Returns False if a QUIT event was received.
        """
        if not PYGAME_OK:
            return True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._dispatch(Action.QUIT)
                return False

            elif event.type == pygame.KEYDOWN:
                action = self._bindings.get(event.key)
                if action:
                    self._dispatch(action)
                    if action == Action.QUIT:
                        return False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._mouse_drag = True
                    self._last_mouse = event.pos
                elif event.button == 4:
                    self._dispatch(Action.ZOOM_IN)
                elif event.button == 5:
                    self._dispatch(Action.ZOOM_OUT)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._mouse_drag = False
                    self._last_mouse = None

            elif event.type == pygame.MOUSEMOTION:
                if self._mouse_drag and self._last_mouse:
                    dx = event.pos[0] - self._last_mouse[0]
                    dy = event.pos[1] - self._last_mouse[1]
                    # Dispatch as pan with attached delta data via a closure trick
                    for cb in self._callbacks.get(Action.RESET_CAMERA, []):
                        pass  # pan handled separately below
                    self._pan_delta = (dx, dy)
                    self._last_mouse = event.pos

            elif event.type == pygame.VIDEORESIZE:
                pass  # handled in main loop

        return True

    def get_pan_delta(self) -> Optional[Tuple[int, int]]:
        """Consume and return pending mouse pan delta (if any)."""
        delta = getattr(self, "_pan_delta", None)
        self._pan_delta = None
        return delta

    def _dispatch(self, action: Action) -> None:
        for cb in self._callbacks.get(action, []):
            try:
                cb()
            except Exception as e:
                print(f"[Controls] Callback error for {action}: {e}")


class HUDRenderer:
    """
    Renders an on-screen overlay with fractal stats and key bindings.
    Uses Pygame's 2D font rendering blitted over the OpenGL frame.

    Strategy: after glFlush, blit a transparent Pygame surface containing
    HUD text directly to the display surface. This avoids any OpenGL text
    complexity and keeps the HUD crisp at any resolution.
    """

    FONT_SIZE_LARGE = 18
    FONT_SIZE_SMALL = 13
    PADDING         = 14
    LINE_HEIGHT     = 20
    HUD_WIDTH       = 310
    COLOR_TITLE     = (0,   255, 255)   # Cyan
    COLOR_KEY       = (255, 200, 0  )   # Amber
    COLOR_VALUE     = (200, 200, 200)   # Light grey
    COLOR_PROGRESS  = (0,   255, 100)   # Green
    COLOR_DIM       = (100, 100, 100)   # Dimmed text
    COLOR_BG        = (0,   0,   0,  160)  # Semi-transparent black

    def __init__(self, screen_w: int, screen_h: int):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.visible  = True
        self._font_large = None
        self._font_small = None
        self._init_fonts()

    def _init_fonts(self) -> None:
        if not PYGAME_OK:
            return
        pygame.font.init()
        try:
            self._font_large = pygame.font.SysFont("Consolas",   self.FONT_SIZE_LARGE)
            self._font_small = pygame.font.SysFont("Consolas",   self.FONT_SIZE_SMALL)
        except Exception:
            self._font_large = pygame.font.Font(None, self.FONT_SIZE_LARGE + 4)
            self._font_small = pygame.font.Font(None, self.FONT_SIZE_SMALL + 4)

    def render(self, surface: "pygame.Surface",
               state: dict) -> None:
        """
        Draw HUD onto `surface`.

        state keys:
            preset_name, iterations, angle, growth_pct, speed,
            edge_count, node_count, fps, color_scheme, mode, paused
        """
        if not self.visible or not PYGAME_OK:
            return

        pad = self.PADDING
        lh  = self.LINE_HEIGHT
        x   = pad
        y   = pad

        def text(txt, color, font=None, bold=False):
            nonlocal y
            f = font or self._font_small
            if bold and self._font_large:
                f = self._font_large
            surf = f.render(txt, True, color)
            surface.blit(surf, (x, y))
            y += lh

        def spacer(n=1):
            nonlocal y
            y += lh * n // 2

        # ── Header ────────────────────────────────────────────────────────────
        text("L-SYSTEM FRACTAL ENGINE", self.COLOR_TITLE, bold=True)
        spacer()

        # ── Stats ─────────────────────────────────────────────────────────────
        preset = state.get("preset_name", "—").replace("_", " ").title()
        text(f"Preset    : {preset}",          self.COLOR_VALUE)
        text(f"Iterations: {state.get('iterations', 0)}", self.COLOR_VALUE)
        text(f"Angle     : {state.get('angle', 0):.1f}°", self.COLOR_VALUE)
        text(f"Nodes     : {state.get('node_count', 0):,}", self.COLOR_VALUE)
        text(f"Edges     : {state.get('edge_count', 0):,}", self.COLOR_VALUE)
        text(f"Mode      : {state.get('mode', 'dfs').upper()}", self.COLOR_VALUE)

        # Pause indicator
        paused_str = "  [PAUSED]" if state.get("paused") else ""
        text(f"Speed     : {state.get('speed', 0):.0f} e/s{paused_str}",
             self.COLOR_KEY if state.get("paused") else self.COLOR_VALUE)

        spacer()

        # ── Progress bar ──────────────────────────────────────────────────────
        pct = state.get("growth_pct", 0.0)
        bar_w = self.HUD_WIDTH - pad
        filled = int(bar_w * pct)
        text(f"Growth: {pct*100:.1f}%", self.COLOR_PROGRESS)
        bar_rect = pygame.Rect(x, y, bar_w, 8)
        pygame.draw.rect(surface, (40, 40, 40), bar_rect, border_radius=4)
        if filled > 0:
            fill_rect = pygame.Rect(x, y, filled, 8)
            pygame.draw.rect(surface, self.COLOR_PROGRESS, fill_rect, border_radius=4)
        y += 16

        spacer()

        # ── FPS ───────────────────────────────────────────────────────────────
        fps = state.get("fps", 0.0)
        fps_color = (0, 255, 0) if fps >= 55 else (255, 200, 0) if fps >= 30 else (255, 50, 50)
        text(f"FPS : {fps:.1f}", fps_color)

        color_names = ["Cyberpunk", "Fire", "Ice", "Matrix"]
        cs = state.get("color_scheme", 0)
        text(f"Color: {color_names[cs % 4]}", self.COLOR_VALUE)

        spacer()

        # ── Key bindings (compact) ─────────────────────────────────────────────
        text("─── Controls ───────────", self.COLOR_DIM)
        for key, desc in HUD_LINES[:8]:    # Show first 8 bindings
            line = f"{key:8s} {desc}"
            text(line, self.COLOR_DIM)
        text("H   Toggle full help", self.COLOR_DIM)

    def toggle(self) -> None:
        self.visible = not self.visible

    def resize(self, w: int, h: int) -> None:
        self.screen_w, self.screen_h = w, h
