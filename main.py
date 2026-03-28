"""
main.py — Generative L-System Fractal Growth Engine
====================================================
Entry point that wires together all modules:

  lsystem.py      → L-System string generation
  graph.py        → Graph / tree data structure construction
  growth_engine.py→ DFS/BFS animated growth controller
  optimizer.py    → Graph pruning and memory management
  renderer.py     → PyOpenGL + GLSL shader rendering
  controls.py     → Keyboard/mouse input + HUD overlay
  exporter.py     → PNG and SVG export

Runtime architecture (single-threaded):

  ┌─────────────────────────────────────────────────────────────┐
  │  main loop (60 fps target)                                   │
  │                                                              │
  │  1. InputHandler.process_events()   ← keyboard/mouse        │
  │  2. GrowthController.tick(dt)       ← advance animation     │
  │  3. renderer.set_visible_count()    ← update draw cursor    │
  │  4. renderer.render()               ← OpenGL draw call      │
  │  5. HUDRenderer.render()            ← Pygame 2D overlay     │
  │  6. pygame.display.flip()           ← swap buffers          │
  └─────────────────────────────────────────────────────────────┘

When L-System parameters change (key press triggers rebuild):
  LSystemGenerator.generate() → LSystemGraphBuilder.build()
  → GraphOptimizer.full_pipeline() → renderer.load_graph()
  → GrowthController.load(graph) → restart animation

This separation means the rebuild is only triggered on user action,
not every frame — keeping the main loop extremely lightweight.
"""

import sys
import os
import time

# Ensure project root on path so relative imports work from any directory
sys.path.insert(0, os.path.dirname(__file__))

try:
    import pygame
    from pygame.locals import DOUBLEBUF, OPENGL, RESIZABLE, VIDEORESIZE
except ImportError:
    print("ERROR: pygame is required.\n  pip install pygame")
    sys.exit(1)

try:
    from OpenGL.GL import glReadPixels
except ImportError:
    print("ERROR: PyOpenGL is required.\n  pip install PyOpenGL PyOpenGL_accelerate")
    sys.exit(1)

from lsystem      import LSystemGenerator, LSystemConfig, PRESETS
from graph        import LSystemGraphBuilder, LSystemGraph
from growth_engine import GrowthController
from optimizer    import GraphOptimizer
from renderer     import FractalRenderer
from controls     import InputHandler, HUDRenderer, Action
from exporter     import export_png, export_svg


# ─── Application state ─────────────────────────────────────────────────────────

class AppState:
    """Central mutable state bag shared across all modules."""

    def __init__(self):
        # L-System parameters
        self.preset_keys = list(PRESETS.keys())
        self.preset_index = 0
        self.iterations   = 5
        self.angle_deg    = PRESETS[self.preset_keys[0]]["angle"]

        # Runtime objects
        self.graph: LSystemGraph | None = None
        self.running = True

        # FPS tracking
        self._fps_samples = []
        self._last_time   = time.perf_counter()

    @property
    def current_preset_key(self) -> str:
        return self.preset_keys[self.preset_index]

    @property
    def current_preset(self) -> dict:
        return PRESETS[self.current_preset_key]

    def next_preset(self) -> None:
        self.preset_index = (self.preset_index + 1) % len(self.preset_keys)
        self.angle_deg = self.current_preset["angle"]

    def prev_preset(self) -> None:
        self.preset_index = (self.preset_index - 1) % len(self.preset_keys)
        self.angle_deg = self.current_preset["angle"]

    def update_fps(self) -> float:
        now = time.perf_counter()
        dt  = now - self._last_time
        self._last_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        self._fps_samples.append(fps)
        if len(self._fps_samples) > 60:
            self._fps_samples.pop(0)
        return sum(self._fps_samples) / len(self._fps_samples)


# ─── Core rebuild pipeline ────────────────────────────────────────────────────

def rebuild(state: AppState,
            generator: LSystemGenerator,
            builder: LSystemGraphBuilder,
            optimizer: GraphOptimizer,
            renderer: FractalRenderer,
            controller: GrowthController) -> None:
    """
    Regenerate the full pipeline from L-System string to GPU geometry.
    Called on startup and whenever parameters change.
    """
    preset = state.current_preset
    config = LSystemConfig(
        axiom=preset["axiom"],
        rules=preset["rules"],
        angle=state.angle_deg,
        iterations=state.iterations,
        preset_name=state.current_preset_key,
    )

    print(f"\n[Main] Rebuilding: {preset['name']} | "
          f"iter={state.iterations} | angle={state.angle_deg:.1f}°")

    # Step 1: Generate L-System string
    result = generator.generate(config)
    analysis = generator.analyze_string(result)
    print(f"[Main] String length: {result.symbol_count:,} | "
          f"Forward moves: {analysis['forward_moves']:,} | "
          f"Max branch depth: {analysis['max_branch_depth']} | "
          f"Generated in {result.generation_time_ms:.1f}ms")

    # Step 2: Build graph
    graph = builder.build(
        lsystem_string=result.string,
        step_length=6.0,
        angle_deg=state.angle_deg,
        start_x=0.0,
        start_y=0.0,
        start_angle_deg=90.0,
    )
    print(f"[Main] Graph: {graph.node_count():,} nodes, "
          f"{graph.edge_count():,} edges, max_depth={graph.max_depth}")

    # Step 3: Optimize
    optimizer.full_pipeline(
        graph,
        max_depth=None,           # No artificial depth cap
        min_edge_length=0.2,      # Remove sub-pixel segments
        compress_chains=True,
    )
    print(f"[Main] After optimization: {graph.node_count():,} nodes, "
          f"{graph.edge_count():,} edges")

    # Step 4: Upload to GPU
    renderer.load_graph(graph)

    # Step 5: Reset growth animation
    controller.load(graph)

    state.graph = graph


# ─── Main entry point ──────────────────────────────────────────────────────────

def main() -> None:
    # ── Init renderer (opens window) ──────────────────────────────────────────
    renderer   = FractalRenderer(width=1280, height=800)
    renderer.init()

    # ── Init subsystems ───────────────────────────────────────────────────────
    state      = AppState()
    generator  = LSystemGenerator()
    builder    = LSystemGraphBuilder(max_depth=50)
    optimizer  = GraphOptimizer()
    controller = GrowthController()
    inputs     = InputHandler()
    hud        = HUDRenderer(1280, 800)

    # ── Wire up input → action callbacks ─────────────────────────────────────

    def do_rebuild():
        rebuild(state, generator, builder, optimizer, renderer, controller)

    # Growth controls
    inputs.register(Action.TOGGLE_PAUSE,   controller.toggle_pause)
    inputs.register(Action.RESTART,        controller.restart)
    inputs.register(Action.RESTART,        lambda: None)   # also triggers rebuild? no, just restart
    inputs.register(Action.REVEAL_ALL,     controller.reveal_all)
    inputs.register(Action.SPEED_UP,       controller.speed_up)
    inputs.register(Action.SPEED_DOWN,     controller.speed_down)
    inputs.register(Action.MODE_DFS,       lambda: controller.set_mode("dfs"))
    inputs.register(Action.MODE_BFS,       lambda: controller.set_mode("bfs"))
    inputs.register(Action.MODE_RANDOM,    lambda: controller.set_mode("random"))

    # L-System parameter changes (trigger rebuild)
    def iter_up():
        state.iterations = min(state.iterations + 1, 9)
        do_rebuild()

    def iter_down():
        state.iterations = max(state.iterations - 1, 1)
        do_rebuild()

    def angle_up():
        state.angle_deg = min(state.angle_deg + 5.0, 170.0)
        do_rebuild()

    def angle_down():
        state.angle_deg = max(state.angle_deg - 5.0, 1.0)
        do_rebuild()

    def next_preset():
        state.next_preset()
        do_rebuild()

    def prev_preset():
        state.prev_preset()
        do_rebuild()

    inputs.register(Action.ITER_UP,    iter_up)
    inputs.register(Action.ITER_DOWN,  iter_down)
    inputs.register(Action.ANGLE_UP,   angle_up)
    inputs.register(Action.ANGLE_DOWN, angle_down)
    inputs.register(Action.NEXT_PRESET, next_preset)
    inputs.register(Action.PREV_PRESET, prev_preset)

    # Camera
    cam = renderer.camera()
    inputs.register(Action.ZOOM_IN,       cam.zoom_in)
    inputs.register(Action.ZOOM_OUT,      cam.zoom_out)
    inputs.register(Action.RESET_CAMERA,  lambda: cam.reset(renderer.width, renderer.height))
    inputs.register(Action.FIT_CAMERA,    lambda: cam.fit_to_graph(state.graph) if state.graph else None)

    # Renderer tweaks
    inputs.register(Action.CYCLE_COLOR,   renderer.cycle_color_scheme)
    inputs.register(Action.GLOW_UP,       lambda: renderer.adjust_glow(+0.2))
    inputs.register(Action.GLOW_DOWN,     lambda: renderer.adjust_glow(-0.2))
    inputs.register(Action.CHROMATIC_UP,  lambda: renderer.adjust_chromatic(+0.003))
    inputs.register(Action.CHROMATIC_DOWN,lambda: renderer.adjust_chromatic(-0.003))
    inputs.register(Action.TOGGLE_HUD,    hud.toggle)

    # Export
    def do_export_png():
        path = export_png(renderer.width, renderer.height)
        print(f"[Main] PNG exported: {path}")

    def do_export_svg():
        if state.graph:
            visible = controller.visible_edges
            path = export_svg(
                state.graph,
                visible_edges=visible if visible else None,
                color_scheme=renderer.color_scheme,
                meta={
                    "preset_name": state.current_preset_key,
                    "iterations":  state.iterations,
                    "angle":       state.angle_deg,
                },
            )
            print(f"[Main] SVG exported: {path}")

    inputs.register(Action.EXPORT_PNG, do_export_png)
    inputs.register(Action.EXPORT_SVG, do_export_svg)
    inputs.register(Action.QUIT, lambda: setattr(state, "running", False))

    # ── Initial build ─────────────────────────────────────────────────────────
    print("[Main] Starting initial build...")
    rebuild(state, generator, builder, optimizer, renderer, controller)

    print("\n[Main] Entering render loop. Press H for controls help.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    clock = pygame.time.Clock()

    while state.running:
        # ── Events ────────────────────────────────────────────────────────────
        if not inputs.process_events():
            break

        # Mouse pan
        pan = inputs.get_pan_delta()
        if pan:
            cam.pan(pan[0], pan[1])

        # ── Growth tick ───────────────────────────────────────────────────────
        growth_state = controller.tick()
        renderer.set_visible_count(growth_state.visible_edge_count)

        # ── OpenGL render ─────────────────────────────────────────────────────
        renderer.render()

        # ── HUD overlay (Pygame 2D over OpenGL) ───────────────────────────────
        fps = state.update_fps()

        if hud.visible:
            # Create transparent surface for HUD
            hud_surf = pygame.Surface(
                (renderer.width, renderer.height), pygame.SRCALPHA
            )
            hud_surf.fill((0, 0, 0, 0))

            hud_state = {
                "preset_name":  state.current_preset_key,
                "iterations":   state.iterations,
                "angle":        state.angle_deg,
                "growth_pct":   growth_state.growth_fraction,
                "speed":        controller.engine.growth_speed,
                "edge_count":   state.graph.edge_count() if state.graph else 0,
                "node_count":   state.graph.node_count() if state.graph else 0,
                "fps":          fps,
                "color_scheme": renderer.color_scheme,
                "mode":         controller.engine.mode,
                "paused":       controller.paused,
            }
            hud.render(hud_surf, hud_state)

            # Blit transparent HUD surface over the OpenGL frame
            screen = pygame.display.get_surface()
            screen.blit(hud_surf, (0, 0))
            pygame.display.flip()

        # ── Frame timing ──────────────────────────────────────────────────────
        clock.tick(60)   # Cap at 60 fps

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[Main] Shutting down.")
    renderer.shutdown()


if __name__ == "__main__":
    main()
