"""
growth_engine.py — DFS Growth Engine with Depth Limiting
=========================================================
Implements an iterative DFS traversal over the LSystemGraph to produce
a time-ordered sequence of edges for animated rendering.

Why DFS for fractal growth?
  L-Systems are inherently recursive / tree-structured. DFS mirrors how
  a plant actually grows: extend one branch fully before starting the next.
  This creates the organic "growing tip" visual effect.

Depth limiting strategy:
  Rather than restricting the graph at build time, the growth engine
  controls which edges are VISIBLE at any given simulation frame.
  A `growth_fraction` parameter (0.0 → 1.0) drives how much of the
  DFS order has been revealed so far.

  This approach separates concerns:
    - Graph module: full structure, always complete
    - Growth engine: controls temporal reveal order
    - Renderer:     draws whatever the engine says is visible

Performance notes:
  - DFS order is pre-computed once after each graph rebuild.
  - Per-frame work is a single integer comparison: O(1) per edge.
  - No recursion at frame time — eliminates any stack overflow risk.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Iterator, Tuple, Optional
from collections import deque

from graph import LSystemGraph, Edge, Node


@dataclass
class GrowthState:
    """
    Current state of the growth animation.
    Updated each frame by the GrowthEngine.
    """
    visible_edge_count: int    = 0    # How many edges are currently shown
    total_edge_count:   int    = 0    # Total edges in the graph
    growth_fraction:    float  = 0.0  # 0.0 = empty, 1.0 = fully grown
    is_complete:        bool   = False
    elapsed_seconds:    float  = 0.0
    current_depth:      int    = 0    # Deepest currently-visible depth


class GrowthEngine:
    """
    Manages the animated reveal of an LSystemGraph using DFS order.

    Lifecycle:
      1. Call load_graph(graph) to precompute the DFS traversal order.
      2. Each frame, call step(dt) to advance the animation.
      3. Query visible_edges() to get the list of edges to render.
      4. Reset with reset() to restart the animation.

    Growth modes:
      - "dfs"   : Depth-first (default) — grows one branch at a time
      - "bfs"   : Breadth-first — grows level by level (like a flood fill)
      - "random": Randomised reveal order (chaotic / generative feel)
    """

    def __init__(self,
                 growth_speed: float = 30.0,
                 mode: str = "dfs",
                 max_depth_limit: int = 999):
        """
        Args:
            growth_speed:    Edges revealed per second.
            mode:            Traversal order: "dfs", "bfs", or "random".
            max_depth_limit: Ignore edges deeper than this (runtime depth cap).
        """
        self.growth_speed   = growth_speed     # edges / second
        self.mode           = mode
        self.max_depth_limit = max_depth_limit

        self._ordered_edges: List[Edge] = []   # Pre-computed reveal sequence
        self._graph: Optional[LSystemGraph] = None
        self._state = GrowthState()
        self._fractional_index: float = 0.0   # Floating-point cursor into ordered list
        self._start_time: float = time.perf_counter()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_graph(self, graph: LSystemGraph) -> None:
        """
        Pre-compute the DFS (or BFS/random) traversal order.
        Must be called once whenever the graph changes.

        Time complexity: O(V + E)
        """
        self._graph = graph
        self._ordered_edges = list(self._compute_order(graph))
        self._fractional_index = 0.0
        self._state = GrowthState(total_edge_count=len(self._ordered_edges))

    def step(self, dt: float) -> GrowthState:
        """
        Advance the growth animation by dt seconds.
        Returns the updated GrowthState.

        Args:
            dt: Delta time in seconds since last frame.
        """
        if not self._ordered_edges:
            return self._state

        # Advance fractional cursor
        self._fractional_index += self.growth_speed * dt
        self._fractional_index = min(self._fractional_index,
                                     float(len(self._ordered_edges)))

        visible_count = int(self._fractional_index)

        # Update state
        self._state.visible_edge_count = visible_count
        self._state.total_edge_count   = len(self._ordered_edges)
        self._state.growth_fraction    = (visible_count / len(self._ordered_edges)
                                          if self._ordered_edges else 0.0)
        self._state.is_complete        = visible_count >= len(self._ordered_edges)
        self._state.elapsed_seconds    = time.perf_counter() - self._start_time

        if visible_count > 0:
            self._state.current_depth = self._ordered_edges[visible_count - 1].depth

        return self._state

    def visible_edges(self) -> List[Edge]:
        """Return the slice of edges that should be drawn this frame."""
        n = self._state.visible_edge_count
        return self._ordered_edges[:n]

    def all_edges(self) -> List[Edge]:
        """Return the full ordered edge list (for non-animated static rendering)."""
        return list(self._ordered_edges)

    def reset(self) -> None:
        """Restart growth animation from empty."""
        self._fractional_index = 0.0
        self._start_time = time.perf_counter()
        self._state = GrowthState(total_edge_count=len(self._ordered_edges))

    def seek(self, fraction: float) -> None:
        """
        Jump to a specific growth fraction [0.0, 1.0].
        Used for scrubbing or instant-reveal.
        """
        fraction = max(0.0, min(1.0, fraction))
        self._fractional_index = fraction * len(self._ordered_edges)
        self.step(0.0)  # Refresh state

    def set_speed(self, speed: float) -> None:
        """Update growth speed (edges per second). Clamped to [1, 5000]."""
        self.growth_speed = max(1.0, min(5000.0, speed))

    def get_state(self) -> GrowthState:
        return self._state

    # ── Traversal order computation ────────────────────────────────────────────

    def _compute_order(self, graph: LSystemGraph) -> Iterator[Edge]:
        """
        Dispatch to the appropriate traversal algorithm.
        """
        if self.mode == "dfs":
            return self._dfs_order(graph)
        elif self.mode == "bfs":
            return self._bfs_order(graph)
        elif self.mode == "random":
            return self._random_order(graph)
        else:
            return self._dfs_order(graph)

    def _dfs_order(self, graph: LSystemGraph) -> List[Edge]:
        """
        Iterative DFS traversal using an explicit stack.
        Avoids Python recursion limits — safe for any graph depth.

        Produces edges in depth-first order: each branch is fully
        extended before the algorithm backtracks.
        """
        if not graph.nodes:
            return []

        # Build adjacency lookup: node_id → list of outgoing edges
        adj: dict[int, List[Edge]] = {nid: [] for nid in graph.nodes}
        for edge in graph.edges:
            if edge.source_id in adj:
                adj[edge.source_id].append(edge)

        result: List[Edge] = []
        visited: set[int] = set()

        # Iterative DFS: stack holds (node_id)
        stack: List[int] = [graph.root_id]
        visited.add(graph.root_id)

        while stack:
            node_id = stack.pop()
            node = graph.nodes.get(node_id)
            if node is None:
                continue

            # Depth limit check
            if node.depth > self.max_depth_limit:
                continue

            # Emit edges from this node in reverse child order
            # (reversed so the first child is processed first after pop)
            for edge in reversed(adj.get(node_id, [])):
                if edge.target_id not in visited:
                    if edge.depth <= self.max_depth_limit:
                        result.append(edge)
                        visited.add(edge.target_id)
                        stack.append(edge.target_id)

        return result

    def _bfs_order(self, graph: LSystemGraph) -> List[Edge]:
        """
        BFS traversal: level-by-level reveal.
        Creates a "branching outward" growth effect.
        """
        if not graph.nodes:
            return []

        adj: dict[int, List[Edge]] = {nid: [] for nid in graph.nodes}
        for edge in graph.edges:
            if edge.source_id in adj:
                adj[edge.source_id].append(edge)

        result: List[Edge] = []
        visited: set[int] = {graph.root_id}
        queue: deque[int] = deque([graph.root_id])

        while queue:
            node_id = queue.popleft()
            for edge in adj.get(node_id, []):
                if edge.target_id not in visited:
                    if edge.depth <= self.max_depth_limit:
                        result.append(edge)
                        visited.add(edge.target_id)
                        queue.append(edge.target_id)

        return result

    def _random_order(self, graph: LSystemGraph) -> List[Edge]:
        """
        Shuffle the DFS order for a chaotic generative reveal.
        """
        import random
        edges = self._dfs_order(graph)
        random.shuffle(edges)
        return edges


class GrowthController:
    """
    High-level controller that wraps GrowthEngine and exposes
    simple keyboard-driven controls for the UI layer.

    This is the interface the main program uses — it never touches
    the engine internals directly.
    """

    def __init__(self):
        self.engine = GrowthEngine(growth_speed=80.0, mode="dfs")
        self.paused = False
        self._last_frame_time = time.perf_counter()

    def load(self, graph: LSystemGraph) -> None:
        self.engine.load_graph(graph)
        self._last_frame_time = time.perf_counter()

    def tick(self) -> GrowthState:
        """Call once per frame. Returns current growth state."""
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now

        if not self.paused:
            return self.engine.step(dt)
        return self.engine.get_state()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def restart(self) -> None:
        self.engine.reset()
        self.paused = False

    def reveal_all(self) -> None:
        self.engine.seek(1.0)

    def speed_up(self) -> None:
        self.engine.set_speed(self.engine.growth_speed * 1.5)

    def speed_down(self) -> None:
        self.engine.set_speed(self.engine.growth_speed / 1.5)

    def set_mode(self, mode: str) -> None:
        self.engine.mode = mode
        if self.engine._graph:
            # Recompute order with new mode
            self.engine._ordered_edges = list(self.engine._compute_order(self.engine._graph))
            self.engine.reset()

    @property
    def visible_edges(self) -> List[Edge]:
        return self.engine.visible_edges()

    @property
    def state(self) -> GrowthState:
        return self.engine.get_state()
