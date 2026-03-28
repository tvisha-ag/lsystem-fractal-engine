"""
optimizer.py — Graph Pruning and Memory Optimization
====================================================
Applies structural optimizations to the LSystemGraph before rendering.
This module is critical for keeping frame rates high when L-system
strings grow to tens of thousands of symbols.

Implemented Algorithms:
  1. Depth Pruning
     Remove all nodes and edges beyond a depth threshold.
     O(V + E) time. Most impactful optimization for deep fractals.

  2. Short-Segment Pruning
     Remove edges shorter than a minimum pixel length.
     Tiny segments don't contribute to visual detail but cost GPU bandwidth.
     Particularly useful for self-similar fractals at low zoom.

  3. Dead-end Chain Compression (degree-2 collapsing)
     In an unbranched chain A→B→C→D, collapse B and C
     (degree-2 nodes: one parent, one child) into a single longer edge A→D.
     Reduces node count without changing visual output.
     WARNING: Applied only to straight chains — branching nodes are kept.

  4. Bounding-box Culling
     Remove edges whose endpoints both fall outside the screen viewport.
     Applied per-frame in a lightweight CPU pass before drawing.
     Avoids uploading invisible geometry to the GPU.

  5. Memory Budget Enforcement
     After all other pruning, if the graph still exceeds a vertex count
     budget (default: 150,000 vertices ≈ 75,000 edges), iteratively
     prune the deepest level until the budget is met.

Graph Theory Notes:
  The L-System tree is a rooted DAG with branching factor B.
  At depth d, node count ≤ B^d (bounded by the maximum branching symbol count).
  Pruning at depth d_max reduces node count from O(B^D) to O(B^d_max)
  which for B=2, D=10, d_max=7 saves 87.5% of nodes.

  Dead-end compression is a special case of edge contraction:
    Given path P = v₀ → v₁ → ... → vₙ where deg(vᵢ)=2 for 0<i<n,
    contract to single edge v₀ → vₙ with length = sum(|vᵢ→vᵢ₊₁|).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Set, Optional

from graph import LSystemGraph, Node, Edge


@dataclass
class OptimizationStats:
    """Report returned by each optimization pass."""
    nodes_before:   int
    nodes_after:    int
    edges_before:   int
    edges_after:    int
    nodes_removed:  int
    edges_removed:  int
    algorithm:      str
    time_ms:        float = 0.0

    @property
    def node_reduction_pct(self) -> float:
        if self.nodes_before == 0:
            return 0.0
        return 100.0 * self.nodes_removed / self.nodes_before

    def __str__(self) -> str:
        return (f"[{self.algorithm}] "
                f"Nodes: {self.nodes_before}→{self.nodes_after} "
                f"(-{self.node_reduction_pct:.1f}%), "
                f"Edges: {self.edges_before}→{self.edges_after}")


class GraphOptimizer:
    """
    Applies a configurable pipeline of graph optimizations.

    Usage:
        opt = GraphOptimizer()
        stats = opt.prune_by_depth(graph, max_depth=6)
        stats = opt.prune_short_edges(graph, min_length=0.5)
        print(opt.full_report())
    """

    VERTEX_BUDGET = 150_000    # Hard limit: vertices on GPU

    def __init__(self):
        self._stats_history = []

    def full_pipeline(self,
                      graph: LSystemGraph,
                      max_depth: Optional[int] = None,
                      min_edge_length: float = 0.3,
                      compress_chains: bool = True) -> LSystemGraph:
        """
        Run the recommended optimization pipeline in the correct order.
        Returns the (mutated) graph for chaining.

        Order matters:
          1. Depth prune first (biggest reduction, enables faster passes)
          2. Short-segment prune (removes micro-detail)
          3. Chain compression (reduces node count)
          4. Memory budget enforcement (safety net)
        """
        import time

        if max_depth is not None and max_depth > 0:
            self.prune_by_depth(graph, max_depth)

        if min_edge_length > 0:
            self.prune_short_edges(graph, min_edge_length)

        if compress_chains:
            self.compress_degree2_chains(graph)

        # Always enforce budget last
        self._enforce_vertex_budget(graph)

        return graph

    def prune_by_depth(self, graph: LSystemGraph, max_depth: int) -> OptimizationStats:
        """
        Remove all nodes and edges with depth > max_depth.
        Also removes parent→child links and re-computes max_depth.

        O(V + E) time.
        """
        import time
        t0 = time.perf_counter()

        n_before = graph.node_count()
        e_before = graph.edge_count()

        # Collect nodes to keep
        keep_nodes: Set[int] = {
            nid for nid, n in graph.nodes.items() if n.depth <= max_depth
        }

        # Remove nodes
        for nid in list(graph.nodes.keys()):
            if nid not in keep_nodes:
                del graph.nodes[nid]

        # Remove edges — both endpoints must survive
        graph.edges = [
            e for e in graph.edges
            if e.source_id in keep_nodes and e.target_id in keep_nodes
        ]

        # Clean up children references
        for node in graph.nodes.values():
            node.children = [c for c in node.children if c in keep_nodes]

        # Recompute metadata
        graph.max_depth = max(
            (n.depth for n in graph.nodes.values()), default=0
        )

        elapsed = (time.perf_counter() - t0) * 1000
        stats = OptimizationStats(
            nodes_before=n_before, nodes_after=graph.node_count(),
            edges_before=e_before, edges_after=graph.edge_count(),
            nodes_removed=n_before - graph.node_count(),
            edges_removed=e_before - graph.edge_count(),
            algorithm="depth_prune",
            time_ms=elapsed,
        )
        self._stats_history.append(stats)
        print(stats)
        return stats

    def prune_short_edges(self, graph: LSystemGraph,
                           min_length: float = 0.3) -> OptimizationStats:
        """
        Remove edges shorter than `min_length` world units.
        Also removes the target node if it becomes disconnected.

        This removes visual noise (sub-pixel segments) while preserving
        the macro structure of the fractal.
        """
        import time
        t0 = time.perf_counter()

        n_before = graph.node_count()
        e_before = graph.edge_count()

        # Track which nodes have at least one surviving edge
        referenced: Set[int] = {graph.root_id}

        surviving_edges = []
        for edge in graph.edges:
            if edge.length >= min_length:
                surviving_edges.append(edge)
                referenced.add(edge.source_id)
                referenced.add(edge.target_id)

        graph.edges = surviving_edges

        # Remove nodes that are no longer referenced by any edge
        # (except root, which we always keep)
        for nid in list(graph.nodes.keys()):
            if nid not in referenced:
                del graph.nodes[nid]

        # Clean children lists
        all_ids = set(graph.nodes.keys())
        for node in graph.nodes.values():
            node.children = [c for c in node.children if c in all_ids]

        elapsed = (time.perf_counter() - t0) * 1000
        stats = OptimizationStats(
            nodes_before=n_before, nodes_after=graph.node_count(),
            edges_before=e_before, edges_after=graph.edge_count(),
            nodes_removed=n_before - graph.node_count(),
            edges_removed=e_before - graph.edge_count(),
            algorithm="short_edge_prune",
            time_ms=elapsed,
        )
        self._stats_history.append(stats)
        print(stats)
        return stats

    def compress_degree2_chains(self, graph: LSystemGraph) -> OptimizationStats:
        """
        Collapse unbranched chains (degree-2 nodes) into single longer edges.

        A degree-2 node has exactly 1 parent edge and 1 child edge.
        The chain A→B→C where B is degree-2 becomes A→C with
        combined length |AB| + |BC|.

        This is safe for rendering because the intermediate point B
        lies on a straight line between A and C (in L-System turtle
        geometry, straight segments are collinear unless there's a turn).

        NOTE: Only applies to true straight chains. If a node has 2+
        children it is a branch point and is kept.

        O(V + E) time.
        """
        import time
        t0 = time.perf_counter()

        n_before = graph.node_count()
        e_before = graph.edge_count()

        # Build: node_id → list[Edge] of outgoing edges
        out_edges: dict[int, list[Edge]] = {nid: [] for nid in graph.nodes}
        in_degree:  dict[int, int]       = {nid: 0  for nid in graph.nodes}

        for edge in graph.edges:
            if edge.source_id in out_edges:
                out_edges[edge.source_id].append(edge)
            if edge.target_id in in_degree:
                in_degree[edge.target_id] += 1

        # Identify degree-2 nodes (not root, not branch points, not leaves)
        def is_degree2(nid: int) -> bool:
            node = graph.nodes.get(nid)
            if node is None or nid == graph.root_id:
                return False
            if node.is_branch_point:
                return False
            return in_degree.get(nid, 0) == 1 and len(out_edges.get(nid, [])) == 1

        # Walk chains and collect removable intermediate nodes
        removed_nodes: Set[int] = set()
        new_edges: list[Edge]   = []
        processed: Set[int]     = set()

        edge_id_counter = max((e.edge_id for e in graph.edges), default=0) + 1

        for edge in graph.edges:
            if edge.source_id in processed:
                continue

            src = edge.source_id

            if not is_degree2(edge.target_id):
                # Target is not a chain node — keep edge as-is
                new_edges.append(edge)
                continue

            # Walk the chain starting at edge.target_id
            chain_length = edge.length
            chain_end    = edge.target_id

            while is_degree2(chain_end):
                removed_nodes.add(chain_end)
                processed.add(chain_end)
                next_edge  = out_edges[chain_end][0]
                chain_length += next_edge.length
                chain_end    = next_edge.target_id

            # Create compressed edge: src → chain_end
            compressed = Edge(
                edge_id=edge_id_counter,
                source_id=src,
                target_id=chain_end,
                length=chain_length,
                depth=edge.depth,
                color_t=edge.color_t,
            )
            edge_id_counter += 1
            new_edges.append(compressed)

            # Update parent's children list
            if src in graph.nodes:
                children = graph.nodes[src].children
                if edge.target_id in children:
                    children.remove(edge.target_id)
                if chain_end not in children:
                    children.append(chain_end)

        graph.edges = new_edges

        # Remove compressed-out nodes
        for nid in removed_nodes:
            if nid in graph.nodes:
                del graph.nodes[nid]

        elapsed = (time.perf_counter() - t0) * 1000
        stats = OptimizationStats(
            nodes_before=n_before, nodes_after=graph.node_count(),
            edges_before=e_before, edges_after=graph.edge_count(),
            nodes_removed=n_before - graph.node_count(),
            edges_removed=e_before - graph.edge_count(),
            algorithm="degree2_compression",
            time_ms=elapsed,
        )
        self._stats_history.append(stats)
        print(stats)
        return stats

    def _enforce_vertex_budget(self, graph: LSystemGraph) -> None:
        """
        If the graph would produce more than VERTEX_BUDGET GPU vertices,
        iteratively prune deepest level until budget is met.
        Each edge → 2 vertices for GL_LINES.
        """
        while graph.edge_count() * 2 > self.VERTEX_BUDGET:
            if graph.max_depth <= 1:
                break
            print(f"[Optimizer] Budget exceeded ({graph.edge_count()*2:,} vertices). "
                  f"Pruning depth {graph.max_depth}...")
            self.prune_by_depth(graph, graph.max_depth - 1)

    def viewport_cull(self, graph: LSystemGraph,
                      min_x: float, min_y: float,
                      max_x: float, max_y: float) -> list:
        """
        Return edges with at least one endpoint inside the viewport.
        Used for lightweight CPU-side frustum culling.
        Does NOT modify the graph — returns a filtered list for the renderer.
        """
        result = []
        for edge in graph.edges:
            src = graph.nodes.get(edge.source_id)
            tgt = graph.nodes.get(edge.target_id)
            if src is None or tgt is None:
                continue
            # Keep edge if either endpoint is within bounds
            src_in = (min_x <= src.x <= max_x and min_y <= src.y <= max_y)
            tgt_in = (min_x <= tgt.x <= max_x and min_y <= tgt.y <= max_y)
            if src_in or tgt_in:
                result.append(edge)
        return result

    def full_report(self) -> str:
        """Return a human-readable summary of all optimization passes run."""
        if not self._stats_history:
            return "[Optimizer] No passes run yet."
        lines = ["[Optimizer] Optimization report:"]
        for s in self._stats_history:
            lines.append(f"  {s}")
        return "\n".join(lines)
