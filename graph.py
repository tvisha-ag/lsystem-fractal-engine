"""
graph.py — Graph / Tree Data Structure for L-System Geometry
============================================================
Converts an L-System string into an explicit graph where:
  - Each node represents a turtle-graphics position in 2D space
  - Each edge represents a line segment (forward move)
  - The graph preserves parent-child relationships for DFS traversal

Turtle Graphics Alphabet (standard L-System symbols):
  F, G  → Move forward, draw a line segment
  f     → Move forward, do NOT draw (jump)
  +     → Turn left by `angle` degrees
  -     → Turn right by `angle` degrees
  [     → Push state onto stack (start a branch)
  ]     → Pop state from stack (return to branch point)
  |     → Turn 180 degrees (reverse direction)
  !     → Decrement step length
  >     → Multiply step length by scale factor
  <     → Divide step length by scale factor

Graph Theory Background:
  The resulting structure is a directed acyclic graph (DAG) — specifically
  a rooted tree because branch pushes/pops create a parent-child hierarchy.
  Nodes at depth d from the root correspond to L-System generation d.

  Pruning is applied post-construction:
    - Remove isolated nodes (no edges)
    - Collapse degree-2 nodes on unbranched chains (edge compression)
    - Limit maximum depth for performance control
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


# ─── Data types ───────────────────────────────────────────────────────────────

@dataclass
class Node:
    """
    A single vertex in the L-System graph.
    Stores 2D position plus metadata for rendering and traversal.
    """
    node_id:  int              # Unique identifier (assigned during construction)
    x:        float            # World-space X coordinate
    y:        float            # World-space Y coordinate
    depth:    int              # Depth from root (0 = starting point)
    parent:   Optional[int]    # Parent node id (None for root)
    children: List[int] = field(default_factory=list)  # Child node ids
    is_branch_point: bool = False  # True if this node starts a [ branch ]
    angle_at_node: float = 0.0    # Heading when this node was created (radians)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, other: "Node") -> float:
        dx, dy = self.x - other.x, self.y - other.y
        return math.hypot(dx, dy)


@dataclass
class Edge:
    """
    A directed edge between two nodes — represents a drawn line segment.
    Carries visual properties used by the shader/renderer.
    """
    edge_id:    int
    source_id:  int            # Parent node id
    target_id:  int            # Child node id
    length:     float          # Segment length in world space
    depth:      int            # Depth of the edge (depth of source node)
    # Visual properties (computed during graph build, used by renderer)
    color_t:    float = 0.0    # Normalized depth-based color lerp parameter [0,1]

    @property
    def key(self) -> Tuple[int, int]:
        return (self.source_id, self.target_id)


@dataclass
class LSystemGraph:
    """
    Complete graph representation of an L-System string expansion.

    Provides O(1) node/edge lookup by id and supports:
      - DFS traversal via children lists
      - Rendering via ordered edge list
      - Pruning/optimization operations
    """
    nodes:      Dict[int, Node] = field(default_factory=dict)
    edges:      List[Edge]       = field(default_factory=list)
    root_id:    int              = 0
    max_depth:  int              = 0
    bounds:     Tuple[float,float,float,float] = (0,0,0,0)  # (min_x,min_y,max_x,max_y)
    symbol_count: int            = 0

    def get_node(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_children(self, node_id: int) -> List[Node]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[c] for c in node.children if c in self.nodes]

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return len(self.edges)

    def get_edges_sorted_by_depth(self) -> List[Edge]:
        """Return edges ordered root→leaves — used for animated growth rendering."""
        return sorted(self.edges, key=lambda e: e.depth)

    def get_leaf_nodes(self) -> List[Node]:
        """Return nodes with no children (tips of branches)."""
        return [n for n in self.nodes.values() if len(n.children) == 0]

    def compute_center(self) -> Tuple[float, float]:
        """Centroid of all node positions."""
        if not self.nodes:
            return (0.0, 0.0)
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


# ─── Builder ──────────────────────────────────────────────────────────────────

class LSystemGraphBuilder:
    """
    Converts an L-System result string into an LSystemGraph via turtle graphics
    simulation. Uses an explicit stack to handle branch push/pop ( [ ] ).

    Turtle state: (x, y, angle_radians, step_length, depth)

    Complexity:
      Time  O(n) where n = length of L-System string
      Space O(d) stack depth where d = max branch depth
    """

    def __init__(self, max_depth: int = 50):
        """
        Args:
            max_depth: Discard branches deeper than this. Hard depth limit
                       prevents runaway recursion in DFS traversal later.
        """
        self.max_depth = max_depth
        self._node_counter = 0
        self._edge_counter = 0

    def build(self,
              lsystem_string: str,
              step_length: float = 5.0,
              angle_deg: float = 25.0,
              start_x: float = 0.0,
              start_y: float = 0.0,
              start_angle_deg: float = 90.0) -> LSystemGraph:
        """
        Main entry point: parse L-System string → LSystemGraph.

        Args:
            lsystem_string: Output of LSystemGenerator.generate()
            step_length:    Length of each F/G forward move (world units)
            angle_deg:      Turn angle for +/- symbols (degrees)
            start_x/y:      Turtle starting position
            start_angle_deg: Initial heading (90° = pointing up)

        Returns:
            Fully constructed LSystemGraph
        """
        self._node_counter = 0
        self._edge_counter = 0

        angle_rad = math.radians(angle_deg)
        start_angle_rad = math.radians(start_angle_deg)

        graph = LSystemGraph()

        # ── Create root node ──────────────────────────────────────────────────
        root = self._new_node(start_x, start_y, 0, None, start_angle_rad)
        graph.nodes[root.node_id] = root
        graph.root_id = root.node_id

        # ── Turtle state stack ────────────────────────────────────────────────
        # Each stack frame: (current_node_id, x, y, heading_rad, step_len, depth)
        current_id  = root.node_id
        x           = start_x
        y           = start_y
        heading     = start_angle_rad
        current_step = step_length
        depth        = 0

        stack: List[Tuple[int, float, float, float, float, int]] = []

        # ── Symbol dispatch ───────────────────────────────────────────────────
        for symbol in lsystem_string:

            if symbol in ("F", "G"):
                # Forward move with line drawn
                if depth <= self.max_depth:
                    nx = x + current_step * math.cos(heading)
                    ny = y + current_step * math.sin(heading)

                    child = self._new_node(nx, ny, depth + 1, current_id, heading)
                    graph.nodes[child.node_id] = child

                    # Register edge
                    edge = self._new_edge(current_id, child.node_id,
                                          current_step, depth)
                    graph.edges.append(edge)

                    # Link parent → child
                    graph.nodes[current_id].children.append(child.node_id)

                    # Advance turtle
                    x, y = nx, ny
                    current_id = child.node_id
                    depth += 1

            elif symbol == "f":
                # Forward move without drawing (teleport)
                x += current_step * math.cos(heading)
                y += current_step * math.sin(heading)

            elif symbol == "+":
                heading += angle_rad      # Turn left

            elif symbol == "-":
                heading -= angle_rad      # Turn right

            elif symbol == "|":
                heading += math.pi        # Reverse (180°)

            elif symbol == "[":
                # Push current turtle state onto stack
                if depth <= self.max_depth:
                    graph.nodes[current_id].is_branch_point = True
                stack.append((current_id, x, y, heading, current_step, depth))

            elif symbol == "]":
                # Pop turtle state — return to saved branch point
                if stack:
                    current_id, x, y, heading, current_step, depth = stack.pop()

            elif symbol == "!":
                # Decrement step length (taper branches)
                current_step = max(1.0, current_step * 0.85)

            elif symbol in (">", "*"):
                current_step *= 1.1      # Grow step

            elif symbol == "<":
                current_step *= 0.9      # Shrink step

            # All other symbols (variables like X, Y, A, B) have no turtle action
            # They are purely grammatical and only appear in rules, not turtle output

        # ── Compute graph metadata ─────────────────────────────────────────────
        graph.max_depth = max((n.depth for n in graph.nodes.values()), default=0)
        graph.symbol_count = len(lsystem_string)
        graph.bounds = self._compute_bounds(graph)

        # ── Normalise color_t on all edges ─────────────────────────────────────
        if graph.max_depth > 0:
            for edge in graph.edges:
                edge.color_t = edge.depth / graph.max_depth
        return graph

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _new_node(self, x, y, depth, parent_id, angle) -> Node:
        nid = self._node_counter
        self._node_counter += 1
        return Node(node_id=nid, x=x, y=y, depth=depth,
                    parent=parent_id, angle_at_node=angle)

    def _new_edge(self, src, tgt, length, depth) -> Edge:
        eid = self._edge_counter
        self._edge_counter += 1
        return Edge(edge_id=eid, source_id=src, target_id=tgt,
                    length=length, depth=depth)

    @staticmethod
    def _compute_bounds(graph: LSystemGraph) -> Tuple[float,float,float,float]:
        if not graph.nodes:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [n.x for n in graph.nodes.values()]
        ys = [n.y for n in graph.nodes.values()]
        return (min(xs), min(ys), max(xs), max(ys))
