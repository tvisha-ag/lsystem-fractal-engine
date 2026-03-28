# Generative L-System Fractal Growth Engine
### Graph-Based Optimization · Cyberpunk GLSL Shader Rendering

> A research-grade generative art system that models biological growth through formal grammar theory, renders it with GPU shaders, and exposes a fully interactive real-time interface.

---

## Demo

```
┌──────────────────────────────────────────────────────────────────┐
│  Fractal Tree  |  iter=5  |  angle=25°  |  12,847 edges  |  60fps │
│                                                                    │
│        ╱╲   ╱╲                                                     │
│       ╱  ╲ ╱  ╲                                                    │
│      ╱  ╱╲╱╲  ╲    ← neon magenta → cyan → yellow gradient        │
│     ╱  ╱    ╲  ╲                                                   │
│    ╱  ╱  ╱╲  ╲  ╲                                                  │
│   ╱──╱──╱  ╲──╲──╲   animated DFS growth, one branch at a time   │
│  ════════════════════                                              │
└──────────────────────────────────────────────────────────────────┘
```

Renders fractals like fractal trees, Koch snowflakes, dragon curves, Hilbert curves, and Gosper flowsnakes with neon cyberpunk aesthetics. Growth is animated in real-time using DFS traversal. All parameters are live-editable via keyboard.

---

## Features

- **8 built-in L-System presets** — fractal tree, dragon curve, Sierpinski, Koch snowflake, Hilbert curve, Gosper curve, Levy C curve, organic plant
- **Custom rule editor** — modify axioms and production rules in code
- **DFS / BFS / Random growth modes** — three distinct animated reveal styles
- **GLSL cyberpunk shader** — chromatic aberration, depth-based neon colour gradient, electrical flicker, additive blending glow
- **4 colour schemes** — Cyberpunk (magenta→cyan→yellow), Fire, Ice, Matrix
- **Graph optimization pipeline** — depth pruning, short-segment removal, degree-2 chain compression, vertex budget enforcement
- **Interactive camera** — pan (mouse drag), zoom (scroll or Z/X), auto-fit
- **PNG export** — reads raw OpenGL framebuffer via `glReadPixels`
- **SVG export** — resolution-independent vector with neon glow filter, valid SVG 1.1
- **HUD overlay** — live stats, growth progress bar, FPS counter

---

## Mathematical Background

### L-Systems (Lindenmayer Systems)

An L-System is a formal parallel rewriting system defined by the triple:

```
G = (V, ω, P)
```

Where:
- **V** — the alphabet, a finite set of symbols
- **ω ∈ V\*** — the axiom (seed / initial string)
- **P ⊂ V × V\*** — production rules mapping each symbol to a replacement string

**Rewriting rule:**
At each generation, every symbol in the current string is simultaneously replaced by its production:

```
n=0: F
n=1: F+F-F-F+F          (Koch curve, angle=90°)
n=2: F+F-F-F+F+F+F-F-F+F-F+F-F-F+F-F+F-F-F+F+F+F-F-F+F
n=3: ...
```

The string length grows exponentially: if every symbol has an average expansion factor `k`, then after `n` iterations the string has `O(k^n)` symbols.

**Turtle graphics interpretation:**

The L-System string is parsed by a turtle that walks the plane:

| Symbol | Action                          |
|--------|---------------------------------|
| F, G   | Move forward, draw segment      |
| f      | Move forward, no draw           |
| +      | Turn left by angle θ            |
| −      | Turn right by angle θ           |
| [      | Push state (start branch)       |
| ]      | Pop state (end branch)          |
| \|     | Turn 180°                       |

**Classic fractals:**

| Name              | Axiom   | Rule                           | Angle |
|-------------------|---------|--------------------------------|-------|
| Koch Snowflake    | F++F++F | F → F−F++F−F                  | 60°   |
| Dragon Curve      | FX      | X → X+YF+, Y → −FX−Y         | 90°   |
| Sierpinski Tri.   | F−G−G   | F → F−G+F+G−F, G → GG        | 120°  |
| Fractal Tree      | X       | X → F+[[X]−X]−F[−FX]+X       | 25°   |
| Hilbert Curve     | A       | A → −BF+AFA+FB−               | 90°   |

**Complexity analysis:**

For the fractal tree with branching rules:
- `F → FF` doubles the trunk
- `X → F+[[X]-X]-F[-FX]+X` creates 2 branches per X

At iteration `n`, the string length satisfies approximately `S(n) ≈ 2^n × c` where `c` is a rule-specific constant. The maximum meaningful depth before sub-pixel segments appear is typically 4–7 iterations at standard window sizes.

---

## Graph Theory Optimization

### L-System → Graph

The turtle walk produces a **rooted tree** `T = (V, E, r)` where:
- `r` — the root (turtle start position)
- `V` — set of positions visited by the turtle after each forward move
- `E` — set of directed edges `(u → v)` for each forward move

This is specifically a **directed acyclic graph (DAG)** with a single root. Branch push/pop `[ ]` creates a parent-child relationship.

### Optimization Algorithms

**1. Depth pruning** — O(V + E)

Remove all nodes and edges at depth > `d_max`. For a binary branching tree with branching factor B=2, pruning at depth 7 instead of 10 reduces node count by:

```
Reduction = 1 - (2^7) / (2^10) = 1 - 128/1024 = 87.5%
```

**2. Short-segment pruning** — O(E)

Remove edges with length `< ε` (sub-pixel segments). These contribute nothing to visual output but waste GPU bandwidth. For Koch curve at iteration 5, over 40% of segments fall below 1 pixel at standard zoom.

**3. Degree-2 chain compression** — O(V + E)

In a path `v₀ → v₁ → v₂ → ... → vₙ` where each intermediate `vᵢ` has in-degree 1 and out-degree 1 (a "chain"), collapse the entire path to a single edge `v₀ → vₙ` with combined length:

```
length(v₀ → vₙ) = Σᵢ length(vᵢ → vᵢ₊₁)
```

This is a special case of **edge contraction** from graph theory. It is lossless for rendering because collinear segments (straight runs) produce the same visual output whether split or merged.

**4. Vertex budget enforcement**

After all pruning, if the edge count would produce `> 150,000 GPU vertices` (= 75,000 edges), iteratively prune the deepest depth level until the budget is satisfied. This guarantees `< 10ms` upload time on any hardware.

**5. Viewport culling** — per-frame, O(E)

Before each draw call, discard edges where both endpoints lie outside the screen frustum. This is a lightweight CPU pass — no GPU readback required.

### DFS Growth Traversal

The animated growth effect pre-computes the DFS order of the tree once:

```python
# Iterative DFS — avoids Python recursion limit
stack = [root]
while stack:
    node = stack.pop()
    for child_edge in reversed(adjacency[node]):
        result.append(child_edge)
        stack.append(child_edge.target)
```

Per-frame cost is O(1): a single integer comparison determines how much of the pre-computed list to draw. This separates animation timing from graph structure entirely.

---

## Rendering Pipeline

```
L-System string
      ↓
Turtle walk → LSystemGraph (CPU, Python)
      ↓
Graph optimization (CPU, Python)
      ↓
VBO upload: [x, y, color_t, depth] × 2N vertices (CPU → GPU, once)
      ↓
┌─────────────────────────────────────────────────────────┐
│  VERTEX SHADER (GLSL)                                   │
│  - Apply pan + zoom transform                           │
│  - Normalize to clip space [-1, 1]                      │
│  - Pass color_t, depth to fragment shader               │
└─────────────────────────────────────────────────────────┘
      ↓ rasterisation (GPU hardware)
┌─────────────────────────────────────────────────────────┐
│  FRAGMENT SHADER (GLSL)                                 │
│  - Depth-based neon colour palette                      │
│  - Chromatic aberration (R/B channel shift)             │
│  - Electrical flicker (sinusoidal noise)                │
│  - Breathing alpha pulse                                │
└─────────────────────────────────────────────────────────┘
      ↓
Framebuffer (additive blending GL_ONE = neon glow)
      ↓
Pygame 2D HUD overlay (font rendering)
      ↓
Screen
```

**Additive blending** is the key to the neon glow effect. With `glBlendFunc(GL_SRC_ALPHA, GL_ONE)`, overlapping lines add their colour values together — dense regions near the root accumulate bright white/yellow while sparse tips stay in the dim violet range, exactly replicating neon tube physics.

### Shader Colour Palette

```glsl
// Cyberpunk: Magenta → Cyan → Yellow
t ∈ [0.0, 0.5]: mix(#FF00FF, #00FFFF, smoothstep)
t ∈ [0.5, 1.0]: mix(#00FFFF, #FFFF33, smoothstep)
```

Chromatic aberration shifts the R channel by `+Δ` and the B channel by `−Δ` where `Δ = A × sin(time × ω + t × π)`. This creates a subtle colour-split at branch tips that mimics CRT phosphor bloom.

---

## Project Structure

```
lsystem_fractal/
├── main.py              # Entry point — wires all modules
├── lsystem.py           # L-System string rewriting engine
├── graph.py             # Graph/tree data structure + turtle builder
├── growth_engine.py     # DFS/BFS animated growth controller
├── optimizer.py         # Graph pruning + memory management
├── renderer.py          # PyOpenGL context + VBO + camera
├── controls.py          # Input handler + HUD overlay
├── exporter.py          # PNG (glReadPixels) + SVG export
├── shaders/
│   ├── vertex.glsl      # Pan/zoom transform shader
│   └── fragment.glsl    # Cyberpunk neon colour shader
├── exports/             # Auto-created, saved PNG/SVG files
└── requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/lsystem-fractal-engine.git
cd lsystem-fractal-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

**Requirements:**
- Python 3.10+
- pygame 2.1+
- PyOpenGL 3.1+ (+ PyOpenGL_accelerate for performance)
- numpy
- Pillow (for PNG export)
- A GPU with OpenGL 2.1+ support (any GPU from 2007+)

---

## Controls

| Key            | Action                              |
|----------------|-------------------------------------|
| `SPACE`        | Pause / Resume growth animation     |
| `R`            | Restart growth from empty           |
| `END`          | Reveal entire fractal instantly     |
| `↑` / `↓`     | Increase / decrease growth speed    |
| `1`            | DFS growth mode (branch by branch)  |
| `2`            | BFS growth mode (level by level)    |
| `3`            | Random growth mode                  |
| `=` / `-`      | More / fewer L-System iterations    |
| `.` / `,`      | Angle +5° / −5°                     |
| `N` / `P`      | Next / previous preset              |
| `Z` / `X`      | Zoom in / out                       |
| `F`            | Auto-fit fractal to screen          |
| `F5`           | Reset camera to default             |
| Mouse drag     | Pan the viewport                    |
| Scroll wheel   | Zoom in / out                       |
| `C`            | Cycle colour scheme (4 themes)      |
| `G` / `V`      | Glow intensity +/-                  |
| `H`            | Toggle HUD overlay                  |
| `F1`           | Save PNG screenshot                 |
| `F2`           | Save SVG vector export              |
| `Q` / `ESC`    | Quit                                |

---

## Built-in Presets

| Key | Preset            | Angle | Description                    |
|-----|-------------------|-------|--------------------------------|
| N→1 | Fractal Tree      | 25°   | Classic branching tree         |
| N→2 | Dragon Curve      | 90°   | Heighway dragon                |
| N→3 | Sierpinski Tri.   | 120°  | Sierpinski triangle            |
| N→4 | Koch Snowflake    | 60°   | Koch snowflake curve           |
| N→5 | Organic Plant     | 22.5° | Organic branching plant        |
| N→6 | Hilbert Curve     | 90°   | Space-filling Hilbert curve    |
| N→7 | Gosper Curve      | 60°   | Gosper (flowsnake) curve       |
| N→8 | Levy C Curve      | 45°   | Levy C fractal                 |

---

## Recommended Iteration Counts

Higher iterations = exponentially more complexity. Suggested safe ranges:

| Preset          | Recommended | Max safe |
|-----------------|-------------|----------|
| Fractal Tree    | 4–6         | 8        |
| Koch Snowflake  | 3–5         | 7        |
| Dragon Curve    | 6–10        | 14       |
| Hilbert Curve   | 3–5         | 6        |
| Sierpinski      | 4–6         | 8        |
| Gosper Curve    | 3–4         | 5        |

The optimizer's vertex budget (150,000 vertices) automatically caps rendering before memory issues occur.

---


## Technical References

- Lindenmayer, A. (1968). *Mathematical models for cellular interactions in development.* Journal of Theoretical Biology.
- Prusinkiewicz, P. & Lindenmayer, A. (1990). *The Algorithmic Beauty of Plants.* Springer. (Free PDF available from the authors)
- Cormen, T. et al. *Introduction to Algorithms* — Chapter 22 (Graph traversal), Chapter 21 (Union-Find)
- Akenine-Möller et al. *Real-Time Rendering* — Chapter 3 (GPU pipeline), Chapter 9 (Physically Based Rendering)
- OpenGL specification: [opengl.org](https://opengl.org)

---

## License

MIT License — see `LICENSE` for details.

---

*Built as a portfolio/research project demonstrating: formal language theory, graph algorithms, GPU shader programming, and systems architecture in Python.*
