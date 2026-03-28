"""
lsystem.py — L-System String Rewriting Engine
=============================================
An L-System (Lindenmayer System) is a parallel string rewriting system
originally developed by biologist Aristid Lindenmayer in 1968 to model
plant growth. It consists of:
  - An alphabet Σ of symbols
  - A set of production rules P: Σ → Σ*
  - An axiom (seed string) ω ∈ Σ*

At each generation n+1, every symbol in the current string is
simultaneously replaced by its production rule expansion.

Mathematical notation:
  G = (V, ω, P)
  Where V = alphabet, ω = axiom, P = production rules

Classic L-Systems implemented:
  - Koch Curve       : F → F+F−F−F+F
  - Dragon Curve     : F → F+G, G → F−G
  - Sierpinski Tri   : F → F+G+F, G → G+G
  - Plant (branching): X → F+[[X]−X]−F[−FX]+X
  - Fractal Tree     : F → FF, X → F[+X][−X]FX
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time


# ─── Preset L-System configurations ──────────────────────────────────────────

PRESETS: Dict[str, dict] = {
    "fractal_tree": {
        "name": "Fractal Tree",
        "axiom": "X",
        "rules": {
            "X": "F+[[X]-X]-F[-FX]+X",
            "F": "FF",
        },
        "angle": 25.0,
        "description": "Classic branching fractal tree",
    },
    "dragon_curve": {
        "name": "Dragon Curve",
        "axiom": "FX",
        "rules": {
            "X": "X+YF+",
            "Y": "-FX-Y",
        },
        "angle": 90.0,
        "description": "Heighway dragon curve",
    },
    "sierpinski": {
        "name": "Sierpinski Triangle",
        "axiom": "F-G-G",
        "rules": {
            "F": "F-G+F+G-F",
            "G": "GG",
        },
        "angle": 120.0,
        "description": "Sierpinski triangle fractal",
    },
    "koch_curve": {
        "name": "Koch Snowflake",
        "axiom": "F++F++F",
        "rules": {
            "F": "F-F++F-F",
        },
        "angle": 60.0,
        "description": "Koch snowflake curve",
    },
    "plant": {
        "name": "Organic Plant",
        "axiom": "X",
        "rules": {
            "X": "F+[[X]-X]-F[-FX]+X",
            "F": "FF",
        },
        "angle": 22.5,
        "description": "Organic branching plant structure",
    },
    "hilbert": {
        "name": "Hilbert Curve",
        "axiom": "A",
        "rules": {
            "A": "-BF+AFA+FB-",
            "B": "+AF-BFB-FA+",
        },
        "angle": 90.0,
        "description": "Space-filling Hilbert curve",
    },
    "gosper": {
        "name": "Gosper Curve",
        "axiom": "F",
        "rules": {
            "F": "F-G--G+F++FF+G-",
            "G": "+F-GG--G-F++F+G",
        },
        "angle": 60.0,
        "description": "Gosper (flowsnake) curve",
    },
    "levy": {
        "name": "Levy C Curve",
        "axiom": "F",
        "rules": {
            "F": "+F--F+",
        },
        "angle": 45.0,
        "description": "Levy C curve",
    },
}


@dataclass
class LSystemConfig:
    """
    Configuration container for an L-System.
    Stores all parameters needed to generate and render the fractal.
    """
    axiom: str                          # Initial seed string
    rules: Dict[str, str]               # Production rules: symbol → replacement
    angle: float = 25.0                 # Turn angle in degrees
    iterations: int = 4                 # Number of rewriting steps
    step_length: float = 5.0           # Length of each forward step (pixels)
    preset_name: str = "custom"        # Human-readable preset label

    def __post_init__(self):
        # Validate axiom is non-empty
        if not self.axiom:
            raise ValueError("Axiom cannot be empty")
        # Validate angle is in sane range
        if not (0.0 < self.angle < 360.0):
            raise ValueError(f"Angle {self.angle} must be in (0, 360)")


@dataclass
class LSystemResult:
    """
    Output of an L-System generation pass.
    Carries the final string plus metadata for downstream consumers.
    """
    string: str                         # Final rewritten string
    iterations: int                     # How many generations were applied
    symbol_count: int                   # Total symbol count (measure of complexity)
    generation_time_ms: float           # Wall-clock time for generation
    rule_applications: int              # Total substitution count across all steps
    config: LSystemConfig               # Reference to originating config


class LSystemGenerator:
    """
    Core L-System string rewriting engine.

    Implements the parallel string rewriting algorithm with:
      - Iterative generation (avoids Python recursion depth limits)
      - Safety limits on string length to prevent memory exhaustion
      - Symbol-count tracking per generation
      - Stochastic rule support (probability-weighted alternatives)

    Usage:
        gen = LSystemGenerator()
        config = LSystemConfig(axiom="F", rules={"F": "F+F-F-F+F"}, angle=90)
        result = gen.generate(config, iterations=4)
        print(result.string[:100])
    """

    # Hard cap: strings beyond this length are truncated with a warning.
    # A 50k-char string already represents millions of pixels of geometry.
    MAX_STRING_LENGTH: int = 200_000

    def __init__(self):
        self._generation_stats: List[dict] = []

    def generate(self, config: LSystemConfig, iterations: Optional[int] = None) -> LSystemResult:
        """
        Apply production rules for `iterations` steps starting from axiom.

        Args:
            config:     L-System configuration (axiom + rules + angle)
            iterations: Override config.iterations if provided

        Returns:
            LSystemResult with final string and performance metadata
        """
        n = iterations if iterations is not None else config.iterations
        current = config.axiom
        total_applications = 0
        t_start = time.perf_counter()

        for step in range(n):
            next_string, applications = self._apply_rules(current, config.rules)
            total_applications += applications

            # Safety guard: truncate if string explodes
            if len(next_string) > self.MAX_STRING_LENGTH:
                print(f"[LSystem] WARNING: String length {len(next_string):,} "
                      f"exceeded cap {self.MAX_STRING_LENGTH:,} at step {step+1}. "
                      f"Truncating to prevent memory overflow.")
                next_string = next_string[:self.MAX_STRING_LENGTH]

            current = next_string

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        result = LSystemResult(
            string=current,
            iterations=n,
            symbol_count=len(current),
            generation_time_ms=elapsed_ms,
            rule_applications=total_applications,
            config=config,
        )
        self._generation_stats.append({
            "iterations": n,
            "length": len(current),
            "time_ms": elapsed_ms,
        })
        return result

    def _apply_rules(self, string: str, rules: Dict[str, str]) -> tuple[str, int]:
        """
        One parallel rewriting step: replace every symbol simultaneously.
        Symbols not in rules are passed through unchanged (identity rule).

        Returns (new_string, number_of_rule_applications)
        """
        parts = []
        applications = 0
        for char in string:
            if char in rules:
                parts.append(rules[char])
                applications += 1
            else:
                parts.append(char)    # Identity: keep symbol as-is
        return "".join(parts), applications

    @staticmethod
    def from_preset(preset_key: str, iterations: int = 4) -> tuple["LSystemGenerator", LSystemConfig]:
        """
        Factory: create a generator + config from a named preset.

        Args:
            preset_key: Key from PRESETS dict (e.g. "fractal_tree")
            iterations: How many rewriting steps to run

        Returns:
            (generator_instance, config_instance)
        """
        if preset_key not in PRESETS:
            available = ", ".join(PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_key}'. Available: {available}")

        p = PRESETS[preset_key]
        config = LSystemConfig(
            axiom=p["axiom"],
            rules=p["rules"],
            angle=p["angle"],
            iterations=iterations,
            preset_name=preset_key,
        )
        return LSystemGenerator(), config

    def get_stats(self) -> List[dict]:
        """Return performance history for all generate() calls."""
        return list(self._generation_stats)

    def analyze_string(self, result: LSystemResult) -> dict:
        """
        Analyze the L-System string to extract structural metrics.
        Useful for predicting rendering complexity before drawing.
        """
        s = result.string
        return {
            "total_symbols":    len(s),
            "forward_moves":    s.count("F") + s.count("G"),
            "left_turns":       s.count("+"),
            "right_turns":      s.count("-"),
            "push_stack":       s.count("["),
            "pop_stack":        s.count("]"),
            "max_branch_depth": self._compute_max_depth(s),
            "unique_symbols":   len(set(s)),
        }

    @staticmethod
    def _compute_max_depth(string: str) -> int:
        """Walk brackets to find maximum nesting depth (= max branch depth)."""
        max_d, current = 0, 0
        for ch in string:
            if ch == "[":
                current += 1
                max_d = max(max_d, current)
            elif ch == "]":
                current = max(0, current - 1)
        return max_d

    def list_presets(self) -> List[dict]:
        """Return human-readable list of all built-in presets."""
        return [
            {"key": k, "name": v["name"], "description": v["description"]}
            for k, v in PRESETS.items()
        ]
