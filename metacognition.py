"""
metacognition.py — GeNeSIS V — The Self-Referential Engine
==============================================================

Carries forward GeNeSIS IV's metacognitive substrate (README IV §3.1,
§3.5, §3.11, §3.12, §3.14, §3.17, §3.20, §4.1) unchanged in its math,
faithfully re-implemented here as the foundation V's consciousness.py
and biology.py build on.

Contents
--------
GodelEncoder          — encode/decode behavioural programs as Gödel integers
CivilizationMemory     — sparse autoassociative Hermitian memory (per-tribe / global)
NoveltyScorer          — geometric-mean novelty index, 5σ breakthrough, Cambrian explosion
PhylogeneticTracker    — greedy clade clustering by meta-eigenspectrum distance
MetaConsciousness      — dual-band meta-Hamiltonian, meta-modulated learning-rate profile

Design choices made where the spec leaves a free parameter (documented
inline, marked "# design choice:") are chosen to be the simplest
faithful reading of the written formula, not an arbitrary embellishment.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------
# Core dimensional constants (README IV §3.1 / §4.1)
# ----------------------------------------------------------------------------
K_DIM: int = 64          # Total Hilbert-space dimension
K_TASK: int = 24         # Task-band dimensions
K_META: int = 40         # Meta-cognitive band dimensions
assert K_TASK + K_META == K_DIM

TASK_DT: float = 0.025   # Task-band Schrödinger time step   (§3.3)
META_DT: float = 0.005   # Meta-band  Schrödinger time step   (§3.3)

GODEL_BASE: int = 19          # prime > N_PRIMITIVES            (§3.11)
MAX_PROGRAM_LEN: int = 16
N_PRIMITIVES: int = 16

EPS: float = 1e-12

# ----------------------------------------------------------------------------
# The 16-primitive behavioural algebra (§3.11)
# ----------------------------------------------------------------------------
PRIMITIVE_CATEGORIES: Dict[str, List[str]] = {
    "spatial":   ["move", "jump", "spiral", "retreat"],
    "metabolic": ["eat", "fast", "store", "burn"],
    "social":    ["signal", "mimic", "teach", "isolate"],
    "cognitive": ["reflect", "dream", "focus", "diffuse"],
}
ALL_PRIMITIVES: List[str] = [p for cat in PRIMITIVE_CATEGORIES.values() for p in cat]
assert len(ALL_PRIMITIVES) == N_PRIMITIVES

PRIMITIVE_TO_CATEGORY: Dict[str, str] = {
    p: cat for cat, plist in PRIMITIVE_CATEGORIES.items() for p in plist
}
PRIMITIVE_TO_INDEX: Dict[str, int] = {p: i for i, p in enumerate(ALL_PRIMITIVES)}


def first_n_primes(n: int) -> np.ndarray:
    """Sieve the first n primes — seeds each agent's immutable ω 'soul' (§3.1)."""
    primes: List[int] = []
    candidate = 2
    while len(primes) < n:
        is_p = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(candidate)
        candidate += 1
    return np.array(primes, dtype=np.float64)


PRIME_TABLE: np.ndarray = first_n_primes(K_DIM)  # cached, first 64 primes


def make_soul(rng: np.random.Generator) -> np.ndarray:
    """
    omega = (p1/100, ..., p64/100), modulated at birth by a random amplitude
    drawn from an exponential distribution (§3.1).

    # design choice: the spec calls for phase/amplitude modulation of omega,
    # but omega must stay REAL because it seeds the diagonal of a Hermitian
    # matrix. The random *phase* draw is therefore expressed instead in the
    # birth of psi (a genuinely complex object) in consciousness.py; here,
    # omega only receives the amplitude modulation, re-centred near 1.0 so
    # the exponential draw's long tail can't blow up the spectrum.
    """
    base = PRIME_TABLE / 100.0
    amplitude = rng.exponential(scale=1.0, size=K_DIM)
    amplitude = 0.5 + 0.5 * (amplitude / (amplitude.mean() + EPS))
    return base * amplitude


# ----------------------------------------------------------------------------
# GodelEncoder  (§3.11)
# ----------------------------------------------------------------------------
class GodelEncoder:
    """Encodes/decodes behavioural programs as unique Gödel integers."""

    base: int = GODEL_BASE

    @classmethod
    def encode(cls, program: Sequence[str]) -> int:
        if not (2 <= len(program) <= MAX_PROGRAM_LEN):
            raise ValueError(
                f"program length must be in [2, {MAX_PROGRAM_LEN}], got {len(program)}"
            )
        g = 0
        for k, prim in enumerate(program):
            idx = PRIMITIVE_TO_INDEX[prim]
            g += (idx + 1) * (cls.base ** k)
        return g

    @classmethod
    def decode(cls, g: int, length: int) -> List[str]:
        program = []
        for k in range(length):
            idx = (g % (cls.base ** (k + 1))) // (cls.base ** k) - 1
            idx = int(max(0, min(N_PRIMITIVES - 1, idx)))
            program.append(ALL_PRIMITIVES[idx])
        return program

    @staticmethod
    def distance(g1: int, g2: int) -> float:
        return abs(math.log1p(g1) - math.log1p(g2))

    @staticmethod
    def diversity(program: Sequence[str]) -> float:
        cats = {PRIMITIVE_TO_CATEGORY[p] for p in program}
        return len(cats) / 4.0


# ----------------------------------------------------------------------------
# CivilizationMemory  (§3.14, §3.20)
# ----------------------------------------------------------------------------
class CivilizationMemory:
    """
    Sparse autoassociative Hermitian memory. One instance per tribe, plus one
    global instance, per the spec's "two instances exist simultaneously".
    """

    def __init__(self, dim: int = K_DIM, decay: float = 0.95, eta: float = 0.05):
        self.dim = dim
        self.decay = decay
        self.eta = eta
        self.M = np.zeros((dim, dim), dtype=np.complex128)

    def store(self, v: np.ndarray) -> None:
        v = np.asarray(v, dtype=np.complex128)
        norm = np.linalg.norm(v)
        if norm < EPS:
            return
        v_hat = v / norm
        outer = np.outer(v_hat, v_hat.conj())  # Hermitian by construction
        self.M = self.decay * self.M + self.eta * outer

    def query(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.complex128)
        norm = np.linalg.norm(q)
        if norm < EPS:
            return 0.0
        q_hat = q / norm
        return float(np.linalg.norm(self.M @ q_hat) ** 2)

    def spectral_summary(self, k: int = 16) -> np.ndarray:
        eigvals = np.linalg.eigvalsh(self.M)
        return eigvals[-k:]


# ----------------------------------------------------------------------------
# NoveltyScorer  (§3.12)
# ----------------------------------------------------------------------------
class _WelfordStats:
    """Online mean/variance via Welford's algorithm."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(max(self.m2 / (self.n - 1), 0.0))


class NoveltyScorer:
    """Geometric-mean Novelty Index, 5-sigma breakthrough, Cambrian explosion."""

    def __init__(self, recent_window: int = 30):
        self.known_godel: List[int] = []
        self.stats = _WelfordStats()
        self.history: List[float] = []
        self.recent_window = recent_window

    def score(
        self, program: Sequence[str], godel_number: int, resonance_rho: float
    ) -> Tuple[float, bool]:
        if self.known_godel:
            d_min = min(GodelEncoder.distance(godel_number, g) for g in self.known_godel)
        else:
            # design choice: an empty corpus has no prior to be novel *against*,
            # so the first invention gets a neutral (not maximal) baseline.
            d_min = 1.0
        div = GodelEncoder.diversity(program)
        n_index = (
            max(d_min, EPS) * max(div, EPS) * (1.0 / (1.0 + resonance_rho))
        ) ** (1.0 / 3.0)

        self.known_godel.append(godel_number)
        self.stats.update(n_index)
        self.history.append(n_index)

        breakthrough = False
        if self.stats.n > 10:
            threshold = self.stats.mean + 5.0 * self.stats.std
            breakthrough = n_index > threshold
        return n_index, breakthrough

    def cambrian_explosion(self) -> bool:
        w = self.recent_window
        if len(self.history) < 2 * w:
            return False
        recent = np.array(self.history[-w:])
        older = np.array(self.history[-2 * w : -w])
        cond_std = recent.std(ddof=1) > 1.5 * (older.std(ddof=1) + EPS)
        cond_mean = recent.mean() > 1.25 * (older.mean() + EPS)
        return bool(cond_std or cond_mean)


# ----------------------------------------------------------------------------
# PhylogeneticTracker  (§4.1)
# ----------------------------------------------------------------------------
class PhylogeneticTracker:
    """
    Greedy clade clustering by meta-eigenspectrum distance, with
    punctuated-equilibrium detection when clades proliferate quickly.

    Fidelity note: split_threshold defaults to 2.5, the value README IV
    §4.6 fixes for evolution.py's usage of this tracker — a placeholder
    of 6.0 stood here until that section of the spec was located.
    """

    def __init__(self, split_threshold: float = 2.5):
        self.split_threshold = split_threshold
        self.clade_reps: List[np.ndarray] = []
        self.clade_births: List[int] = []
        self.clade_of: Dict[int, int] = {}

    def assign(self, agent_id: int, meta_eigvals: np.ndarray, tick: int) -> int:
        if not self.clade_reps:
            self.clade_reps.append(meta_eigvals.copy())
            self.clade_births.append(tick)
            self.clade_of[agent_id] = 0
            return 0

        dists = [float(np.linalg.norm(meta_eigvals - rep)) for rep in self.clade_reps]
        best = int(np.argmin(dists))
        if dists[best] > self.split_threshold:
            new_idx = len(self.clade_reps)
            self.clade_reps.append(meta_eigvals.copy())
            self.clade_births.append(tick)
            self.clade_of[agent_id] = new_idx
            return new_idx

        self.clade_reps[best] = 0.95 * self.clade_reps[best] + 0.05 * meta_eigvals
        self.clade_of[agent_id] = best
        return best

    def punctuated_equilibrium(
        self, tick: int, window: int = 50, min_new_clades: int = 3
    ) -> bool:
        recent_births = sum(1 for b in self.clade_births if tick - b <= window)
        return recent_births >= min_new_clades

    @property
    def n_clades(self) -> int:
        return len(self.clade_reps)


# ----------------------------------------------------------------------------
# MetaConsciousness  (§3.2, §3.5, §4.1)
# ----------------------------------------------------------------------------
class MetaConsciousness:
    """
    The meta-band of an agent's cognition: H_meta in M_40(C), modelling
    *how the agent learns* rather than what it perceives.
    """

    def __init__(self, omega: np.ndarray, rng: np.random.Generator):
        assert omega.shape[0] >= K_META
        B = rng.normal(0, 0.08, (K_META, K_META)) + 1j * rng.normal(
            0, 0.08, (K_META, K_META)
        )
        H = np.diag(0.3 * omega[:K_META] + 0.5).astype(np.complex128) + 0.5 * (
            B + B.conj().T
        )
        self.H = 0.5 * (H + H.conj().T)  # enforce exact Hermiticity against fp drift

        psi0 = rng.normal(0, 1, K_META) + 1j * rng.normal(0, 1, K_META)
        self.psi = psi0 / np.linalg.norm(psi0)

        self._V: Optional[np.ndarray] = None
        self._lam: Optional[np.ndarray] = None
        self._dirty = True

        self.exploration_counts = np.zeros(K_META, dtype=np.int64)

    def _eig(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._dirty or self._V is None:
            self._lam, self._V = np.linalg.eigh(self.H)
            self._dirty = False
        return self._V, self._lam

    def evolve_meta(self, dt: float = META_DT) -> None:
        V, lam = self._eig()
        phase = np.exp(-1j * lam * dt)
        self.psi = V @ (phase * (V.conj().T @ self.psi))
        n = np.linalg.norm(self.psi)
        if n > EPS:
            self.psi = self.psi / n

    def learning_rate_profile(self, full_dim: int = K_DIM) -> np.ndarray:
        """mu in [0.2, 3.0]^40, interpolated up to R^{full_dim} (§3.5)."""
        mag = np.abs(self.psi)
        mu = 0.2 + 2.8 * mag / (mag.max() + EPS)
        x_old = np.linspace(0.0, 1.0, K_META)
        x_new = np.linspace(0.0, 1.0, full_dim)
        return np.interp(x_new, x_old, mu)

    def attempt_meta_invention(self) -> float:
        """Perturb the least-explored meta-eigenmode; return cognitive surprise."""
        V, _ = self._eig()
        k_dark = int(np.argmin(self.exploration_counts))
        self.exploration_counts[k_dark] += 1
        v_dark = V[:, k_dark]

        psi_before = self.psi.copy()
        outer = np.outer(v_dark, v_dark.conj())
        dH = 0.02 * 0.5 * (outer + outer.conj().T)
        self.H = self.H + dH
        self.H = 0.5 * (self.H + self.H.conj().T)
        self._dirty = True

        self.evolve_meta()
        return float(np.linalg.norm(self.psi - psi_before))

    def eigvals(self) -> np.ndarray:
        _, lam = self._eig()
        return lam


if __name__ == "__main__":
    # Minimal self-test; the full smoke-test suite lives in test_smoke.py
    rng = np.random.default_rng(0)
    omega = make_soul(rng)
    print("omega[:5] =", omega[:5])

    prog = ["move", "eat", "signal", "reflect"]
    g = GodelEncoder.encode(prog)
    back = GodelEncoder.decode(g, len(prog))
    assert back == prog, (prog, back)
    print("Gödel round-trip OK:", g, "->", back)

    mc = MetaConsciousness(omega, rng)
    mc.evolve_meta()
    surprise = mc.attempt_meta_invention()
    print("meta invention surprise =", surprise)
    print("metacognition.py self-test passed.")
