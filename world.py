"""
world.py — GeNeSIS V — Hyper-Horizon World Physics
======================================================

The physical substrate every agent senses and acts on. Carries forward
GeNeSIS IV's world feature set (README IV §4.4, v3.0 features 16-24)
plus V's memory-discipline additions from the master plan:

    - §2.1 Procedural-over-stored terrain: a deterministic value-noise
      field evaluated at any (x, y), never fully materialised.
    - §3.1 Physics substrate: a real 2D heat equation drives a local
      temperature field, coupling into resource regeneration.

Deliberate deviation from the IV spec, stated plainly: IV specifies a
PyTorch PhysicsOracle. V implements the identical *role* — a frozen,
never-trained function defining a "discoverable law" agents can try to
reverse-engineer — as a small NumPy MLP with fixed random weights
instead. Importing torch costs several hundred MB of baseline RAM
(masterplan §2, RAM budget table) for a 3-layer network with no
training loop and no autograd use; NumPy gives the same frozen
function for effectively free. torch is dropped from requirements.txt
for this reason — nothing else in the codebase needs it either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from biology import laplacian

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_WORLD_SIZE: int = 72  # toroidal H x W

RESOURCE_NAMES: List[str] = ["water", "food", "minerals", "energy"]
N_RESOURCE_TYPES: int = len(RESOURCE_NAMES)

# design choice: the spec calls for a 16-channel pheromone grid without
# naming the channels; these 16 are a reasonable, documented convention.
PHEROMONE_NAMES: List[str] = [
    "trail", "alarm", "food", "water", "mate", "territory", "help", "danger",
    "gathering", "migration", "nest", "hunt", "rest", "play", "mourning", "celebration",
]
N_PHEROMONE_CHANNELS: int = len(PHEROMONE_NAMES)
assert N_PHEROMONE_CHANNELS == 16

# design choice: the spec names three example meme categories
# ("Danger/Resource/Sacred/...") for an 8-channel grid; the remaining
# five are a documented extension of that theme.
MEME_NAMES: List[str] = [
    "danger", "resource", "sacred", "territory",
    "curiosity", "authority", "kinship", "mystery",
]
N_MEME_CHANNELS: int = len(MEME_NAMES)
assert N_MEME_CHANNELS == 8

SEASON_PERIOD: int = 200          # ticks per full summer/winter cycle (feature 19)
HEAT_DIFFUSIVITY: float = 0.12    # alpha in the heat equation (§3.1)
PHEROMONE_DECAY: float = 0.97
PHEROMONE_DIFFUSE: float = 0.10
MEME_DECAY: float = 0.995         # memes persist far longer than pheromones — culture vs. scent
MEME_DIFFUSE: float = 0.05

EPS: float = 1e-12


# ----------------------------------------------------------------------------
# Procedural, infinite terrain  (masterplan §2.1)
# ----------------------------------------------------------------------------
def _hash2d(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Deterministic integer-lattice hash -> [0, 1). Pure NumPy, no external
    noise library dependency. Overflow in the uint32 arithmetic below is
    intentional: it is exactly what scrambles the bits."""
    ix = ix.astype(np.uint32)
    iy = iy.astype(np.uint32)
    s = np.uint32(seed & 0xFFFFFFFF)
    with np.errstate(over="ignore"):
        # uint32 wraparound on overflow is the hash, not a bug — silenced deliberately.
        h = ix * np.uint32(374761393) + iy * np.uint32(668265263) + s * np.uint32(2246822519)
        h = (h ^ (h >> np.uint32(13))) * np.uint32(1274126177)
        h = h ^ (h >> np.uint32(16))
    return h.astype(np.float64) / np.float64(0xFFFFFFFF)


def _fade(t: np.ndarray) -> np.ndarray:
    """Perlin's smoothstep: 6t^5 - 15t^4 + 10t^3 — zero first and second
    derivative at the lattice points, so tiles join with no visible seams."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise_2d(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Bilinear-interpolated value noise at arbitrary (possibly non-integer,
    possibly out-of-range) coordinates x, y. Same (x, y, seed) always
    returns the same value: this is what makes terrain evaluable on demand
    rather than stored."""
    x0 = np.floor(x)
    y0 = np.floor(y)
    fx = x - x0
    fy = y - y0

    h00 = _hash2d(x0, y0, seed)
    h10 = _hash2d(x0 + 1, y0, seed)
    h01 = _hash2d(x0, y0 + 1, seed)
    h11 = _hash2d(x0 + 1, y0 + 1, seed)

    ux = _fade(fx)
    uy = _fade(fy)

    top = h00 * (1 - ux) + h10 * ux
    bot = h01 * (1 - ux) + h11 * ux
    return top * (1 - uy) + bot * uy


def fractal_noise_2d(
    x: np.ndarray, y: np.ndarray, seed: int,
    octaves: int = 4, persistence: float = 0.5, lacunarity: float = 2.0,
    scale: float = 0.08,
) -> np.ndarray:
    """Multi-octave (fractal) value noise, normalised to [0, 1]."""
    total = np.zeros_like(x, dtype=np.float64)
    amplitude = 1.0
    frequency = scale
    max_amp = 0.0
    for o in range(octaves):
        total += amplitude * value_noise_2d(x * frequency, y * frequency, seed + o * 101)
        max_amp += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max_amp


# ----------------------------------------------------------------------------
# Frozen physics oracle  (§4.4; NumPy replacement for the spec's torch NN)
# ----------------------------------------------------------------------------
class PhysicsOracle:
    """
    A frozen (never-trained) 3-layer NumPy MLP. It defines a fixed,
    deterministic "law of physics" — agents/civilization.py try to
    reverse-engineer this function from observation (Level 9, Physics
    Discovery), exactly as the original torch version was meant to be
    used; only the implementation substrate changed.
    """

    def __init__(self, in_dim: int = 8, hidden: int = 16, out_dim: int = 4, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.6, (in_dim, hidden))
        self.b1 = rng.normal(0, 0.1, hidden)
        self.W2 = rng.normal(0, 0.6, (hidden, hidden))
        self.b2 = rng.normal(0, 0.1, hidden)
        self.W3 = rng.normal(0, 0.6, (hidden, out_dim))
        self.b3 = rng.normal(0, 0.1, out_dim)

    def __call__(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        h1 = np.tanh(x @ self.W1 + self.b1)
        h2 = np.tanh(h1 @ self.W2 + self.b2)
        out = np.tanh(h2 @ self.W3 + self.b3)
        return out


# ----------------------------------------------------------------------------
# World structures  (feature 22)
# ----------------------------------------------------------------------------
@dataclass
class Structure:
    x: int
    y: int
    kind: str = "generic"

    def tick_effect(self, world: "GenesisWorld") -> None:
        pass


@dataclass
class Trap(Structure):
    kind: str = "trap"
    damage: float = 0.3

    def apply_to_agent(self, agent_energy: float) -> float:
        return max(0.0, agent_energy - self.damage)


@dataclass
class Battery(Structure):
    kind: str = "battery"
    stored: float = 0.0
    capacity: float = 10.0

    def deposit(self, amount: float) -> float:
        accepted = min(amount, self.capacity - self.stored)
        self.stored += accepted
        return accepted

    def withdraw(self, amount: float) -> float:
        given = min(amount, self.stored)
        self.stored -= given
        return given


@dataclass
class Cultivator(Structure):
    kind: str = "cultivator"
    radius: int = 3
    boost: float = 1.5

    def tick_effect(self, world: "GenesisWorld") -> None:
        world.boost_resource_region(self.x, self.y, self.radius, self.boost)


@dataclass
class MegaResource:
    """Cooperative harvest node — requires >=2 co-located agents (feature 23)."""

    x: int
    y: int
    resource_type: int
    amount: float
    required_agents: int = 2

    def harvest(self, n_agents_present: int) -> float:
        if n_agents_present < self.required_agents:
            return 0.0
        yield_amount = self.amount
        self.amount = 0.0
        return yield_amount

    @property
    def depleted(self) -> bool:
        return self.amount <= EPS


# ----------------------------------------------------------------------------
# GenesisWorld
# ----------------------------------------------------------------------------
class GenesisWorld:
    def __init__(self, height: int = DEFAULT_WORLD_SIZE, width: int = DEFAULT_WORLD_SIZE, seed: int = 42):
        self.height = height
        self.width = width
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tick = 0

        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        self._xx = xx.astype(np.float64)
        self._yy = yy.astype(np.float64)

        fertility = fractal_noise_2d(self._xx, self._yy, seed=seed, octaves=4)
        self.fertility = fertility  # static procedural terrain field, [0, 1]

        self.resource_grid = np.zeros((height, width, N_RESOURCE_TYPES), dtype=np.float32)
        for r in range(N_RESOURCE_TYPES):
            base = fractal_noise_2d(self._xx, self._yy, seed=seed + 1000 * (r + 1), octaves=3)
            self.resource_grid[:, :, r] = base * fertility

        self.pheromone_grid = np.zeros((height, width, N_PHEROMONE_CHANNELS), dtype=np.float32)
        self.meme_grid = np.zeros((height, width, N_MEME_CHANNELS), dtype=np.float32)

        # heat field: latitude-like gradient (warmer toward the vertical
        # centre) plus a touch of noise, per §3.1
        lat = 1.0 - np.abs((yy - height / 2.0) / (height / 2.0))
        self.heat_field = (0.4 + 0.6 * lat + 0.05 * fractal_noise_2d(
            self._xx, self._yy, seed=seed + 99, octaves=2
        )).astype(np.float64)

        self.oracle = PhysicsOracle(seed=seed + 7)

        self.structures: List[Structure] = []
        self.mega_resources: List[MegaResource] = []

        self.weather_amplitude: float = 1.0
        self._weather_votes: List[float] = []

        self.entropy_ledger: float = 0.0  # cumulative dissipated energy (§3.1)
        self.population_density: float = 0.0  # set externally by evolution.py each tick

    # ------------------------------------------------------------------
    # Toroidal helpers
    # ------------------------------------------------------------------
    def wrap(self, x: int, y: int) -> Tuple[int, int]:
        return int(x) % self.width, int(y) % self.height

    # ------------------------------------------------------------------
    # Season / weather  (features 19, 20)
    # ------------------------------------------------------------------
    @property
    def season_phase(self) -> float:
        """In [-1, 1]; >0 is summer, <0 is winter."""
        return float(np.sin(2 * np.pi * self.tick / SEASON_PERIOD))

    @property
    def season_name(self) -> str:
        return "summer" if self.season_phase >= 0 else "winter"

    def vote_weather(self, delta: float) -> None:
        """Collective agent vote nudges weather amplitude (feature 20)."""
        self._weather_votes.append(delta)

    def _apply_weather_votes(self) -> None:
        if self._weather_votes:
            shift = float(np.mean(self._weather_votes))
            self.weather_amplitude = float(np.clip(self.weather_amplitude + 0.05 * shift, 0.2, 2.0))
            self._weather_votes.clear()

    # ------------------------------------------------------------------
    # Heat diffusion  (§3.1: real 2D heat equation)
    # ------------------------------------------------------------------
    def _step_heat(self) -> None:
        forcing = 0.03 * self.season_phase  # summer warms, winter cools
        self.heat_field = self.heat_field + HEAT_DIFFUSIVITY * laplacian(self.heat_field) + forcing
        self.heat_field = np.clip(self.heat_field, 0.0, 2.0)

    # ------------------------------------------------------------------
    # Resource regeneration, coupled to heat + season + adaptive density  (features 19, 24)
    # ------------------------------------------------------------------
    def _step_resources(self) -> None:
        # Arrhenius-flavoured regen: warmer cells regenerate faster, up to a point.
        # design choice: a simple monotone reaction-rate proxy (real Arrhenius
        # kinetics with an explicit activation energy belongs to chemistry.py,
        # once it exists) — this keeps world.py's physics<->biology coupling
        # honest without pre-empting that module.
        rate_multiplier = 0.5 + 0.5 * self.heat_field  # in [0.5, 1.5]-ish
        season_multiplier = 1.0 + 0.3 * self.season_phase  # richer in summer

        # feature 24: adaptive spawn, inversely proportional to population density
        density_damping = 1.0 / (1.0 + self.population_density)

        growth = (
            0.04
            * self.weather_amplitude
            * density_damping
            * rate_multiplier[..., None]
            * season_multiplier
            * self.fertility[..., None]
        )
        self.resource_grid = np.clip(self.resource_grid + growth.astype(np.float32), 0.0, 5.0)

        for s in self.structures:
            s.tick_effect(self)

    def boost_resource_region(self, x: int, y: int, radius: int, factor: float) -> None:
        """Used by Cultivator structures (feature 22)."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy > radius * radius:
                    continue
                gx, gy = self.wrap(x + dx, y + dy)
                self.resource_grid[gy, gx, :] = np.clip(
                    self.resource_grid[gy, gx, :] * factor, 0.0, 5.0
                )

    # ------------------------------------------------------------------
    # Pheromone / meme decay + diffusion  (features 17, 18)
    # ------------------------------------------------------------------
    def _step_stigmergy(self) -> None:
        for c in range(N_PHEROMONE_CHANNELS):
            field = self.pheromone_grid[:, :, c].astype(np.float64)
            field = field * PHEROMONE_DECAY + PHEROMONE_DIFFUSE * laplacian(field)
            self.pheromone_grid[:, :, c] = np.clip(field, 0.0, 10.0)

        for c in range(N_MEME_CHANNELS):
            field = self.meme_grid[:, :, c].astype(np.float64)
            field = field * MEME_DECAY + MEME_DIFFUSE * laplacian(field)
            self.meme_grid[:, :, c] = np.clip(field, 0.0, 10.0)

    def deposit_pheromone(self, x: int, y: int, channel: int, amount: float) -> None:
        gx, gy = self.wrap(x, y)
        self.pheromone_grid[gy, gx, channel] = min(10.0, self.pheromone_grid[gy, gx, channel] + amount)

    def deposit_meme(self, x: int, y: int, channel: int, amount: float) -> None:
        gx, gy = self.wrap(x, y)
        self.meme_grid[gy, gx, channel] = min(10.0, self.meme_grid[gy, gx, channel] + amount)

    # ------------------------------------------------------------------
    # Sensing  (feeds HarmonicResonanceConsciousness.decide())
    # ------------------------------------------------------------------
    def sense(self, x: int, y: int, radius: int = 2) -> np.ndarray:
        """Local observation vector: resources, pheromones, memes, heat —
        concatenated and flattened, ready to hand to HRC.decide()."""
        gx, gy = self.wrap(x, y)
        ys = [self.wrap(gx, gy + dy)[1] for dy in range(-radius, radius + 1)]
        xs = [self.wrap(gx + dx, gy)[0] for dx in range(-radius, radius + 1)]

        local_res = self.resource_grid[np.ix_(ys, xs)].mean(axis=(0, 1))
        local_pher = self.pheromone_grid[np.ix_(ys, xs)].mean(axis=(0, 1))
        local_meme = self.meme_grid[np.ix_(ys, xs)].mean(axis=(0, 1))
        local_heat = np.array([self.heat_field[gy, gx]])

        return np.concatenate([local_res, local_pher, local_meme, local_heat]).astype(np.float64)

    def oracle_features(self, x: int, y: int) -> np.ndarray:
        gx, gy = self.wrap(x, y)
        res = self.resource_grid[gy, gx, :]
        return np.concatenate(
            [res, [self.heat_field[gy, gx]], [self.season_phase], [self.weather_amplitude], [0.0]]
        )[: self.oracle.W1.shape[0]]

    def query_oracle(self, x: int, y: int) -> np.ndarray:
        return self.oracle(self.oracle_features(x, y))

    # ------------------------------------------------------------------
    # Mega-resources  (feature 23)
    # ------------------------------------------------------------------
    def try_harvest_mega(self, x: int, y: int, n_agents_present: int) -> float:
        total = 0.0
        gx, gy = self.wrap(x, y)
        for mr in self.mega_resources:
            if mr.x == gx and mr.y == gy and not mr.depleted:
                total += mr.harvest(n_agents_present)
        self.mega_resources = [mr for mr in self.mega_resources if not mr.depleted]
        return total

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def step(self) -> None:
        self._apply_weather_votes()
        self._step_heat()
        self._step_resources()
        self._step_stigmergy()
        self.tick += 1


if __name__ == "__main__":
    w = GenesisWorld(height=48, width=48, seed=123)
    w.structures.append(Cultivator(x=10, y=10))
    w.structures.append(Trap(x=20, y=20))
    battery = Battery(x=5, y=5)
    w.structures.append(battery)
    w.mega_resources.append(MegaResource(x=15, y=15, resource_type=1, amount=3.0, required_agents=2))

    for _ in range(150):
        w.vote_weather(float(np.random.default_rng(w.tick).normal(0, 0.3)))
        w.step()

    assert np.isfinite(w.resource_grid).all()
    assert np.isfinite(w.pheromone_grid).all()
    assert np.isfinite(w.meme_grid).all()
    assert np.isfinite(w.heat_field).all()
    assert w.resource_grid.min() >= 0.0
    assert 0.2 <= w.weather_amplitude <= 2.0

    obs = w.sense(24, 24, radius=2)
    print("sense() shape:", obs.shape, "sample:", obs[:6])

    oracle_out = w.query_oracle(24, 24)
    print("oracle output:", oracle_out)

    yield_amt = w.try_harvest_mega(15, 15, n_agents_present=1)
    assert yield_amt == 0.0, "should refuse harvest with only 1 agent"
    yield_amt2 = w.try_harvest_mega(15, 15, n_agents_present=2)
    assert yield_amt2 == 3.0, f"should yield full amount with 2 agents, got {yield_amt2}"
    print("mega-resource cooperative gate OK")

    w.deposit_pheromone(24, 24, 3, 5.0)
    w.deposit_meme(24, 24, 0, 5.0)
    w.step()
    print("pheromone[24,24,3] after decay:", w.pheromone_grid[24, 24, 3])
    print("meme[24,24,0] after decay:", w.meme_grid[24, 24, 0])

    # determinism check on procedural terrain
    a = fractal_noise_2d(np.array([5.0]), np.array([5.0]), seed=1)
    b = fractal_noise_2d(np.array([5.0]), np.array([5.0]), seed=1)
    c = fractal_noise_2d(np.array([5.0]), np.array([5.0]), seed=2)
    assert a == b, "noise must be deterministic for identical (x, y, seed)"
    assert a != c, "different seeds must give different fields"
    print("procedural terrain determinism OK")

    print("world.py self-test passed.")
