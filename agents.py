"""
agents.py — GeNeSIS V — BioHyperAgent
=========================================

The complete organism: wires HarmonicResonanceConsciousness (cognition),
biology.py's codon reader (genome), and GenesisWorld (environment) into
one living lifecycle, per README IV §4.3:

    Sense -> Decide -> Mode-bias -> Execute -> Learn -> Evolve -> Causal-update

20 actions, N-tick staggered execution, death/apoptosis with spectral
wisdom transfer, epigenetic inheritance on reproduction (§3.18), and
spectral resonance coupling for communication/reproduction/tribes (§3.15).

This module intentionally does NOT know about tribes, tech trees, or
diplomacy — those are civilization.py's job (not yet built). Where a
civilization-level service would normally be consulted (novelty
scoring, shared memory), BioHyperAgent accepts it as an optional
injected dependency so this file is fully testable on its own today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from metacognition import GodelEncoder, K_TASK, MAX_PROGRAM_LEN, ALL_PRIMITIVES
from consciousness import HarmonicResonanceConsciousness, EMOTION_INDEX
from biology import genome_fingerprint, genetic_distance
from world import GenesisWorld, Trap, Battery, Cultivator, MEME_NAMES

EPS = 1e-12

# ----------------------------------------------------------------------------
# The 20-action algebra  (§4.3)
# ----------------------------------------------------------------------------
MOVE_DIRECTIONS: List[Tuple[int, int]] = [
    (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1),
]  # N, NE, E, SE, S, SW, W, NW
MOVE_NAMES: List[str] = [
    "move_n", "move_ne", "move_e", "move_se", "move_s", "move_sw", "move_w", "move_nw",
]
ACTION_NAMES: List[str] = MOVE_NAMES + [
    "eat", "attack", "communicate", "reproduce", "invent", "rest",
    "build_artifact", "absorb_artifact", "meta_invent", "compose_action",
    "trade", "punish",
]
assert len(ACTION_NAMES) == 20
ACTION_INDEX: Dict[str, int] = {a: i for i, a in enumerate(ACTION_NAMES)}

# Fixed energy costs (§4.3 table). Actions not listed have no flat cost —
# their net energy effect is computed directly where they execute.
ACTION_ENERGY_COST: Dict[str, float] = {
    **{m: 0.0005 for m in MOVE_NAMES},
    "attack": 0.20,
    "invent": 0.15,
    "build_artifact": 0.05,
    "meta_invent": 0.06,
    "compose_action": 0.02,
    "trade": 0.005,
    "punish": 0.10,
    # "reproduce" uses the Malthusian formula (§3.18), computed at call time.
}

REPRODUCE_BASE_COST: float = 0.35  # C0, §3.18
INTERACTION_RADIUS: int = 3        # how far an agent can reach for social actions

ROLES: List[str] = ["Forager", "Processor", "Warrior", "Queen"]

# N-tick staggered schedule (§4.3)
KURAMOTO_EVERY: int = 5
ROLE_AND_GOL_EVERY: int = 10
MEME_ABSORPTION_EVERY: int = 20
VIRAL_BROADCAST_EVERY: int = 50

KURAMOTO_KAPPA: float = 0.5
KURAMOTO_DT: float = 0.1

VIRAL_BLEND: float = 0.03  # shared by apoptosis absorption and viral broadcast


# ----------------------------------------------------------------------------
# Spectral resonance coupling  (§3.15) — an inter-agent quantity, lives here
# rather than in consciousness.py, which only models a single agent.
# ----------------------------------------------------------------------------
def spectral_resonance(a: HarmonicResonanceConsciousness, b: HarmonicResonanceConsciousness, k_task: int = K_TASK) -> float:
    _, lam_a = a._eig()
    _, lam_b = b._eig()
    la, lb = lam_a[:k_task], lam_b[:k_task]
    na, nb = np.linalg.norm(la), np.linalg.norm(lb)
    if na < EPS or nb < EPS:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(la, lb) / (na * nb))
    return max(0.015, (cos_sim + 1.0) / 2.0)


def communicate_blend(receiver: HarmonicResonanceConsciousness, sender: HarmonicResonanceConsciousness) -> float:
    """One-directional wave-state blend during communication (§3.15).
    Returns the resonance rho used, for the caller's reward shaping."""
    rho = spectral_resonance(receiver, sender)
    sigma_sender = sender.psi * sender.omega  # soul-modulated broadcast
    blended = (1 - 0.07 * rho) * receiver.psi + 0.07 * rho * sigma_sender
    n = np.linalg.norm(blended)
    if n > EPS:
        receiver.psi = blended / n
    return rho


# ----------------------------------------------------------------------------
# Conway's Game of Life scratchpad  (feature 27)
# ----------------------------------------------------------------------------
def gol_step(grid: np.ndarray) -> np.ndarray:
    """One step of Conway's Game of Life on a toroidal 8x8 grid."""
    neighbors = sum(
        np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dy == 0 and dx == 0)
    )
    born = (neighbors == 3) & (grid == 0)
    survives = ((neighbors == 2) | (neighbors == 3)) & (grid == 1)
    return (born | survives).astype(np.int8)


def gol_seed_from_dna(dna: str) -> np.ndarray:
    """
    Seeds the 8x8 (=64-cell) Game of Life scratchpad directly from the
    agent's codon-read DNA string (biology.py) — the same K=64 substrate
    read a third way: cognition (consciousness.py), genetics (biology.py),
    and now a Turing-complete internal computation, all off one array.
    """
    bits = np.array([1 if b in ("G", "T") else 0 for b in dna], dtype=np.int8)
    return bits.reshape(8, 8)


# ----------------------------------------------------------------------------
# BioHyperAgent
# ----------------------------------------------------------------------------
class BioHyperAgent:
    def __init__(
        self,
        agent_id: int,
        x: int,
        y: int,
        seed: Optional[int] = None,
        generation: int = 0,
        hrc: Optional[HarmonicResonanceConsciousness] = None,
    ):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.generation = generation
        self.age = 0
        self.alive = True

        self.hrc = hrc if hrc is not None else HarmonicResonanceConsciousness(agent_id, seed=seed)

        dna = genome_fingerprint(self.hrc.eigenvectors())["dna"]
        self.dna = dna

        self.energy: float = 1.0
        self.health: float = 1.0
        self.role: str = "Forager"
        self.tribe_id: Optional[int] = None

        self.inventory: Dict[str, float] = {"red": 0.0, "green": 0.0, "blue": 0.0}
        self.trust: Dict[int, float] = {}
        self.causal_model: Dict[str, List[float]] = {a: [] for a in ACTION_NAMES}
        self.action_counts: Dict[str, int] = {a: 0 for a in ACTION_NAMES}  # unbounded, unlike causal_model's capped history — evolution.py's cultural ratchet needs true lifetime frequencies
        self.n_meta_inventions: int = 0  # feeds evolution.py's meta-fitness formula
        self.discoveries: List[Tuple[List[str], int, str]] = []  # (program, godel, name)

        self.theta: float = float(np.random.default_rng(seed).uniform(0, 2 * np.pi))
        self.natural_freq: float = float(np.random.default_rng(seed).normal(1.0, 0.01))
        self.circadian_phase: float = 0.0

        self.gol_grid: np.ndarray = gol_seed_from_dna(dna)

        self.tick_count = 0

    # ------------------------------------------------------------------
    # Sense
    # ------------------------------------------------------------------
    def sense(self, world: GenesisWorld) -> np.ndarray:
        return world.sense(self.x, self.y)

    # ------------------------------------------------------------------
    # Decide + mode-bias  (§4.2 emotion-effect table, "Intelligence pacifism")
    # ------------------------------------------------------------------
    def decide(self, observation: np.ndarray) -> int:
        action_idx, probs = self.hrc.decide(observation, n_actions=len(ACTION_NAMES))
        action_idx = self._apply_mode_bias(action_idx, probs)
        return action_idx

    def _apply_mode_bias(self, action_idx: int, probs: np.ndarray) -> int:
        fear = self.hrc.emotion("FEAR")
        anger = self.hrc.emotion("ANGER")
        phi_last = self.hrc.phi_history[-1] if self.hrc.phi_history else 0.0

        # SURVIVE mode: strong fear overrides toward rest (retreat-to-safety proxy)
        if fear > 0.7:
            return ACTION_INDEX["rest"]

        # Intelligence Prevents War: verified low-Phi-threshold or 2+ discoveries
        # suppress an angry DOMINATE (attack) pick, per the spec's named
        # "Intelligence pacifism" phenomenon.
        if ACTION_NAMES[action_idx] == "attack" and (phi_last > 0.005 or len(self.discoveries) >= 2):
            # redirect the energy of the impulse into invention instead of violence
            return ACTION_INDEX["invent"]

        # DOMINATE mode: high anger nudges an otherwise-uncommitted pick toward attack
        if anger > 0.7 and self.rng_choice_bias(probs) and phi_last <= 0.005:
            return ACTION_INDEX["attack"]

        return action_idx

    def rng_choice_bias(self, probs: np.ndarray) -> bool:
        # design choice: anger only overrides when the quantum decision was
        # already fairly uncertain (no single action dominating the distribution)
        return float(np.max(probs)) < 0.25

    # ------------------------------------------------------------------
    # Execute  (§4.3 action mechanics)
    # ------------------------------------------------------------------
    def execute(
        self,
        action_idx: int,
        world: GenesisWorld,
        population: List["BioHyperAgent"],
        novelty_scorer=None,
        civ_memory=None,
    ) -> float:
        name = ACTION_NAMES[action_idx]
        self.action_counts[name] += 1
        cost = ACTION_ENERGY_COST.get(name, 0.0)
        self.energy = max(0.0, self.energy - cost)
        reward = -cost

        if name in ACTION_INDEX and name.startswith("move_"):
            reward += self._do_move(name, world)
        elif name == "eat":
            reward += self._do_eat(world)
        elif name == "attack":
            reward += self._do_attack(world, population)
        elif name == "communicate":
            reward += self._do_communicate(population)
        elif name == "reproduce":
            reward += self._do_reproduce(world, population)
        elif name == "invent":
            reward += self._do_invent(novelty_scorer, civ_memory)
        elif name == "rest":
            reward += self._do_rest(world)
        elif name == "build_artifact":
            reward += self._do_build_artifact(world)
        elif name == "absorb_artifact":
            reward += self._do_absorb_artifact(world)
        elif name == "meta_invent":
            reward += self._do_meta_invent()
        elif name == "compose_action":
            reward += self._do_compose_action()
        elif name == "trade":
            reward += self._do_trade(population)
        elif name == "punish":
            reward += self._do_punish(population)

        self.causal_model[name].append(reward)
        if len(self.causal_model[name]) > 100:
            self.causal_model[name] = self.causal_model[name][-100:]

        return reward

    # -- individual action mechanics -----------------------------------
    def _do_move(self, name: str, world: GenesisWorld) -> float:
        idx = MOVE_NAMES.index(name)
        dx, dy = MOVE_DIRECTIONS[idx]
        gx, gy = world.wrap(self.x + dx, self.y + dy)
        self.x, self.y = gx, gy
        # move toward local resource/pheromone gradient is left to the agent's
        # own cognition (it chose this direction via Born-rule decide()) —
        # a small existence-cost-relative reward keeps movement non-neutral.
        local = world.resource_grid[gy, gx, :].sum()
        return 0.001 * float(local)

    def _do_eat(self, world: GenesisWorld) -> float:
        gx, gy = world.wrap(self.x, self.y)
        available = world.resource_grid[gy, gx, :].copy()
        consumed = np.minimum(available, 0.3)
        world.resource_grid[gy, gx, :] -= consumed
        gained = float(consumed.sum())
        self.energy = min(2.0, self.energy + gained)

        # Inventory Economy: a strong meal has a chance to mint a token (feature 9)
        if gained > 0.5:
            color = ["red", "green", "blue"][int(gained * 100) % 3]
            self.inventory[color] += 1.0
        # synergy bonus when holding at least one of every token colour
        if all(v > 0 for v in self.inventory.values()):
            self.energy = min(2.0, self.energy + 0.02)
        return gained

    def _do_attack(self, world: GenesisWorld, population: List["BioHyperAgent"]) -> float:
        target = self._nearest_other(population)
        if target is None:
            return -0.05
        mitigation = float(np.clip(self.trust.get(target.agent_id, 0.0), 0.0, 1.0))
        damage = 0.3 * (1.0 - 0.5 * mitigation)
        target.health = max(0.0, target.health - damage)
        world.deposit_meme(self.x, self.y, MEME_NAMES.index("danger"), 2.0)
        self.trust[target.agent_id] = self.trust.get(target.agent_id, 0.0) - 0.1
        return 0.15 * damage

    def _do_communicate(self, population: List["BioHyperAgent"]) -> float:
        target = self._nearest_other(population)
        if target is None:
            return -0.01
        rho = communicate_blend(self.hrc, target.hrc)
        acc = self.hrc.theory_of_mind_update(target.agent_id, target.hrc.psi)
        self.trust[target.agent_id] = float(
            np.clip(self.trust.get(target.agent_id, 0.0) + 0.02 * rho, -1.0, 1.0)
        )
        return 0.05 * rho + 0.02 * acc

    def _do_reproduce(self, world: GenesisWorld, population: List["BioHyperAgent"]) -> float:
        n = len(population)
        cost = REPRODUCE_BASE_COST * (1.0 + 0.5 * (n / 128.0) ** 2)
        if self.energy < cost + 0.1:
            return -0.05
        partner = self._nearest_compatible_partner(population)
        if partner is None:
            return -0.02
        self.energy -= cost
        # child creation is handled by the caller (evolution.py doesn't exist
        # yet); here we just signal intent + pay the cost. See make_child().
        self._pending_reproduction_partner = partner
        return 0.1

    def _do_invent(self, novelty_scorer, civ_memory) -> float:
        program, godel = self.hrc.attempt_invention()
        name = f"invention_{len(self.discoveries)}"
        self.discoveries.append((program, godel, name))
        reward = 0.05
        if novelty_scorer is not None:
            rho = civ_memory.query(_program_embedding(program)) if civ_memory is not None else 0.0
            n_index, breakthrough = novelty_scorer.score(program, godel, rho)
            reward += n_index * 0.5 + (1.0 if breakthrough else 0.0)
            if civ_memory is not None:
                civ_memory.store(_program_embedding(program))
        return reward

    def _do_rest(self, world: GenesisWorld) -> float:
        self.energy = min(2.0, self.energy + 0.02)
        self.health = min(1.0, self.health + 0.02)
        wonder = self.hrc.emotion("WONDER")
        world.vote_weather(wonder - 0.5)
        return 0.01

    def _do_build_artifact(self, world: GenesisWorld) -> float:
        already_here = any(s.x == self.x and s.y == self.y for s in world.structures)
        if already_here:
            return -0.01
        if self.role == "Warrior":
            world.structures.append(Trap(x=self.x, y=self.y))
        elif self.role == "Processor":
            world.structures.append(Battery(x=self.x, y=self.y))
        else:
            world.structures.append(Cultivator(x=self.x, y=self.y))
        return 0.03

    def _do_absorb_artifact(self, world: GenesisWorld) -> float:
        for s in world.structures:
            if s.x == self.x and s.y == self.y:
                self.hrc._bump_emotion("WONDER", 0.05)
                if isinstance(s, Battery):
                    self.energy = min(2.0, self.energy + s.withdraw(0.1))
                return 0.02
        return 0.0

    def _do_meta_invent(self) -> float:
        surprise = self.hrc.meta.attempt_meta_invention()
        self.n_meta_inventions += 1
        return 0.1 * surprise

    def _do_compose_action(self) -> float:
        mag = np.abs(self.hrc.meta.psi)
        top2 = np.argsort(mag)[-2:]
        program = [ALL_PRIMITIVES[int(i) % len(ALL_PRIMITIVES)] for i in top2]
        try:
            GodelEncoder.encode(program)
            return 0.02
        except ValueError:
            return 0.0

    def _do_trade(self, population: List["BioHyperAgent"]) -> float:
        target = self._nearest_other(population)
        if target is None:
            return -0.005
        my_colors = [c for c, v in self.inventory.items() if v > 0]
        their_colors = [c for c, v in target.inventory.items() if v > 0]
        if not my_colors or not their_colors:
            return -0.005
        give, take = my_colors[0], their_colors[0]
        self.inventory[give] -= 1.0
        target.inventory[give] += 1.0
        target.inventory[take] -= 1.0
        self.inventory[take] += 1.0
        self.trust[target.agent_id] = float(np.clip(self.trust.get(target.agent_id, 0.0) + 0.05, -1.0, 1.0))
        target.trust[self.agent_id] = float(np.clip(target.trust.get(self.agent_id, 0.0) + 0.05, -1.0, 1.0))
        return 0.03

    def _do_punish(self, population: List["BioHyperAgent"]) -> float:
        candidates = [p for p in self._others_in_radius(population) if self.trust.get(p.agent_id, 0.0) < -0.2]
        if not candidates:
            return -0.02
        target = min(candidates, key=lambda p: self.trust.get(p.agent_id, 0.0))
        target.energy = max(0.0, target.energy - 0.15)
        return 0.02  # altruistic punishment: costly to self (via the flat action cost), small direct return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _toroidal_dist(self, other: "BioHyperAgent", world: Optional[GenesisWorld] = None) -> float:
        dx = abs(self.x - other.x)
        dy = abs(self.y - other.y)
        return float(np.hypot(dx, dy))

    def _others_in_radius(self, population: List["BioHyperAgent"], radius: int = INTERACTION_RADIUS) -> List["BioHyperAgent"]:
        return [
            p for p in population
            if p.agent_id != self.agent_id and p.alive and self._toroidal_dist(p) <= radius
        ]

    def _nearest_other(self, population: List["BioHyperAgent"]) -> Optional["BioHyperAgent"]:
        candidates = self._others_in_radius(population)
        if not candidates:
            return None
        return min(candidates, key=lambda p: self._toroidal_dist(p))

    def _nearest_compatible_partner(self, population: List["BioHyperAgent"], min_rho: float = 0.05) -> Optional["BioHyperAgent"]:
        best, best_rho = None, 0.0
        for p in self._others_in_radius(population):
            rho = spectral_resonance(self.hrc, p.hrc)
            if rho > min_rho and rho > best_rho:
                best, best_rho = p, rho
        return best

    # ------------------------------------------------------------------
    # Learn / Evolve / Causal update
    # ------------------------------------------------------------------
    def learn(self, reward: float) -> float:
        iq_bonus = 0.01 * float(np.std(np.abs(self.hrc.psi)))  # feature 8
        return self.hrc.learn(reward + iq_bonus)

    def evolve(self) -> Dict[str, float]:
        return self.hrc.evolve()

    # ------------------------------------------------------------------
    # Kuramoto phase sync  (§3.13, every 5 ticks)
    # ------------------------------------------------------------------
    def kuramoto_update(self, neighbor_thetas: List[float]) -> None:
        if not neighbor_thetas:
            coupling = 0.0
        else:
            coupling = (KURAMOTO_KAPPA / len(neighbor_thetas)) * sum(
                np.sin(t - self.theta) for t in neighbor_thetas
            )
        dtheta = self.natural_freq + coupling
        self.theta = float((self.theta + dtheta * KURAMOTO_DT) % (2 * np.pi))

    # ------------------------------------------------------------------
    # GoL step  (every 10 ticks; role assignment is population-relative —
    # see update_roles() below, called by whoever drives the population)
    # ------------------------------------------------------------------
    def step_gol(self) -> None:
        self.gol_grid = gol_step(self.gol_grid)

    @property
    def gol_alive_fraction(self) -> float:
        return float(self.gol_grid.mean())

    # ------------------------------------------------------------------
    # Viral gene transfer  (feature 28, every 50 ticks)
    # ------------------------------------------------------------------
    def broadcast_viral_packet(self) -> Dict[str, np.ndarray]:
        _, lam = self.hrc._eig()
        return {"spectral_fingerprint": lam.copy(), "soul_fragment": self.hrc.omega[:8].copy()}

    def receive_viral_packet(self, packet: Dict[str, np.ndarray]) -> None:
        fp = packet["spectral_fingerprint"]
        k = min(len(fp), self.hrc.H.shape[0])
        diag_indices = np.arange(k)
        self.hrc.H[diag_indices, diag_indices] = (
            (1 - VIRAL_BLEND) * self.hrc.H[diag_indices, diag_indices]
            + VIRAL_BLEND * fp[:k]
        )
        self.hrc.H = 0.5 * (self.hrc.H + self.hrc.H.conj().T)
        self.hrc._dirty = True

    # ------------------------------------------------------------------
    # Death & apoptosis  (feature 29)
    # ------------------------------------------------------------------
    def death_packet(self) -> Dict[str, object]:
        _, lam = self.hrc._eig()
        meta_lam = self.hrc.meta.eigvals()
        top_discoveries = [name for (_, _, name) in self.discoveries[-3:]]
        causal_top5 = sorted(
            self.causal_model.items(), key=lambda kv: np.mean(kv[1]) if kv[1] else -1e9, reverse=True
        )[:5]
        return {
            "spectral_fingerprint": lam.copy(),
            "meta_H_corner": self.hrc.meta.H[:8, :8].copy(),
            "top_discoveries": top_discoveries,
            "soul_fragment": self.hrc.omega[:8].copy(),
            "causal_top5": [name for name, _ in causal_top5],
        }

    def absorb_spectral_wisdom(self, packet: Dict[str, object]) -> None:
        self.receive_viral_packet({"spectral_fingerprint": packet["spectral_fingerprint"]})

    def maybe_die(self) -> bool:
        if self.energy <= 0.0 or self.health <= 0.0:
            self.alive = False
        return not self.alive

    # ------------------------------------------------------------------
    # Full per-tick lifecycle  (§4.3 schedule)
    # ------------------------------------------------------------------
    def step(
        self,
        world: GenesisWorld,
        population: List["BioHyperAgent"],
        tick: int,
        novelty_scorer=None,
        civ_memory=None,
    ) -> float:
        if not self.alive:
            return 0.0

        self.age += 1
        self.tick_count += 1

        events = self.evolve()
        obs = self.sense(world)
        action_idx = self.decide(obs)
        reward = self.execute(action_idx, world, population, novelty_scorer, civ_memory)
        self.learn(reward)

        if tick % KURAMOTO_EVERY == 0:
            neighbors = self._others_in_radius(population)
            self.kuramoto_update([p.theta for p in neighbors])

        if tick % ROLE_AND_GOL_EVERY == 0:
            self.step_gol()

        self.maybe_die()
        return reward


def _program_embedding(program: List[str], dim: int = 64) -> np.ndarray:
    """Shared with test_smoke.py's placeholder — see civilization.py (built)
    for the eventual authoritative invention-encoding vector (§3.12, §3.20).
    Exported as `invention_embedding` for cross-module reuse so nothing else
    needs a fourth copy of this same placeholder."""
    from metacognition import PRIMITIVE_TO_INDEX

    v = np.zeros(dim, dtype=np.complex128)
    for i, prim in enumerate(program):
        v[(PRIMITIVE_TO_INDEX[prim] * 4 + i) % dim] += 1.0
    return v


invention_embedding = _program_embedding


def update_roles(population: List[BioHyperAgent], queen_population_threshold: int = 80) -> None:
    """
    Population-relative caste assignment (feature 12), called every
    ROLE_AND_GOL_EVERY ticks by whoever drives the population.

    Fixed absolute thresholds on cognitive eigenspread don't work here:
    the scale of `lambda_spread` is emergent and run-dependent (an early
    or slow-learning population might never cross a hardcoded 8.0, while
    a fast-crystallising one might blow past it for everyone) — a run
    that happens to cluster entirely inside one band would collapse to a
    single role for the whole population, which is exactly what an
    earlier fixed-threshold version of this function did. Tertiles of
    the population's own current spread distribution are scale-free and
    guarantee a real three-way split among non-Queens regardless of the
    absolute numbers that particular run produced.

    Queen: population-gated, top-energy fraction (feature 13, eusociality
    / fertility gating at high population) — evaluated before the tertile
    split, since it is a gate, not a rank.
    """
    alive = [p for p in population if p.alive]
    if not alive:
        return

    is_large_pop = len(alive) > queen_population_threshold
    commoners = []
    for p in alive:
        if is_large_pop and p.energy > 1.5:
            p.role = "Queen"
        else:
            commoners.append(p)

    if not commoners:
        return

    spreads = np.array([p.hrc.spectral_summary()["lambda_spread"] for p in commoners])
    lo, hi = np.percentile(spreads, [33.0, 66.0])
    for p, s in zip(commoners, spreads):
        if s <= lo:
            p.role = "Forager"
        elif s <= hi:
            p.role = "Processor"
        else:
            p.role = "Warrior"


def population_order_parameter(population: List[BioHyperAgent]) -> float:
    """Global Kuramoto order parameter r (§3.13)."""
    alive = [p for p in population if p.alive]
    if not alive:
        return 0.0
    phases = np.array([p.theta for p in alive])
    return float(np.abs(np.mean(np.exp(1j * phases))))


def make_child(
    parent_a: BioHyperAgent, parent_b: BioHyperAgent, child_id: int, world: GenesisWorld
) -> BioHyperAgent:
    """Epigenetic psi/H inheritance (§3.18)."""
    rng = np.random.default_rng(child_id)
    alpha = float(rng.uniform(0.35, 0.65))

    psi_a, psi_b = parent_a.hrc.psi, parent_b.hrc.psi
    xi = rng.normal(0, 0.02, psi_a.shape) + 1j * rng.normal(0, 0.02, psi_a.shape)
    psi_child = alpha * psi_a + (1 - alpha) * psi_b + xi
    psi_child = psi_child / np.linalg.norm(psi_child)

    Ha, Hb = parent_a.hrc.H, parent_b.hrc.H
    M = rng.normal(0, 0.020, Ha.shape) + 1j * rng.normal(0, 0.020, Ha.shape)
    H_child = (alpha * Ha + (1 - alpha) * Hb + 0.5 * (M + M.conj().T))
    H_child = 0.5 * (H_child + H_child.conj().T)

    Hm_a, Hm_b = parent_a.hrc.meta.H, parent_b.hrc.meta.H
    N = rng.normal(0, 0.018, Hm_a.shape) + 1j * rng.normal(0, 0.018, Hm_a.shape)
    Hm_child = 0.60 * Hm_a + 0.40 * Hm_b + 0.5 * (N + N.conj().T)
    Hm_child = 0.5 * (Hm_child + Hm_child.conj().T)

    child_hrc = HarmonicResonanceConsciousness(agent_id=child_id, seed=int(rng.integers(0, 2**31)))
    child_hrc.psi = psi_child
    child_hrc.H = H_child
    child_hrc._dirty = True
    child_hrc.meta.H = Hm_child
    child_hrc.meta._dirty = True

    gx, gy = world.wrap(parent_a.x, parent_a.y)
    child = BioHyperAgent(
        agent_id=child_id, x=gx, y=gy,
        generation=max(parent_a.generation, parent_b.generation) + 1,
        hrc=child_hrc,
    )
    # cultural vertical transmission: inherit the primary parent's first 2 discoveries
    child.discoveries = list(parent_a.discoveries[:2])
    return child


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    world = GenesisWorld(height=40, width=40, seed=5)
    population: List[BioHyperAgent] = [
        BioHyperAgent(agent_id=i, x=int(rng.integers(0, 40)), y=int(rng.integers(0, 40)), seed=200 + i)
        for i in range(12)
    ]

    births, deaths = 0, 0
    for tick in range(300):
        world.population_density = len([p for p in population if p.alive]) / (world.height * world.width)
        world.step()

        for agent in list(population):
            if not agent.alive:
                continue
            agent._pending_reproduction_partner = None
            agent.step(world, population, tick)

            if getattr(agent, "_pending_reproduction_partner", None) is not None:
                partner = agent._pending_reproduction_partner
                child = make_child(agent, partner, child_id=1000 + births, world=world)
                population.append(child)
                births += 1

            if not agent.alive:
                packet = agent.death_packet()
                for other in agent._others_in_radius(population, radius=3):
                    other.absorb_spectral_wisdom(packet)
                deaths += 1

        if tick % VIRAL_BROADCAST_EVERY == 0 and tick > 0:
            alive = [p for p in population if p.alive]
            if alive:
                fittest = max(alive, key=lambda p: len(p.discoveries))
                packet = fittest.broadcast_viral_packet()
                for p in alive:
                    if p.agent_id != fittest.agent_id:
                        p.receive_viral_packet(packet)

        if tick % MEME_ABSORPTION_EVERY == 0 and tick > 0:
            for p in population:
                if p.alive:
                    world.deposit_meme(p.x, p.y, MEME_NAMES.index("resource"), 0.1)

        if tick % ROLE_AND_GOL_EVERY == 0:
            update_roles(population)

    alive_pop = [p for p in population if p.alive]
    print(f"ticks=300  final_alive={len(alive_pop)}  births={births}  deaths={deaths}")
    print(f"kuramoto order parameter r = {population_order_parameter(population):.4f}")

    for p in alive_pop[:20]:
        assert np.isfinite(p.hrc.psi).all()
        assert np.allclose(p.hrc.H, p.hrc.H.conj().T, atol=1e-8)
        assert 0.0 <= p.energy <= 2.0
        assert 0.0 <= p.health <= 1.0
    print("agent invariants OK across sampled survivors.")

    role_counts: Dict[str, int] = {}
    for p in alive_pop:
        role_counts[p.role] = role_counts.get(p.role, 0) + 1
    print("role distribution:", role_counts)

    total_discoveries = sum(len(p.discoveries) for p in alive_pop)
    print(f"total surviving discoveries: {total_discoveries}")
    print("agents.py self-test passed.")
