"""
civilization.py — GeNeSIS V — Tribes, Diplomacy, TechTree
===============================================================

README IV §4.5, using the exact §3.16 (Tribal Power), §3.17 (Epistemic
Schism), and §3.20 (Autoassociative Civilisation Memory) formulas.

Tribe formation: an unaffiliated agent samples up to 6 members from
each existing tribe, computes mean spectral resonance against each
sample, and joins whichever tribe scores highest — if that best score
clears 0.08. Otherwise it founds a new tribe.

Diplomacy: pairwise tribal power ratios drive Alliance/War rolls;
allied tribes are continuously pulled together by cultural
assimilation (every tick) and can still be pulled apart by an
epistemic schism if their meta-cognitive identities diverge past the
spec's threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from metacognition import CivilizationMemory, GodelEncoder, NoveltyScorer
from consciousness import HarmonicResonanceConsciousness
from agents import BioHyperAgent, invention_embedding

EPS = 1e-12

# ----------------------------------------------------------------------------
# Constants  (§3.16, §3.17, §4.5)
# ----------------------------------------------------------------------------
TRIBE_JOIN_THRESHOLD: float = 0.08
TRIBE_SAMPLE_SIZE: int = 6

SCHISM_THRESHOLD: float = 48.0        # Theta_schism, §3.17
CULTURAL_ASSIMILATION_RATE: float = 0.05  # the 0.05 in 0.95/0.05 blend, §3.17

ALLIANCE_RATIO_LOW: float = 0.5
ALLIANCE_RATIO_HIGH: float = 2.0
ALLIANCE_ROLL_PROB: float = 0.60

WAR_RATIO_HIGH: float = 3.0
WAR_RATIO_LOW: float = 1.0 / 3.0
WAR_ROLL_PROB: float = 0.02  # "rare, extreme disparity only"

B_TECH_GROWTH: float = 1.003
B_TECH_CAP: float = 3.0

SINGULARITY_NOVELTY_THRESHOLD: float = 0.55
SINGULARITY_MIN_TECH_NODES: int = 15

# design choice, not specified by the spec: cadence for the diplomacy check.
# Tribe assignment and cultural assimilation are both explicitly "every
# tick" in the spec text; diplomacy is not given a cadence, and checking
# every possible tribe pair every tick is wasteful for something this slow-
# moving (alliances/wars/schisms are civilisation-scale events, not
# per-tick ones) — every 20 ticks is a reasonable, documented choice.
DIPLOMACY_EVERY: int = 20


# ----------------------------------------------------------------------------
# Spectral RGB — a tribe's colour is derived from its founder's spectrum
# ----------------------------------------------------------------------------
def spectral_rgb(hrc: HarmonicResonanceConsciousness) -> Tuple[int, int, int]:
    """
    # design choice: the spec says a tribe's colour comes from the founder's
    # "spectral RGB" without fixing a mapping. This one takes the min,
    # median, and max of the founder's Hamiltonian eigenvalues and min-max
    # normalises each into a fixed, empirically-observed eigenvalue range
    # (roughly [-2, 12], per values seen across every self-test in this
    # project so far) into a 0-255 channel.
    """
    _, lam = hrc._eig()
    lo, mid, hi = float(lam.min()), float(np.median(lam)), float(lam.max())

    def to_channel(v: float) -> int:
        return int(np.clip((v + 2.0) / 14.0 * 255.0, 0, 255))

    return to_channel(lo), to_channel(mid), to_channel(hi)


# ----------------------------------------------------------------------------
# Tribe
# ----------------------------------------------------------------------------
@dataclass
class Tribe:
    tribe_id: int
    founder_id: int
    color: Tuple[int, int, int]
    founded_tick: int
    member_ids: Set[int] = field(default_factory=set)
    alliances: Set[int] = field(default_factory=set)
    discoveries: int = 0
    memory: CivilizationMemory = field(default_factory=CivilizationMemory)

    def members(self, population: List[BioHyperAgent]) -> List[BioHyperAgent]:
        return [p for p in population if p.agent_id in self.member_ids and p.alive]

    def trade_count(self, population: List[BioHyperAgent]) -> int:
        return sum(p.action_counts.get("trade", 0) for p in self.members(population))


# ----------------------------------------------------------------------------
# Civilization
# ----------------------------------------------------------------------------
class Civilization:
    def __init__(self):
        self.tribes: Dict[int, Tribe] = {}
        self._next_tribe_id: int = 0

        self.tech_tree = nx.DiGraph()
        self.b_tech: float = 1.0

        self.global_memory = CivilizationMemory()
        self.novelty_scorer = NoveltyScorer()

        self.events: List[str] = []

        # Needed by nobel.py's Peace category: when each currently-active
        # alliance formed, and how long past alliances lasted before ending
        # (by schism or by a member tribe going extinct).
        self.alliance_formed_tick: Dict[frozenset, int] = {}
        self.alliance_lifespans: List[int] = []

    # ------------------------------------------------------------------
    # Tribe formation
    # ------------------------------------------------------------------
    def found_tribe(self, founder: BioHyperAgent, tick: int) -> Tribe:
        tribe = Tribe(
            tribe_id=self._next_tribe_id,
            founder_id=founder.agent_id,
            color=spectral_rgb(founder.hrc),
            founded_tick=tick,
        )
        tribe.member_ids.add(founder.agent_id)
        self.tribes[tribe.tribe_id] = tribe
        founder.tribe_id = tribe.tribe_id
        self._next_tribe_id += 1
        return tribe

    def assign_tribe(self, agent: BioHyperAgent, population: List[BioHyperAgent], rng: np.random.Generator, tick: int) -> int:
        if not self.tribes:
            return self.found_tribe(agent, tick).tribe_id

        from agents import spectral_resonance  # local import: avoids a module-level cycle

        best_tribe_id, best_score = None, -1.0
        for tribe in self.tribes.values():
            members = [p for p in tribe.members(population) if p.agent_id != agent.agent_id]
            if not members:
                continue
            sample = list(rng.choice(members, size=min(TRIBE_SAMPLE_SIZE, len(members)), replace=False))
            scores = [spectral_resonance(agent.hrc, m.hrc) for m in sample]
            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_tribe_id, best_score = tribe.tribe_id, mean_score

        if best_tribe_id is not None and best_score > TRIBE_JOIN_THRESHOLD:
            self.tribes[best_tribe_id].member_ids.add(agent.agent_id)
            agent.tribe_id = best_tribe_id
            return best_tribe_id

        return self.found_tribe(agent, tick).tribe_id

    def liquid_dunbar_max(self) -> int:
        """N_max(t) = 12 + 3 * |TechTree(t)| — tribe capacity grows with technology."""
        return 12 + 3 * self.tech_tree.number_of_nodes()

    # ------------------------------------------------------------------
    # TechTree + invention registration  (§3.11, §3.12, §3.20, Singularity Override)
    # ------------------------------------------------------------------
    def register_invention(
        self, agent: BioHyperAgent, program: List[str], godel: int, tick: int
    ) -> Tuple[float, bool]:
        if len(self.tech_tree) > 0:
            nearest = min(self.tech_tree.nodes, key=lambda g: GodelEncoder.distance(g, godel))
            self.tech_tree.add_edge(nearest, godel)
        self.tech_tree.add_node(godel, tick=tick, tribe_id=agent.tribe_id, program=list(program))
        self.b_tech = min(B_TECH_CAP, self.b_tech * B_TECH_GROWTH)

        embedding = invention_embedding(program)
        tribe = self.tribes.get(agent.tribe_id) if agent.tribe_id is not None else None
        rho = tribe.memory.query(embedding) if tribe is not None else self.global_memory.query(embedding)

        n_index, breakthrough = self.novelty_scorer.score(program, godel, rho)

        if tribe is not None:
            tribe.discoveries += 1
            tribe.memory.store(embedding)
        self.global_memory.store(embedding)

        if n_index > SINGULARITY_NOVELTY_THRESHOLD and self.tech_tree.number_of_nodes() >= SINGULARITY_MIN_TECH_NODES:
            if not breakthrough:
                self.events.append(
                    f"⚡ SINGULARITY OVERRIDE: tick {tick}, agent {agent.agent_id} — "
                    f"novelty {n_index:.3f} auto-promoted at {self.tech_tree.number_of_nodes()} tech nodes"
                )
            breakthrough = True

        return n_index, breakthrough

    # ------------------------------------------------------------------
    # Tribal power  (§3.16, exact 8-term formula)
    # ------------------------------------------------------------------
    def compute_tribal_power(self, tribe_id: int, population: List[BioHyperAgent]) -> float:
        tribe = self.tribes[tribe_id]
        members = tribe.members(population)
        if not members:
            return 0.0

        sum_energy = sum(p.energy for p in members)
        sum_health = sum(p.health for p in members)
        d = tribe.discoveries
        size = len(members)
        n_alliances = len(tribe.alliances)
        mean_sigma_meta = float(np.mean([np.std(p.hrc.meta.eigvals()) for p in members]))
        mean_phi = float(np.mean([p.hrc.phi_history[-1] if p.hrc.phi_history else 0.0 for p in members]))
        n_trades = tribe.trade_count(population)

        return (
            0.30 * sum_energy
            + 0.20 * sum_health
            + 2.5 * d
            + 0.55 * size
            + 1.5 * n_alliances
            + 3.0 * mean_sigma_meta
            + 5.0 * mean_phi
            + 0.10 * n_trades
        )

    # ------------------------------------------------------------------
    # Epistemic schism + cultural assimilation  (§3.17, exact formulas)
    # ------------------------------------------------------------------
    def _tribal_meta_hamiltonian(self, tribe_id: int, population: List[BioHyperAgent]) -> Optional[np.ndarray]:
        members = self.tribes[tribe_id].members(population)
        if not members:
            return None
        H_sum = sum(p.hrc.meta.H for p in members)
        return H_sum / len(members)

    def epistemic_distance(self, tribe_a: int, tribe_b: int, population: List[BioHyperAgent]) -> Optional[float]:
        Ha = self._tribal_meta_hamiltonian(tribe_a, population)
        Hb = self._tribal_meta_hamiltonian(tribe_b, population)
        if Ha is None or Hb is None:
            return None
        lam_a = np.linalg.eigvalsh(Ha)
        lam_b = np.linalg.eigvalsh(Hb)
        return float(np.linalg.norm(lam_a - lam_b))

    def step_cultural_assimilation(self, population: List[BioHyperAgent]) -> None:
        """Every tick: each member's meta-Hamiltonian is nudged toward its
        tribe's average — the force that opposes epistemic schism."""
        for tribe_id, tribe in self.tribes.items():
            H_tribe = self._tribal_meta_hamiltonian(tribe_id, population)
            if H_tribe is None:
                continue
            for member in tribe.members(population):
                Hm = member.hrc.meta.H
                Hm_new = (1 - CULTURAL_ASSIMILATION_RATE) * Hm + CULTURAL_ASSIMILATION_RATE * H_tribe
                member.hrc.meta.H = 0.5 * (Hm_new + Hm_new.conj().T)
                member.hrc.meta._dirty = True

    # ------------------------------------------------------------------
    # Diplomacy  (Alliance / War / Schism)
    # ------------------------------------------------------------------
    def step_diplomacy(self, population: List[BioHyperAgent], rng: np.random.Generator, tick: int) -> None:
        tribe_ids = [tid for tid, t in self.tribes.items() if t.members(population)]
        for i, a in enumerate(tribe_ids):
            for b in tribe_ids[i + 1 :]:
                pa = self.compute_tribal_power(a, population)
                pb = self.compute_tribal_power(b, population)
                if pb < EPS:
                    continue
                ratio = pa / pb

                already_allied = b in self.tribes[a].alliances
                if already_allied:
                    d_epi = self.epistemic_distance(a, b, population)
                    if d_epi is not None and d_epi > SCHISM_THRESHOLD:
                        self.tribes[a].alliances.discard(b)
                        self.tribes[b].alliances.discard(a)
                        pair = frozenset((a, b))
                        formed = self.alliance_formed_tick.pop(pair, tick)
                        self.alliance_lifespans.append(tick - formed)
                        self.events.append(
                            f"\U0001F531 SCHISM: tribe {a} and tribe {b} at tick {tick} "
                            f"(epistemic distance {d_epi:.2f} > {SCHISM_THRESHOLD})"
                        )
                    continue

                if ALLIANCE_RATIO_LOW < ratio < ALLIANCE_RATIO_HIGH and rng.random() < ALLIANCE_ROLL_PROB:
                    self.tribes[a].alliances.add(b)
                    self.tribes[b].alliances.add(a)
                    self.alliance_formed_tick[frozenset((a, b))] = tick
                    self.events.append(f"\U0001F91D ALLIANCE: tribe {a} and tribe {b} at tick {tick}")
                elif (ratio > WAR_RATIO_HIGH or ratio < WAR_RATIO_LOW) and rng.random() < WAR_ROLL_PROB:
                    self.events.append(f"\u2694\uFE0F WAR: tribe {a} vs tribe {b} at tick {tick} (power ratio {ratio:.2f})")

    # ------------------------------------------------------------------
    # Full per-tick step
    # ------------------------------------------------------------------
    def step(self, population: List[BioHyperAgent], rng: np.random.Generator, tick: int) -> Dict[str, object]:
        for agent in population:
            if agent.alive and agent.tribe_id is None:
                self.assign_tribe(agent, population, rng, tick)

        self.step_cultural_assimilation(population)

        if tick % DIPLOMACY_EVERY == 0 and tick > 0:
            self.step_diplomacy(population, rng, tick)

        return {
            "n_tribes": len([t for t in self.tribes.values() if t.members(population)]),
            "b_tech": self.b_tech,
            "n_tech_nodes": self.tech_tree.number_of_nodes(),
            "n_events": len(self.events),
        }


if __name__ == "__main__":
    from world import GenesisWorld
    from agents import make_child, MEME_NAMES, VIRAL_BROADCAST_EVERY, MEME_ABSORPTION_EVERY, update_roles
    from evolution import EvolutionEngine

    rng = np.random.default_rng(0)
    world = GenesisWorld(height=40, width=40, seed=5)
    population: List[BioHyperAgent] = [
        BioHyperAgent(agent_id=i, x=int(rng.integers(0, 40)), y=int(rng.integers(0, 40)), seed=200 + i)
        for i in range(32)
    ]
    next_agent_id = 100000
    civ = Civilization()
    evo = EvolutionEngine()

    N_TICKS = 500
    for tick in range(N_TICKS):
        alive_now = [p for p in population if p.alive]
        world.population_density = len(alive_now) / (world.height * world.width)
        world.step()

        for agent in list(population):
            if not agent.alive:
                continue
            agent._pending_reproduction_partner = None
            agent.step(world, population, tick, novelty_scorer=civ.novelty_scorer,
                       civ_memory=civ.tribes[agent.tribe_id].memory if agent.tribe_id is not None else civ.global_memory)

            # invention -> register with the tech tree/tribal memory too
            if agent.discoveries and agent.discoveries[-1][2] == f"invention_{len(agent.discoveries) - 1}":
                last_program, last_godel, _ = agent.discoveries[-1]
                if not hasattr(agent, "_last_registered") or agent._last_registered != last_godel:
                    civ.register_invention(agent, last_program, last_godel, tick)
                    agent._last_registered = last_godel

            if getattr(agent, "_pending_reproduction_partner", None) is not None:
                partner = agent._pending_reproduction_partner
                child = make_child(agent, partner, child_id=next_agent_id, world=world)
                population.append(child)
                next_agent_id += 1

            if not agent.alive:
                packet = agent.death_packet()
                for other in agent._others_in_radius(population, radius=3):
                    other.absorb_spectral_wisdom(packet)

        if tick % 10 == 0:
            update_roles(population)

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

        _, next_agent_id = evo.step(population, world, tick, next_agent_id)
        civ_events = civ.step(population, rng, tick)

        if tick % 100 == 0:
            print(f"tick {tick:4d}  alive={len([p for p in population if p.alive]):3d}  "
                  f"tribes={civ_events['n_tribes']:2d}  tech_nodes={civ_events['n_tech_nodes']:3d}  "
                  f"b_tech={civ_events['b_tech']:.3f}  events={civ_events['n_events']}")

    print()
    print(f"Final tribes: {len([t for t in civ.tribes.values() if t.members(population)])}")
    print(f"Tech tree nodes: {civ.tech_tree.number_of_nodes()}")
    print(f"b_tech final: {civ.b_tech:.4f} (cap {B_TECH_CAP})")
    print(f"Civilization events logged: {len(civ.events)}")
    for e in civ.events[:10]:
        print("  ", e)

    alive_pop = [p for p in population if p.alive]
    powers = {tid: civ.compute_tribal_power(tid, population) for tid, t in civ.tribes.items() if t.members(population)}
    print(f"Tribal powers: { {k: round(v, 2) for k, v in powers.items()} }")

    # epistemic distance sanity check, same honest-finding methodology as evolution.py's clades
    tribe_ids = list(powers.keys())
    if len(tribe_ids) >= 2:
        d = civ.epistemic_distance(tribe_ids[0], tribe_ids[1], population)
        print(f"Sample epistemic distance between tribe {tribe_ids[0]} and {tribe_ids[1]}: {d:.4f} "
              f"(schism threshold: {SCHISM_THRESHOLD})")

    for p in alive_pop[:20]:
        assert np.isfinite(p.hrc.psi).all()
        assert np.allclose(p.hrc.H, p.hrc.H.conj().T, atol=1e-8)
        assert np.allclose(p.hrc.meta.H, p.hrc.meta.H.conj().T, atol=1e-7)
    print("\ncivilization.py self-test passed.")
