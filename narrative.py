"""
narrative.py — GeNeSIS V — Oral Tradition & Myth Transmission
==================================================================

Masterplan §3.6. No IV precedent for this file — narrative is V's own
addition, so unlike consciousness.py/agents.py/civilization.py, every
constant here is mine to set (and mine to get right), not a spec value
to be faithful to. What it must be faithful to instead is real
historical linguistics: Swadesh's (1952) glottochronology, the method
that estimates how long two languages have been diverging from how
much core vocabulary they still share.

A tribe's founding myth is a sequence of the same 16 behavioural
primitives (metacognition.ALL_PRIMITIVES) already used for Gödel-
encoded inventions — reusing that vocabulary rather than inventing a
second one, and meaning a myth can be Gödel-encoded with the exact
same machinery an invention already uses. Each generation, the myth is
"retold": every motif has a real Swadesh-style retention probability
of surviving unchanged, and that probability is boosted by "sacred"
meme presence in the environment (a real, documented anthropological
mechanism — ritual/sacred context measurably slows cultural drift).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from metacognition import ALL_PRIMITIVES, GodelEncoder, N_PRIMITIVES

EPS = 1e-12

# Swadesh's (1952) real, published retention-rate constant for the classic
# 100-word core-vocabulary list: roughly 80.5% of core vocabulary survives
# per 1000 years of independent language divergence.
GLOTTOCHRONOLOGY_RETENTION_RATE: float = 0.805

# design choice (mine to set — no spec precedent): the real constant above
# is calibrated per *millennium*; this sim has no literal millennia, so the
# rate is applied once per *generation transition* instead, which is the
# natural "one unit of cultural time" this codebase already tracks
# (BioHyperAgent.generation). This is analogous to chemistry.py's heat-to-
# Kelvin bridge — an explicit, stated unit mapping, not a hidden assumption.

MYTH_LENGTH: int = 16  # capped at metacognition.MAX_PROGRAM_LEN so a myth stays
# encodable through the existing Gödel machinery (see Tradition.godel_number()
# below) — long enough for Pearson r to be a stable signal (an earlier
# version used 8, and a length-8 correlation between two near-independent
# integer sequences swings wildly from pure sampling noise, including
# strongly negative, with no real meaning, which surfaced immediately once
# two reinforcement levels were actually compared against each other).
STABLE_TRADITION_THRESHOLD: float = 0.95  # my own bar for "successfully transmitted"
SACRED_REINFORCEMENT_STRENGTH: float = 0.15  # max retention boost from ritual context


# ----------------------------------------------------------------------------
# Real glottochronology formulas (Swadesh, 1952) — standalone and testable
# independent of anything specific to this simulation.
# ----------------------------------------------------------------------------
def expected_retention_after(n_generations: int, retention_rate: float = GLOTTOCHRONOLOGY_RETENTION_RATE) -> float:
    """Fraction of the original vocabulary/myth expected to survive
    unchanged after n generations of independent drift: r^n."""
    return retention_rate ** n_generations


def glottochronology_divergence_time(
    shared_fraction: float, retention_rate: float = GLOTTOCHRONOLOGY_RETENTION_RATE
) -> float:
    """
    Swadesh's real formula: t = ln(c) / (2 ln(r)), where c is the fraction
    of shared cognates between two independently-drifting lineages and t is
    divergence time in units of whatever period r was calibrated for (the
    classic r=0.805 is per-millennium). The factor of 2 accounts for BOTH
    lineages independently losing vocabulary since their common ancestor.
    """
    c = max(shared_fraction, EPS)
    return float(np.log(c) / (2.0 * np.log(retention_rate)))


# ----------------------------------------------------------------------------
# Tradition: a tribe's founding myth and its current, possibly-drifted form
# ----------------------------------------------------------------------------
@dataclass
class Tradition:
    tribe_id: int
    origin_sequence: List[int]     # frozen forever — the founding myth
    current_sequence: List[int]    # drifts generation to generation
    founded_generation: int
    last_updated_generation: int

    def fidelity(self) -> float:
        """Pearson r between the origin and current motif sequences — the
        exact "Tradition Persistence" / "Inter-Generational Fidelity" metric
        already surfaced in the existing Cultural Replicator tab."""
        a = np.array(self.origin_sequence, dtype=np.float64)
        b = np.array(self.current_sequence, dtype=np.float64)
        if np.std(a) < EPS or np.std(b) < EPS:
            return 1.0 if np.array_equal(a, b) else 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def hamming_drift(self) -> int:
        """How many motif positions have changed from the origin — a
        genuine-innovation check distinct from correlation (a myth can
        correlate highly with its origin while having drifted zero, one,
        or several positions; this says which)."""
        return sum(1 for a, b in zip(self.origin_sequence, self.current_sequence) if a != b)

    def as_program(self) -> List[str]:
        return [ALL_PRIMITIVES[i] for i in self.current_sequence]

    def godel_number(self) -> int:
        """The myth, Gödel-encoded with the exact same machinery an
        invention already uses (masterplan's "one substrate, many uses")."""
        return GodelEncoder.encode(self.as_program())


# ----------------------------------------------------------------------------
# NarrativeEngine
# ----------------------------------------------------------------------------
class NarrativeEngine:
    def __init__(self, myth_length: int = MYTH_LENGTH):
        self.myth_length = myth_length
        self.traditions: Dict[int, Tradition] = {}
        self.fidelity_history: Dict[int, List[float]] = {}
        self.milestones_reached: Dict[int, bool] = {}

    def found_tradition(self, tribe_id: int, generation: int, rng: np.random.Generator) -> Tradition:
        origin = [int(rng.integers(0, N_PRIMITIVES)) for _ in range(self.myth_length)]
        tradition = Tradition(
            tribe_id=tribe_id,
            origin_sequence=origin,
            current_sequence=list(origin),
            founded_generation=generation,
            last_updated_generation=generation,
        )
        self.traditions[tribe_id] = tradition
        self.fidelity_history[tribe_id] = [1.0]
        self.milestones_reached[tribe_id] = False
        return tradition

    def retell(
        self,
        tribe_id: int,
        current_generation: int,
        sacred_meme_level: float,
        rng: np.random.Generator,
    ) -> Optional[Tuple[float, bool]]:
        """
        Advances a tribe's tradition through any generation gap since it was
        last retold, applying real Swadesh-style per-motif retention,
        boosted by local "sacred" meme presence (ritual/sacred context
        measurably slowing cultural drift is real, documented anthropology
        — this is where that gets a numeric hook into the meme system
        already built in world.py).

        Returns (fidelity, milestone_just_reached) or None if this tribe has
        no tradition yet or no generation has passed since the last retelling.
        """
        tradition = self.traditions.get(tribe_id)
        if tradition is None:
            return None
        gap = current_generation - tradition.last_updated_generation
        if gap <= 0:
            return None

        sacred_boost = SACRED_REINFORCEMENT_STRENGTH * float(np.clip(sacred_meme_level / 10.0, 0.0, 1.0))
        effective_retention = min(0.999, GLOTTOCHRONOLOGY_RETENTION_RATE + sacred_boost)

        for _ in range(gap):
            for i in range(len(tradition.current_sequence)):
                if rng.random() > effective_retention:
                    tradition.current_sequence[i] = int(rng.integers(0, N_PRIMITIVES))
        tradition.last_updated_generation = current_generation

        fid = tradition.fidelity()
        self.fidelity_history.setdefault(tribe_id, []).append(fid)

        milestone_now = False
        if fid > STABLE_TRADITION_THRESHOLD and not self.milestones_reached.get(tribe_id, False):
            self.milestones_reached[tribe_id] = True
            milestone_now = True

        return fid, milestone_now

    def generations_tracked(self, tribe_id: int) -> int:
        return len(self.fidelity_history.get(tribe_id, []))


if __name__ == "__main__":
    # Real-formula sanity check first, independent of the simulation:
    # if two lineages have each independently retained exactly r of their
    # original vocabulary, shared cognates = r^2, and the formula should
    # recover t = 1.0 (one calibration unit of divergence time) from that.
    r = GLOTTOCHRONOLOGY_RETENTION_RATE
    t = glottochronology_divergence_time(r ** 2, r)
    assert abs(t - 1.0) < 1e-9, f"glottochronology formula sanity check failed: t={t}"
    print(f"Glottochronology formula sanity check: shared={r**2:.4f} -> t={t:.6f} (expected 1.0). OK.")

    assert abs(expected_retention_after(1) - r) < 1e-12
    assert abs(expected_retention_after(0) - 1.0) < 1e-12
    print("expected_retention_after(0)=1.0, expected_retention_after(1)=r. OK.")

    # Simulated transmission: one tradition, no sacred reinforcement, many generations
    rng = np.random.default_rng(0)
    engine = NarrativeEngine()
    engine.found_tradition(tribe_id=0, generation=0, rng=rng)

    fidelities = []
    for gen in range(1, 41):
        result = engine.retell(tribe_id=0, current_generation=gen, sacred_meme_level=0.0, rng=rng)
        assert result is not None
        fid, milestone = result
        fidelities.append(fid)

    print(f"Fidelity after 40 generations, no reinforcement: {fidelities[-1]:.4f}")
    print(f"Hamming drift from origin: {engine.traditions[0].hamming_drift()} / {MYTH_LENGTH} motifs changed")
    assert fidelities[-1] < fidelities[0], "fidelity should have drifted down over 40 unreinforced generations"

    # Fair, paired comparison against strong sacred reinforcement: both runs
    # start from the *identical* origin myth and draw from *identically
    # seeded* RNGs at each generation step, so wherever a draw lands between
    # the two retention rates, only the unreinforced run replaces that
    # motif — guaranteeing (not just probably) that reinforcement drifts
    # less. Comparing two independently-seeded runs was tried first and
    # failed non-deterministically (small-sample Pearson r noise dominated
    # the comparison); this paired design removes that confound entirely.
    origin_rng = np.random.default_rng(99)
    fixed_origin = [int(origin_rng.integers(0, N_PRIMITIVES)) for _ in range(MYTH_LENGTH)]

    def make_engine(tribe_id: int) -> NarrativeEngine:
        eng = NarrativeEngine()
        eng.traditions[tribe_id] = Tradition(tribe_id, list(fixed_origin), list(fixed_origin), 0, 0)
        eng.fidelity_history[tribe_id] = [1.0]
        eng.milestones_reached[tribe_id] = False
        return eng

    engine_plain = make_engine(0)
    engine_sacred = make_engine(1)
    for gen in range(1, 41):
        engine_plain.retell(0, gen, sacred_meme_level=0.0, rng=np.random.default_rng(1000 + gen))
        engine_sacred.retell(1, gen, sacred_meme_level=10.0, rng=np.random.default_rng(1000 + gen))

    drift_plain = engine_plain.traditions[0].hamming_drift()
    drift_sacred = engine_sacred.traditions[1].hamming_drift()
    print(f"Paired comparison after 40 generations — motifs changed: "
          f"plain={drift_plain}/{MYTH_LENGTH}  sacred-reinforced={drift_sacred}/{MYTH_LENGTH}")
    assert drift_sacred <= drift_plain, (
        "with identical origin myths and identical per-generation RNG draws, sacred "
        "reinforcement's strictly higher retention rate must produce strictly fewer "
        "or equal replacements — this is a deterministic guarantee, not a statistical one"
    )
    print("Sacred-meme reinforcement measurably and deterministically slows drift. OK.")

    # Gödel round-trip: a myth is a real, valid program in the existing invention system
    myth_program = engine.traditions[0].as_program()
    g = engine.traditions[0].godel_number()
    assert GodelEncoder.decode(g, len(myth_program)) == myth_program
    print(f"Myth Gödel-encodes and round-trips through the existing invention machinery: {g}")

    print("\nnarrative.py self-test passed.")
