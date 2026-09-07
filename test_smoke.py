"""
test_smoke.py — GeNeSIS V — Cross-module integration smoke test
===================================================================

Not a unit-test framework file (no pytest dependency) — a single,
readable script that stands a small population up, runs it for real,
and asserts the invariants that matter: unit-norm psi, exactly
Hermitian H, finite (no NaN/Inf) fields everywhere, and that every
public code path in metacognition.py, consciousness.py, and biology.py
gets exercised together at least once, the way agents.py (not yet
built) eventually will.

Run:  python3 test_smoke.py
"""

from __future__ import annotations

import numpy as np

from metacognition import (
    CivilizationMemory,
    GodelEncoder,
    K_DIM,
    NoveltyScorer,
    PhylogeneticTracker,
    PRIMITIVE_TO_INDEX,
)
from consciousness import HarmonicResonanceConsciousness
from biology import (
    french_flag_zones,
    gray_scott_step,
    genetic_distance,
    genome_fingerprint,
    init_grid,
    biome_rgb,
)

POP_SIZE = 8
N_TICKS = 200
OBS_DIM = 16


def invention_embedding(program, dim: int = K_DIM) -> np.ndarray:
    """
    Placeholder encoding of a behavioural program into a 64-dim vector for
    CivilizationMemory queries in this smoke test. The eventual civilization.py
    module owns the real "invention encoding v" from §3.12 / §3.20 — this is
    a deliberately simple stand-in so the memory/novelty pipeline can be
    exercised end-to-end today.
    """
    v = np.zeros(dim, dtype=np.complex128)
    for i, prim in enumerate(program):
        v[(PRIMITIVE_TO_INDEX[prim] * 4 + i) % dim] += 1.0
    return v


def check_agent_invariants(hrc: HarmonicResonanceConsciousness) -> None:
    assert np.isfinite(hrc.psi).all(), f"agent {hrc.agent_id}: non-finite psi"
    assert np.isfinite(hrc.H).all(), f"agent {hrc.agent_id}: non-finite H"
    assert abs(np.linalg.norm(hrc.psi) - 1.0) < 1e-8, (
        f"agent {hrc.agent_id}: psi not unit norm ({np.linalg.norm(hrc.psi)})"
    )
    assert np.allclose(hrc.H, hrc.H.conj().T, atol=1e-9), (
        f"agent {hrc.agent_id}: H not Hermitian"
    )
    assert np.isfinite(hrc.meta.psi).all(), f"agent {hrc.agent_id}: non-finite psi_meta"
    assert np.allclose(hrc.meta.H, hrc.meta.H.conj().T, atol=1e-9), (
        f"agent {hrc.agent_id}: H_meta not Hermitian"
    )


def main() -> None:
    rng = np.random.default_rng(2026)
    population = [
        HarmonicResonanceConsciousness(agent_id=i, seed=1000 + i) for i in range(POP_SIZE)
    ]

    global_memory = CivilizationMemory()
    novelty = NoveltyScorer()
    phylo = PhylogeneticTracker(split_threshold=6.0)

    phi_log = []
    breakthroughs = 0

    for tick in range(N_TICKS):
        for hrc in population:
            obs = rng.normal(size=OBS_DIM)
            events = hrc.evolve()
            action_idx, probs = hrc.decide(obs)
            assert 0 <= action_idx < len(probs)
            assert abs(probs.sum() - 1.0) < 1e-6

            reward = float(rng.normal(0, 1.2))
            hrc.learn(reward)

            if tick % 5 == 0:
                hrc.active_inference_step(obs)

            if "phi" in events:
                phi_log.append(events["phi"])

            if tick % 25 == 0 and tick > 0:
                program, godel = hrc.attempt_invention()
                v = invention_embedding(program)
                rho = global_memory.query(v)
                n_index, is_breakthrough = novelty.score(program, godel, rho)
                global_memory.store(v)
                if is_breakthrough:
                    breakthroughs += 1

                clade = phylo.assign(hrc.agent_id, hrc.meta.eigvals(), tick)

            check_agent_invariants(hrc)

        # cross-agent theory of mind, every 10 ticks
        if tick % 10 == 0:
            a, b = population[0], population[1]
            acc = a.theory_of_mind_update(b.agent_id, b.psi)
            assert -1.0 <= acc <= 1.0

    # ---- Reporting -------------------------------------------------------
    print(f"Ran {N_TICKS} ticks over {POP_SIZE} agents without invariant violations.")
    print(f"Phi samples collected: {len(phi_log)}  mean={np.mean(phi_log):.5f}  max={np.max(phi_log):.5f}")
    print(f"Breakthroughs detected: {breakthroughs} / {novelty.stats.n} inventions scored")
    print(f"Cambrian explosion flag: {novelty.cambrian_explosion()}")
    print(f"Phylogenetic clades formed: {phylo.n_clades}")
    print(f"Global CivilizationMemory spectral summary (top 4): "
          f"{global_memory.spectral_summary(4)}")

    # ---- Gödel round-trip sanity ------------------------------------------
    test_program = ["move", "signal", "reflect", "eat", "teach"]
    g = GodelEncoder.encode(test_program)
    assert GodelEncoder.decode(g, len(test_program)) == test_program
    print("Gödel encode/decode round-trip OK.")

    # ---- Biology: codon reading + genetic distance across the population --
    dnas = [genome_fingerprint(hrc.eigenvectors())["dna"] for hrc in population]
    for dna in dnas:
        assert len(dna) == K_DIM
        assert set(dna) <= {"A", "C", "G", "T"}
    pairwise = [
        genetic_distance(dnas[i], dnas[j])
        for i in range(POP_SIZE)
        for j in range(i + 1, POP_SIZE)
    ]
    print(f"Genetic distance across population: mean={np.mean(pairwise):.3f} "
          f"min={np.min(pairwise):.3f} max={np.max(pairwise):.3f}")

    # ---- Biology: Gray-Scott morphogenesis + French Flag body-plan zones --
    A, B = init_grid(64, 64, seed=11)
    for _ in range(600):
        A, B = gray_scott_step(A, B)
    assert np.isfinite(A).all() and np.isfinite(B).all()
    hue = rng.uniform(0, 1, size=A.shape)
    rgb = biome_rgb(A, B, hue)
    assert rgb.shape == (64, 64, 3)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    zones = french_flag_zones(A)
    assert set(np.unique(zones)) <= {0, 1, 2}
    print("Gray-Scott + biome_rgb + French Flag zoning OK.")

    print("\nALL SMOKE TESTS PASSED.")


if __name__ == "__main__":
    main()
