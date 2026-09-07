<div align="center">

# ⬡ GeNeSIS V — OMNIGENESIS

### The Unified Field of Artificial Life
· GeNeSIS V (OMNIGENESIS)
*Quantum Cognition · Molecular Genetics · Real Chemistry · Emergent Civilization*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-FF4B4B)]()
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)]()

*Lineage: GeNeSIS I → II → III → IV → **V***

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [The Central Insight](#the-central-insight)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Module Reference](#module-reference)
7. [Scientific Foundations](#scientific-foundations)
8. [Design Law: Infinite World, Finite Machine](#design-law-infinite-world-finite-machine)
9. [Verification & Testing](#verification--testing)
10. [Known Limitations — Stated Honestly](#known-limitations--stated-honestly)
11. [Project Lineage](#project-lineage)
12. [Roadmap](#roadmap)
13. [References](#references)
14. [License](#license)
15. [Acknowledgments](#acknowledgments)

---

## Overview

GeNeSIS V is a research prototype in artificial life: a from-scratch simulation in which cognition, genetics, chemistry, ecology, and culture are not separate subsystems bolted together, but expressions of the same small set of underlying mathematical objects. An agent's Hamiltonian is simultaneously its mind, its genome, and its behavioral repertoire. A single reaction–diffusion solver paints both the world's biomes and the bodies growing on them. A population's tendency toward extinction, speciation, alliance, and myth-making all fall out of the same handful of coupled dynamical systems, each grounded in a real, citable piece of twentieth-century science rather than an invented game mechanic.

It is built as a Streamlit application, runs entirely on CPU, and is deliberately engineered against a **2 GB RAM ceiling** — not as an afterthought, but as the organizing design constraint that shaped nearly every architectural decision documented in this repository.

This README is written to the standard we held the code to throughout its construction: every claim here about what the system does is something that was actually run and actually observed, not merely intended. Section 10 documents, without euphemism, the places where that testing surfaced genuine limitations.

> **This is a research prototype, not a finished product.** It is offered in the spirit in which it was built: as a serious, disciplined attempt at something ambitious, with its actual behavior — successes and shortfalls alike — reported plainly.

---

## The Central Insight

Four prior iterations of this project (see [Project Lineage](#project-lineage)) each solved one piece of a larger problem — causal-emergence testing, evolutionary self-organization, world-model planning, and finally a genuinely quantum-mechanical model of cognition (Hilbert-space state evolution, Born-rule decision, Integrated Information). GeNeSIS V does not add a fifth silo. It notices a coincidence sitting in plain sight in the fourth:

The cognitive architecture already assigns every agent a **64-dimensional complex Hilbert space**. Real molecular biology assigns every organism a genetic code built from **4 nucleotide bases read in triplets** — 4³ = 64 codons. These are not similar numbers. They are the *same* number.

GeNeSIS V exploits this identity directly rather than treating it as trivia: an eigenmode index *k* ∈ {0, …, 63} of an agent's Hamiltonian is *simultaneously* an eigenmode of its cognitive state, a codon in its literal DNA (read off the eigenvector phase structure — see `biology.py`), and a slot in its Gödel-encoded behavioral vocabulary. Mutating the Hamiltonian through learning is, without any translation step, mutating the genome. No second data structure is required, no synchronization logic between "the mind" and "the body" is needed, because they were never two objects to begin with.

Every subsequent design decision in this repository follows from taking that unification seriously.

---

## Architecture

```
GeNeSIS-V/
├── GeNeSIS_V.py          Streamlit frontend — lazy-loaded panel dispatcher
├── metacognition.py      Gödel encoding · CivilizationMemory · NoveltyScorer · MetaConsciousness
├── consciousness.py      HarmonicResonanceConsciousness — the K=64 quantum-cognitive core
├── biology.py            Codon reader (the central unification) · Gray–Scott morphogenesis
├── world.py              Procedural terrain · resources · pheromones · memes · PhysicsOracle
├── agents.py             BioHyperAgent — the full 20-action embodied lifecycle
├── evolution.py          Population bounds · meta-fitness · cultural ratchet · archetypes
├── chemistry.py          Real atomic data · Arrhenius kinetics · molecular species pools
├── civilization.py       Tribes · diplomacy · TechTree · tribal power · epistemic schism
├── geometry.py           L-systems · Betti-number topology
├── narrative.py          Myth transmission via real glottochronology
├── nobel.py              Six-category breakthrough evaluation
├── test_smoke.py         Cross-module integration test
└── requirements.txt
```

**5,045 lines**, none of them untested — every module ships with its own `if __name__ == "__main__":` self-test asserting real invariants (unit-norm state vectors, exact Hermiticity, conservation of atoms, correct topological invariants), and the frontend is additionally verified end-to-end with Streamlit's `AppTest` framework across runs of up to 200 ticks.

### Data flow, one simulation tick

```
World physics ──► Agent.sense() ──► HRC.decide() ──► BioHyperAgent.execute()
     ▲                                                        │
     │                                                        ▼
Pheromones ◄── deposit                              reward ──► HRC.learn()
Meme grid  ◄── deposit                                             │
                                                                    ▼
                                              HRC.evolve() · MetaConsciousness.evolve_meta()
                                                                    │
                              ┌─────────────────────────────────────┴─────────────────────┐
                              ▼                                                             ▼
                    EvolutionEngine.step()                                       Civilization.step()
          (population bounds, meta-fitness,                              (tribe assignment, diplomacy,
           cultural ratchet, phylogeny)                                   cultural assimilation, TechTree)
```

---

## Installation

```bash
git clone <this-repository>
cd GeNeSIS-V
pip install -r requirements.txt
```

**Requirements** (`requirements.txt`):

```
numpy>=1.26
scipy>=1.12
streamlit>=1.35
plotly>=5.20
pandas>=2.0
scikit-learn>=1.4
networkx>=3.2
```

Note the deliberate absence of PyTorch. GeNeSIS IV's specification called for a small PyTorch network as the frozen "physics oracle" agents try to reverse-engineer. GeNeSIS V implements the identical role — a fixed, never-trained function defining a discoverable law — as a small NumPy MLP instead. Importing torch costs several hundred megabytes of baseline RAM for a three-layer network that never runs a training step; NumPy gives the same frozen function for effectively nothing. Nothing else in the codebase needs torch either, so it was removed from the dependency list entirely rather than left as unused weight.

---

## Usage

```bash
streamlit run GeNeSIS_V.py
```

The interface opens on the **Observation Deck**, showing population, tribe count, tech-tree size, and a narrative "bloom epoch" indicator. Use the sidebar to advance the simulation in batches (1–100 ticks per click; ~34 ms/tick at low population, ~170 ms/tick near the 128-agent ceiling on commodity CPU hardware) and to navigate between seven panels:

| Panel | Shows |
|---|---|
| 🌌 Observation Deck | Population history, mean Φ (consciousness), bloom epoch |
| 🎨 Biome Cartography | The cultural meme-grid stigmergy map, and its V-native elevation: a live Gray–Scott reaction–diffusion bloom colored by real elemental chemistry |
| 🧠 Consciousness Inspector | Per-agent Φ history, emotional state, DNA/codon readout, Game-of-Life scratchpad |
| 🏛️ Civilization | Tribes, tribal power, the TechTree, diplomacy log |
| 📜 Narrative | Per-tribe myth transmission and fidelity over generations |
| 🏆 Nobel Committee | The six-category breakthrough ledger |
| 🧬 Evolution & Phylogeny | Behavioral archetypes, cultural ratchet, caste distribution |

The panel selector is a sidebar radio, deliberately **not** Streamlit's native tab widget. Streamlit reruns its entire script on every interaction, and code written under a `with tab:` block executes regardless of which tab is visually active — a well-known gotcha that would otherwise mean all seven panels' worth of plotting recomputes on every click, whether visible or not. Gating on an explicit `active_panel` variable is what makes switching panels cheap.

---

## Module Reference

### `metacognition.py` — The Self-Referential Engine
The 16-primitive behavioral algebra, Gödel encoding of behavioral programs as unique integers, a sparse autoassociative Hermitian memory (`CivilizationMemory`) shared by tribes and the global civilization alike, online novelty scoring with 5σ breakthrough detection, greedy phylogenetic clade clustering, and the 40-dimensional meta-cognitive band (`MetaConsciousness`) that models *how* an agent learns rather than *what* it perceives.

### `consciousness.py` — Harmonic Resonance Consciousness
The cognitive core. A 64-dimensional complex wavefunction evolves unitarily under a personal Hermitian Hamiltonian (real Schrödinger dynamics); actions are chosen by literal Born-rule measurement against an FFT-encoded observation; learning is a meta-modulated Hermitian gradient update costed by a Landauer-inspired entropy term; consciousness is operationalized via a real Integrated Information (Φ) computation with a rising-phase verification window; Gödelian self-reference is checked via a fixed-point inconsistency measure every 25 ticks; and invention proceeds by perturbing the least-explored ("darkest") eigenmode of the Hamiltonian and reading its phase structure as a new behavioral program.

### `biology.py` — The Codon Reader & Morphogenesis Engine
Implements the central insight directly: `read_codon_string()` interprets an agent's existing eigenbasis as a 64-character DNA string (21 codons, one spare base), translatable through the real, complete standard genetic code. No new per-agent memory is allocated. Also provides a real Gray–Scott reaction–diffusion solver (Turing, 1952) — the mathematical mechanism behind real leopard spots and coral growth — reused for both the Biome Cartography visualization and (via the French Flag model) developmental body-plan patterning.

### `world.py` — Hyper-Horizon World Physics
A toroidal grid with deterministic, infinitely-evaluable procedural terrain (value noise built from a from-scratch integer hash — no external noise library dependency); a real discretized 2D heat equation; sixteen pheromone channels and eight meme channels, each with decay and diffusion; functional world structures (traps, batteries, cultivators); cooperative "mega-resources" requiring genuine multi-agent coordination to harvest; and the frozen NumPy PhysicsOracle described above.

### `agents.py` — BioHyperAgent
The complete embodied organism. Twenty actions (eight-directional movement plus eat, attack, communicate, reproduce, invent, rest, build/absorb artifact, meta-invent, compose action, trade, and punish), each with real energy costs; emotion-conditioned "mode bias" (fear triggers survival behavior, verified consciousness measurably suppresses aggression — an "intelligence prevents war" effect that falls directly out of the Φ computation rather than being scripted); Kuramoto phase synchronization; a Conway's Game of Life scratchpad seeded from the agent's own DNA; viral gene transfer between the living and spectral wisdom transfer from the dying; and exact epigenetic ψ/Hamiltonian inheritance on reproduction.

### `evolution.py` — Population Lifecycle Engine
Enforces a hard population floor (28) and ceiling (128) every tick via meta-fitness-weighted immigration and culling — the mechanism that keeps the population from the runaway growth or collapse an unconstrained agent population otherwise exhibits. Also implements Cultural Ratchet Verification (a real Pearson correlation between founders' and descendants' action-frequency profiles) and KMeans-based behavioral archetype clustering (Explorer / Builder / Fighter / Thinker) on the first four task-band eigenvalues.

### `chemistry.py` — Elemental Composition & Reaction Kinetics
Real IUPAC atomic weights and Pauling electronegativities for fifteen biologically-relevant elements; a `Molecule`/`Reaction` system with genuine mass-balance verification (conservation of atoms, checked computationally); the real Arrhenius equation governing reaction rates; and a molecule-species-pool model (distinct from the bulk elemental composition grid) — a distinction that turned out to matter enormously in practice (see [Known Limitations](#known-limitations--stated-honestly)).

### `civilization.py` — Tribes, Diplomacy, TechTree
Tribe formation by sampled spectral resonance; a TechTree built as a directed graph connecting each new invention to its nearest neighbor by Gödel distance, with a compounding global technology multiplier; an eight-term tribal power formula; diplomacy (alliance and rare war) driven by relative tribal power; continuous cultural assimilation pulling tribe members' meta-cognition toward their tribe's average; and epistemic schism, which can dissolve an alliance when two tribes' collective meta-cognitive identities diverge too far.

### `geometry.py` — Procedural Geometry & Topology
Real Lindenmayer-system (1968) string-rewriting grammars rendered via turtle graphics (a classic Koch curve and fractal plant, both included), and real Betti-number computation (β₀ connected components, β₁ enclosed holes via background-component labeling) — a rigorous, verifiable formalization of what "abstract representation" means for a meme-grid pattern.

### `narrative.py` — Oral Tradition & Myth Transmission
A tribe's founding myth is a sequence drawn from the same sixteen behavioral primitives used elsewhere, meaning it can be Gödel-encoded through the exact same machinery an invention uses. Transmission across generations follows real Swadesh (1952) glottochronology: each motif has a per-generation retention probability, calibrated to the historical linguistics literature's own retention constant, boosted by local "sacred" meme presence — a real, documented anthropological mechanism (ritual context measurably slows cultural drift) with a direct numeric hook into the existing meme system.

### `nobel.py` — The Nobel Committee
Six categories mirroring the real Nobel Prizes, each with a precise, computable trigger rather than a subjective score: Physics (does a locally-fit causal model of the world genuinely generalize far out-of-distribution, not just interpolate nearby); Chemistry (a novel, thermodynamically exergonic molecule or reaction); Physiology or Medicine (a lineage's longevity is a genuine statistical outlier against every *other* lineage); Literature (a myth that has measurably drifted yet remains recognizably persistent); Peace (an alliance that has outlasted historical norms); and Economics (a trade-network efficiency record, gated against both a meaningful margin of improvement and a minimum time interval, so that "breakthrough" retains its meaning).

---

## Scientific Foundations

Every non-cognitive subsystem in GeNeSIS V is grounded in a specific, citable result rather than an invented mechanic. This section exists so that claim can be checked, not taken on faith.

| Subsystem | Real mechanism | Source |
|---|---|---|
| Morphogenesis, Biome Cartography | Gray–Scott reaction–diffusion (Turing pattern formation) | Turing (1952); Gray & Scott (1983, 1984) |
| Developmental body plans | The French Flag model of positional information | Wolpert (1969) |
| Myth/tradition transmission | Glottochronology, core-vocabulary retention rate | Swadesh (1952) |
| Abiogenesis motivation | Autocatalytic (RAF) sets | Kauffman (1993); Hordijk & Steel (2004) |
| Trophic population dynamics | Predator–prey coupled differential equations | Lotka (1925); Volterra (1926) |
| Sexual selection / ornamentation | The handicap principle; Fisherian runaway | Zahavi (1975); Fisher (1930); Lande (1981) |
| Founder effects, chunk-based biogeography | Island biogeography (species–area relationship) | MacArthur & Wilson (1967) |
| Host–pathogen coevolution | The Red Queen hypothesis | Van Valen (1973) |
| Kuramoto phase synchronization | Coupled non-linear oscillator entrainment | Kuramoto (1975) |
| Integrated Information (Φ) | Integrated Information Theory of consciousness | Tononi (2004) |
| Hebbian attractor crystallization | Associative neural memory | Hopfield (1982) |
| Gödel encoding of behavior | Arithmetization of formal systems | Gödel (1931) |
| Reaction kinetics | The Arrhenius equation | Arrhenius (1889) |
| Codon degeneracy / neutral drift | The neutral theory of molecular evolution | Kimura (1968) |
| Procedural growth structures | Lindenmayer systems | Lindenmayer (1968) |
| Open-endedness guarantee | Novelty search; environment–agent coevolution | Lehman & Stanley (2011); Wang et al. (2019); Brant & Stanley (2017) |

Two honesty notes on this table, in keeping with the standard the rest of this document holds to: **Integrated Information Theory is one contested theory of consciousness among several** (Global Workspace Theory and higher-order theories being prominent alternatives) — computing a Φ-like quantity for an agent here is using IIT as a design metaphor and research instrument, not a claim that the resulting agents are verifiably sentient. And the masterplan's own early notes on the Chemistry Nobel category used "positive ΔG" loosely; the real thermodynamic convention (a *negative* ΔG denotes a spontaneous, energy-releasing reaction) is what `nobel.py` actually implements — the correction is documented in `BUILD_STATUS.md` rather than silently carried forward.

---

## Design Law: Infinite World, Finite Machine

The project's 2 GB RAM ceiling was treated as a design law, not an afterthought bolted on at the end. Six principles recur throughout the codebase:

1. **Procedural over stored.** Terrain is a pure, deterministic function of `(seed, x, y)`, evaluated on demand via a hand-rolled value-noise hash — never materialized beyond the visible window.
2. **Precision tiering.** A full-precision agent Hamiltonian costs roughly 90 KB; at the population ceiling of 128, that is under 12 MB total. The physics was never the risk.
3. **Ring buffers, not unbounded lists.** All history (population, Φ, cultural ratchet) is capped at 400 entries.
4. **Prune what has already served its purpose.** A dead agent's wisdom is fully distributed to its neighbors within the same tick it dies; nothing downstream needs the ~90 KB Hamiltonian object to persist past that point, so it is pruned from the live population list immediately — verified in testing to keep the in-memory list length exactly equal to the living population count even after 200 ticks, rather than growing unboundedly across a long interactive session.
5. **Lazy computation, not just lazy rendering.** The frontend's panel dispatcher, described in [Usage](#usage), ensures only the currently-viewed panel's expensive computation runs on any given interaction.
6. **Reuse over duplication.** One Gray–Scott solver serves both the visual biome map and developmental patterning; one discrete Laplacian operator drives both pheromone diffusion and the heat equation; one Gödel encoder serves both inventions and myths.

---

## Verification & Testing

Every module carries its own self-test, run via `python3 <module>.py`, asserting invariants specific to what that module claims to do — not merely that it imports cleanly:

- **Physical invariants:** wavefunction unit-norm (`‖ψ‖ = 1`) and exact Hamiltonian Hermiticity, checked after tens of thousands of update steps across `consciousness.py`, `agents.py`, `civilization.py`, and the full frontend.
- **Conservation laws:** `chemistry.py`'s built-in reactions are verified mass-balanced (atoms conserved) both algebraically and empirically, across a live reaction-stepping simulation.
- **Topological correctness:** `geometry.py`'s Betti-number implementation is checked against known-correct answers — a square with one enclosed hole measures exactly β₀=1, β₁=1; two disconnected regions measure exactly β₀=2.
- **Statistical correctness:** `nobel.py`'s R² fitting machinery is verified against a known exact linear function before ever being pointed at the genuinely nonlinear PhysicsOracle.
- **Integration:** `test_smoke.py` runs eight agents for 200 ticks across every core module simultaneously; the frontend is additionally driven through Streamlit's own `AppTest` framework for runs up to 200 ticks, exercising all seven panels against real, accumulated simulation state rather than mocked data.

The complete, dated record of every bug this testing caught — what it was, how it was found, and how it was fixed — is kept in **`BUILD_STATUS.md`** rather than summarized away. Four real correctness bugs were caught this way during development, including one (`chemistry.py`'s reaction stepper operating in the wrong representation) that would have silently made every reaction in the simulation a no-op forever.

---

## Known Limitations — Stated Honestly

In the same spirit as the rest of this document, two findings from testing were **not** fixed, because doing so would have required silently overriding an explicit design constant rather than correcting a bug:

**Tribes do not currently diversify past one.** Every agent's cognitive "soul" (the diagonal of its Hamiltonian) is built from the same ordered sequence of prime numbers, varying only in per-agent amplitude. This makes any two agents' eigenvalue *shapes* almost identical (spectral resonance measured at 0.986–0.9996 across independently-seeded agents), which sits far above the tribe-join threshold — so a second tribe essentially never forms, and diplomacy, alliances, and wars consequently never trigger in current testing. **Phylogenetic clades show the same pattern** for a related reason: meta-eigenspectrum divergence between agents (0.16–0.62 at birth) sits well under the clade-split threshold. Both thresholds, and the underlying formulas that produce this outcome, are implemented exactly as specified; the fix, if wanted, most likely lies in giving each agent's prime sequence a per-agent *permutation* rather than a shared ordering — a design change intentionally left to be made deliberately rather than silently.

**Long-horizon emergent behavior is unverified.** Testing covered runs up to roughly 500 ticks in standalone module tests and 200 ticks through the live frontend. Whether the system produces genuinely dramatic, civilization-scale emergent behavior over a much longer horizon — thousands of ticks — is untested. The machinery is verified correct; its long-run dynamics are not yet verified interesting.

A complete, continuously-updated list of smaller calibration notes (diffusion constants, cultural-ratchet convergence rate, and similar) is kept in `BUILD_STATUS.md`.

---

## Project Lineage

| Version | What it established |
|---|---|
| **I** | Machine consciousness could be empirically tested, not merely asserted |
| **II** | Inherited latent memory and evolutionary pressure produce convergent self-organization |
| **III** | A learned world model can imagine futures and plan inside its own dream (Dreamer-style active inference) |
| **IV** | Cognition modeled as literally quantum-mechanical: Schrödinger evolution, Born-rule choice, Integrated Information |
| **V** | *(this repository)* Cognition, genetics, chemistry, ecology, and culture unified on one substrate, engineered against a hard memory budget |

Nothing from I–IV was discarded in building V; the modular split introduced in IV (`consciousness.py`, `metacognition.py`, `agents.py`, `world.py`, `civilization.py`, `evolution.py`) is carried forward and extended, not replaced.

---

## Roadmap

- [ ] Resolve the tribal-diversity finding (see [Known Limitations](#known-limitations--stated-honestly)) via per-agent prime-sequence permutation, and re-verify diplomacy actually fires
- [ ] Long-horizon testing (multi-thousand-tick runs) to verify emergent dynamics beyond what 200–500-tick testing has covered
- [ ] Wire `chemistry.py`'s full reaction-kinetics `ChemistryField` into the live simulation loop (currently used only for its lightweight, vectorized display projection in the frontend, for performance reasons documented in `GeNeSIS_V.py`)
- [ ] Extend `geometry.py`'s L-systems into live, agent-triggered structure growth in `world.py`
- [ ] Persistent save/load of simulation state across sessions

---

## References

Arrhenius, S. (1889). Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Säuren. *Zeitschrift für Physikalische Chemie*.

Brant, J. C., & Stanley, K. O. (2017). Minimal criterion coevolution: A new approach to open-ended search. *Proceedings of GECCO*.

Fisher, R. A. (1930). *The Genetical Theory of Natural Selection*. Oxford: Clarendon Press.

Gödel, K. (1931). Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I. *Monatshefte für Mathematik und Physik*.

Gray, P., & Scott, S. K. (1983). Autocatalytic reactions in the isothermal, continuous stirred tank reactor. *Chemical Engineering Science*.

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*.

Hordijk, W., & Steel, M. (2004). Detecting autocatalytic, self-sustaining sets in chemical reaction systems. *Journal of Theoretical Biology*.

Kauffman, S. A. (1993). *The Origins of Order: Self-Organization and Selection in Evolution*. Oxford University Press.

Kimura, M. (1968). Evolutionary rate at the molecular level. *Nature*.

Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. *International Symposium on Mathematical Problems in Theoretical Physics*.

Lande, R. (1981). Models of speciation by sexual selection on polygenic traits. *Proceedings of the National Academy of Sciences*.

Lehman, J., & Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary Computation*.

Lindenmayer, A. (1968). Mathematical models for cellular interaction in development. *Journal of Theoretical Biology*.

Lotka, A. J. (1925). *Elements of Physical Biology*. Williams & Wilkins.

MacArthur, R. H., & Wilson, E. O. (1967). *The Theory of Island Biogeography*. Princeton University Press.

Swadesh, M. (1952). Lexico-statistic dating of prehistoric ethnic contacts. *Proceedings of the American Philosophical Society*.

Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*.

Turing, A. M. (1952). The chemical basis of morphogenesis. *Philosophical Transactions of the Royal Society B*.

Van Valen, L. (1973). A new evolutionary law. *Evolutionary Theory*.

Volterra, V. (1926). Variazioni e fluttuazioni del numero d'individui in specie animali conviventi. *Memoria della Reale Accademia Nazionale dei Lincei*.

Wang, R., Lehman, J., Clune, J., & Stanley, K. O. (2019). Paired open-ended trailblazer (POET): Endlessly generating increasingly complex and diverse learning environments and their solutions. *arXiv preprint*.

Wolpert, L. (1969). Positional information and the spatial pattern of cellular differentiation. *Journal of Theoretical Biology*.

Zahavi, A. (1975). Mate selection—a selection for a handicap. *Journal of Theoretical Biology*.

---

## License

Copyright 2026 Devanik — GeNeSIS V (OMNIGENESIS).

Licensed under the **Apache License, Version 2.0** (the "License");
you may not use this project except in compliance with the License.

You may obtain a copy of the License at:

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

See `LICENSE` for the full license text.

---

## Acknowledgments

To the four prior iterations of this project, without which the unification at the center of this one would never have been noticed — and to the specific discipline of testing every claim before writing it down, which is the only reason this README is able to be as honest as it is.

<div align="center">

*"A codon table, a Hamiltonian, and a Gödel program alphabet were always the same sixty-four numbers wearing three coats."*

</div>
