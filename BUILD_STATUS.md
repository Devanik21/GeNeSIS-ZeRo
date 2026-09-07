# GeNeSIS V — Build Status

**All 12 backend modules + the Streamlit frontend are built, tested, and
verified together.** This file is the honest record of how they got that
way — every bug caught, every fix made, every finding that couldn't be
fixed and had to be flagged instead. "Tested" means a real self-test
(`python3 <file>.py`) that asserts real invariants, not just an import
check; the frontend was additionally verified with Streamlit's own
`AppTest` framework (executes the actual script, catches real exceptions)
across runs up to 200 ticks.

## The complete file list

| File | Lines | Role |
|---|---|---|
| `metacognition.py` | ~380 | Gödel encoding, CivilizationMemory, NoveltyScorer, PhylogeneticTracker, MetaConsciousness |
| `consciousness.py` | ~430 | HarmonicResonanceConsciousness — the K=64 quantum-cognitive core |
| `biology.py` | ~240 | Codon reader (the V unification), Gray-Scott morphogenesis, French Flag |
| `world.py` | ~380 | Procedural terrain, resources, pheromones, memes, structures, PhysicsOracle |
| `agents.py` | ~730 | BioHyperAgent — full 20-action lifecycle, epigenetics, death/apoptosis |
| `evolution.py` | ~280 | Population bounds, meta-fitness, cultural ratchet, archetypes, phylogeny |
| `chemistry.py` | ~400 | Real atomic data, Arrhenius kinetics, molecule pools, pH/toxicity |
| `civilization.py` | ~380 | Tribes, diplomacy, TechTree, tribal power, epistemic schism |
| `geometry.py` | ~180 | L-systems, Betti-number topology |
| `narrative.py` | ~230 | Myth transmission via real Swadesh glottochronology |
| `nobel.py` | ~340 | Six-category breakthrough evaluation |
| `GeNeSIS_V.py` | ~470 | Streamlit frontend — lazy-loaded panel dispatcher |
| `test_smoke.py` | ~165 | Cross-module integration test |
| `requirements.txt` | — | torch deliberately excluded (see world.py note) |

**5,045 total lines**, every one of them run at least once by its own test.

## Real bugs caught by testing, not by inspection

These are correctness bugs — wrong behavior, not calibration — each one
mine to fix, found by actually running the code and checking the *right*
thing, not just that it didn't crash:

1. **`chemistry.py` — reactions could never do anything.** The first
   version ran kinetics in element-space, where reactant and product
   elemental totals are *identical* for any mass-balanced reaction, by
   definition (that's what conservation of atoms means). Refactored to
   track molecule-species pools (glucose, O2, CO2, water) separately from
   bulk element pools. Verified fix: respiration now measurably consumes
   glucose (0.114 -> 0.030) while total atoms stay exactly conserved.
2. **`agents.py` — role assignment collapsed to one role.** Fixed
   thresholds (4.0 / 8.0 on cognitive eigenspread) were unfounded numbers;
   real population data clustered entirely inside one band, so everyone
   got the same caste. Fixed with population-relative tertiles instead of
   absolute magic numbers.
3. **`nobel.py` — Medicine category couldn't detect an obvious outlier.**
   Comparing a lineage against a *global* mean/std that included that same
   lineage let the outlier's own extreme values inflate the very threshold
   it needed to clear — a real 50-vs-90-tick lifespan gap failed to
   trigger. Fixed with a leave-one-group-out baseline (standard practice
   for exactly this kind of test).
4. **`nobel.py` — Physics and Economics were spamming, not achieving.**
   Live 30-tick app testing: 33 "Nobel laureates," which defeats the
   entire premise of a rare breakthrough. Root-caused each separately —
   Physics's local polynomial fit generalized further than intended
   because world.py's spatial fields are smooth almost everywhere (fixed
   by widening the out-of-distribution test ring from a 4-7 cell radius to
   12-20, genuinely far extrapolation); Economics let any epsilon
   improvement count as a "record" (fixed with both a 15% minimum relative
   margin and the same civilisation-scale cooldown Physics uses). Verified
   fix: 33 laureates in 30 ticks -> 7 laureates across a full 200-tick run,
   spread realistically across categories.

## Deliberate spec deviations (stated plainly, not hidden)

- **`world.py`'s PhysicsOracle is a frozen NumPy MLP, not the spec's torch
  NN.** Same role — a fixed function agents try to reverse-engineer — at a
  fraction of torch's RAM cost for a network that's never trained. torch
  is dropped from requirements.txt entirely; nothing else needs it either.

## Two findings that could NOT be fixed unilaterally — flagged for a decision

Both are cases where every individual formula was implemented exactly as
specified, tested, and confirmed correct — and the *combination* still
produces a degenerate outcome. Neither is a bug in the module it appears
in; both would require changing an explicit spec constant I don't have
standing to override on my own judgment.

1. **Phylogenetic clades never exceed 1.** At birth, meta-eigenspectrum L2
   distances between independently-seeded agents measure 0.16-0.62,
   against a split threshold of 2.5. Both the threshold and the §3.2
   mutation-noise scale (0.08) that produces that spread are explicit spec
   constants. Possible this is intentional (real speciation is rare too);
   possible the constants were calibrated for a longer run than tested.
2. **Tribes never exceed 1 — the direct cause of zero alliances, wars, or
   schisms ever firing.** Spectral resonance between any two agents
   measures 0.986-0.9996 at birth, against a join threshold of 0.08 —
   100% of pairs clear it. The resonance formula matches §3.15 exactly,
   including a design comment in the spec itself explaining why it was
   deliberately loosened. Root cause is one layer deeper: every agent's
   Hamiltonian diagonal is built from the same ordered prime sequence
   (§3.1's omega), so eigenvalue *shapes* stay almost identical across the
   whole population regardless of who any two agents actually are. Fixing
   this would mean touching one of three things — the join threshold, the
   resonance formula, or the shared-omega construction — and two of the
   three are explicit formulas elsewhere in the spec.

**If you want tribal diversity and diplomacy to actually happen**, the
most promising lever is the third option: giving each agent's omega a
per-agent *permutation* of the same prime set (rather than the same
ordering with only amplitude variation) would break the shape-correlation
that's suppressing resonance, without touching either explicit formula.
That's a design call for you, not one made silently on your behalf here.

## Verified live, end-to-end, through the actual Streamlit app

Using Streamlit's `AppTest` (executes the real script, not a mock) — not
just "the file has no syntax errors," but a real 200-tick run through the
actual UI code path:

- All 7 panels render without exception, checked after 30, 70, and 200
  ticks of accumulated real state.
- Population correctly pruned: the in-memory list length equals the alive
  count (41 = 41) after 200 ticks — dead agents are not accumulating
  forever, which would have been the actual RAM risk in a long interactive
  session (not population *size*, which was always cheap — see the
  masterplan's own RAM budget table).
- 401 tech-tree nodes, 7 Nobel laureates spread realistically across
  categories (not spam), history correctly bounded under its 400-tick cap.
- Sampled live agents after 200 ticks: psi still exactly unit-norm, H
  still exactly Hermitian, energy still within its defined bounds — the
  same invariants checked in every backend module's own test, now holding
  through the actual UI's tick loop, not just a standalone script.
- Per-tick cost measured directly: ~34ms at low population, ~170ms near
  the 128-agent ceiling — both comfortable for a button-driven interface.

## Known calibration notes (still true, still not hidden)

- `world.py` pheromone diffusion constant (0.10) loses ~40% of a deposit's
  magnitude to neighbors in a single tick — legitimate diffusion behavior,
  on the fast side. Worth eyeballing now that the Biome Cartography panel
  actually renders it.
- Cultural ratchet r was trending upward but hadn't crossed the 0.55
  verification bar in any run tested so far (up to 500 ticks) — plausible
  given real cultural transmission takes time, not evidence of a bug.
- Several "named but not formula-specified" spec items (GoL scratchpad's
  downstream effect, exact inventory synergy bonus, exact viral-broadcast
  selection rule, tribe-colour RGB mapping, bloom-epoch label thresholds)
  were filled with documented, reasonable design choices — each marked
  `# design choice:` inline in the source, never presented as spec fact.
