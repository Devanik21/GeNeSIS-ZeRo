"""
GeNeSIS_V.py — OMNIGENESIS: The Unified Field of Artificial Life
======================================================================

The Streamlit frontend. Ties together all eleven backend modules built
across this project: metacognition, consciousness, biology, world,
agents, evolution, chemistry, civilization, geometry, narrative, nobel.

Lazy-loading architecture (masterplan §6, honouring the 2GB RAM law in
§2): a sidebar radio — not st.tabs — decides which panel's EXPENSIVE
body actually executes on any given rerun. Plain st.tabs() does not
give you this for free: Streamlit reruns the whole script top-to-bottom
on every interaction, and code written under `with tab:` executes
regardless of which tab is visually active. Gating on an explicit
`active_panel` variable is what makes switching panels not re-run the
whole civilization's worth of plotting on every click.

Run with:  streamlit run GeNeSIS_V.py
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from metacognition import ALL_PRIMITIVES, K_DIM, PhylogeneticTracker
from consciousness import EMOTION_NAMES
from biology import (
    gray_scott_step, init_grid, biome_rgb, genome_fingerprint, codon_split, translate,
    genetic_distance, french_flag_zones, STANDARD_GENETIC_CODE,
)
from world import GenesisWorld, MEME_NAMES, PHEROMONE_NAMES, RESOURCE_NAMES
from agents import (
    BioHyperAgent, make_child, update_roles, spectral_resonance,
    MEME_ABSORPTION_EVERY, VIRAL_BROADCAST_EVERY, ACTION_NAMES,
)
from evolution import EvolutionEngine, POP_INITIAL, POP_FLOOR, POP_CEILING
from civilization import Civilization
from narrative import (
    NarrativeEngine, glottochronology_divergence_time, expected_retention_after,
    GLOTTOCHRONOLOGY_RETENTION_RATE,
)
from nobel import NobelCommittee
from chemistry import (
    RESOURCE_ELEMENT_PROFILE, ELEMENTS, N_ELEMENTS, ATOMIC_WEIGHT, ELECTRONEGATIVITY,
    GLUCOSE, WATER, CO2, O2, RESPIRATION, PHOTOSYNTHESIS,
    arrhenius_rate_constant, heat_to_kelvin,
)
from geometry import (
    betti_0, betti_1, topology_summary, KOCH_CURVE, FRACTAL_PLANT,
    turtle_render, bounding_box,
)

# ----------------------------------------------------------------------------
# Page config & constants
# ----------------------------------------------------------------------------
st.set_page_config(page_title="GeNeSIS V — OMNIGENESIS", page_icon="⬡", layout="wide")

WORLD_SIZE: int = 56          # balances visual richness against the 2GB budget
HISTORY_CAP: int = 400        # ring-buffer cap (masterplan §2.5)
GRAYSCOTT_STEPS_PER_TICK: int = 3   # morphogen field advances in lockstep with sim time
TECH_TREE_RENDER_LIMIT: int = 60    # render only the most recent N nodes — a tree with
# thousands of nodes (observed after long runs in earlier testing) would make a
# force-directed layout unreadable and slow; the full graph stays intact internally.

PANELS: List[str] = [
    "🌌 Observation Deck",
    "🎨 Biome Cartography",
    "🧠 Consciousness Inspector",
    "🏛️ Civilization",
    "📜 Narrative",
    "🏆 Nobel Committee",
    "🧬 Evolution & Phylogeny",
    "🧪 Chemistry Lab",
    "🌀 Geometry & Topology",
]


# ----------------------------------------------------------------------------
# Chemistry bridge (vectorised — the ChemistryField class in chemistry.py
# does a real per-cell Python loop for its full reaction-kinetics role,
# which is the right tool there; this is a lighter, display-only projection
# that never needs a Python loop over the grid)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _resource_profile_matrix(resource_names: tuple) -> np.ndarray:
    M = np.zeros((len(resource_names), N_ELEMENTS))
    for i, rname in enumerate(resource_names):
        for el, frac in RESOURCE_ELEMENT_PROFILE.get(rname, {}).items():
            M[i, ELEMENTS.index(el)] = frac
    return M


def dominant_element_hue_map(world: GenesisWorld) -> np.ndarray:
    """Fast, vectorised 'which element dominates this cell' field in [0, 1],
    built from world.py's resource grid and chemistry.py's real element
    profile table — without instantiating a full ChemistryField."""
    M = _resource_profile_matrix(tuple(RESOURCE_NAMES))
    element_field = world.resource_grid.astype(np.float64) @ M
    dominant_idx = np.argmax(element_field, axis=-1).astype(np.float64)
    return dominant_idx / max(N_ELEMENTS - 1, 1)


def bloom_epoch_label(tick: int, n_tribes: int, tech_nodes: int, mean_fidelity: Optional[float]) -> str:
    """A DISPLAY heuristic (masterplan §19), not a rigorous trigger —
    narrative flavour for the Observation Deck, explicitly labelled as such
    in the UI rather than presented as a precise measurement."""
    if tech_nodes == 0:
        return "0 · Primordial Soup"
    if tech_nodes < 10:
        return "1 · First Replicators"
    if n_tribes <= 1 and tech_nodes < 150:
        return "2 · Colonization"
    if n_tribes <= 1:
        return "3 · Cambrian Bloom"
    if mean_fidelity is not None and mean_fidelity > 0.5:
        return "5 · Civilization"
    return "4 · Coevolutionary Steady-State"


# ----------------------------------------------------------------------------
# Simulation state
# ----------------------------------------------------------------------------
def init_state(seed: int = 42) -> None:
    world = GenesisWorld(height=WORLD_SIZE, width=WORLD_SIZE, seed=seed)
    rng = np.random.default_rng(seed)
    population = [
        BioHyperAgent(agent_id=i, x=int(rng.integers(0, WORLD_SIZE)), y=int(rng.integers(0, WORLD_SIZE)), seed=1000 + i)
        for i in range(POP_INITIAL)
    ]
    A0, B0 = init_grid(WORLD_SIZE, WORLD_SIZE, seed=seed)

    st.session_state.seed = seed
    st.session_state.world = world
    st.session_state.population = population
    st.session_state.next_agent_id = 100000
    st.session_state.tick = 0
    st.session_state.rng = rng
    st.session_state.evo = EvolutionEngine()
    st.session_state.civ = Civilization()
    st.session_state.narrative = NarrativeEngine()
    st.session_state.nobel = NobelCommittee()
    st.session_state.morphogen_A = A0
    st.session_state.morphogen_B = B0
    st.session_state.history: Dict[str, list] = {
        "tick": [], "population": [], "tribes": [], "tech_nodes": [],
        "b_tech": [], "mean_phi": [], "cultural_ratchet": [],
    }
    st.session_state.active_panel = PANELS[0]
    st.session_state.initialized = True


def _append_history(key: str, value: float) -> None:
    st.session_state.history[key].append(value)
    if len(st.session_state.history[key]) > HISTORY_CAP:
        st.session_state.history[key] = st.session_state.history[key][-HISTORY_CAP:]


def run_ticks(n: int) -> None:
    world = st.session_state.world
    population: List[BioHyperAgent] = st.session_state.population
    evo: EvolutionEngine = st.session_state.evo
    civ: Civilization = st.session_state.civ
    narrative: NarrativeEngine = st.session_state.narrative
    nobel: NobelCommittee = st.session_state.nobel
    rng: np.random.Generator = st.session_state.rng

    for _ in range(n):
        tick = st.session_state.tick
        alive_now = [p for p in population if p.alive]
        world.population_density = len(alive_now) / (world.height * world.width)
        world.step()

        for agent in list(population):
            if not agent.alive:
                continue
            agent._pending_reproduction_partner = None
            tribe = civ.tribes.get(agent.tribe_id) if agent.tribe_id is not None else None
            civ_mem = tribe.memory if tribe is not None else civ.global_memory
            agent.step(world, population, tick, novelty_scorer=civ.novelty_scorer, civ_memory=civ_mem)

            if agent.discoveries:
                last_program, last_godel, _ = agent.discoveries[-1]
                if getattr(agent, "_last_registered_godel", None) != last_godel:
                    civ.register_invention(agent, last_program, last_godel, tick)
                    agent._last_registered_godel = last_godel
                    nobel.check_physics(agent.x, agent.y, world, rng, agent.agent_id, tick)

            if getattr(agent, "_pending_reproduction_partner", None) is not None:
                partner = agent._pending_reproduction_partner
                child = make_child(agent, partner, child_id=st.session_state.next_agent_id, world=world)
                population.append(child)
                st.session_state.next_agent_id += 1

            if not agent.alive:
                packet = agent.death_packet()
                nobel.death_log.record(agent.tribe_id if agent.tribe_id is not None else -1, agent.age)
                for other in agent._others_in_radius(population, radius=3):
                    other.absorb_spectral_wisdom(packet)

        # RAM discipline (masterplan §2.5): a dead agent's wisdom has
        # already been fully distributed by this point in the same tick —
        # nothing downstream needs the ~90KB Hamiltonian object to persist,
        # only the lightweight records (death log, tribe counts, tech tree)
        # already extracted above. Keeping dead agents in this list forever
        # is exactly the unbounded-growth failure mode the whole memory
        # design law exists to prevent in a long-running interactive session.
        population[:] = [p for p in population if p.alive]

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

        _, next_id = evo.step(population, world, tick, st.session_state.next_agent_id)
        st.session_state.next_agent_id = next_id
        civ.step(population, rng, tick)

        for tribe_id, tribe in list(civ.tribes.items()):
            members = tribe.members(population)
            if not members:
                continue
            if tribe_id not in narrative.traditions:
                narrative.found_tradition(tribe_id, generation=members[0].generation, rng=rng)
            else:
                gen = max(m.generation for m in members)
                sacred_idx = MEME_NAMES.index("sacred")
                sacred_level = float(np.mean([world.meme_grid[m.y, m.x, sacred_idx] for m in members]))
                narrative.retell(tribe_id, gen, sacred_level, rng)
            nobel.check_literature(narrative, tick)

        nobel.check_medicine(tick)
        nobel.check_peace(civ, tick)
        nobel.check_economics(population, tick)

        st.session_state.morphogen_A, st.session_state.morphogen_B = gray_scott_step(
            st.session_state.morphogen_A, st.session_state.morphogen_B
        )
        for _ in range(GRAYSCOTT_STEPS_PER_TICK - 1):
            st.session_state.morphogen_A, st.session_state.morphogen_B = gray_scott_step(
                st.session_state.morphogen_A, st.session_state.morphogen_B
            )

        alive_count = len([p for p in population if p.alive])
        phi_values = [p.hrc.phi_history[-1] for p in population if p.alive and p.hrc.phi_history]
        mean_phi = float(np.mean(phi_values)) if phi_values else 0.0
        r = evo.cultural_ratchet_history[-1] if evo.cultural_ratchet_history else np.nan

        _append_history("tick", tick)
        _append_history("population", alive_count)
        _append_history("tribes", len([t for t in civ.tribes.values() if t.members(population)]))
        _append_history("tech_nodes", civ.tech_tree.number_of_nodes())
        _append_history("b_tech", civ.b_tech)
        _append_history("mean_phi", mean_phi)
        _append_history("cultural_ratchet", r)

        st.session_state.tick += 1


# ----------------------------------------------------------------------------
# Panel renderers
# ----------------------------------------------------------------------------
def _bar(x, y, title=None, color_seq=None, color=None, color_map=None,
         color_scale=None, xtitle=None, ytitle=None, angle=None):
    """px.bar wrapper: plotly 7 rejects two bare lists for x and y (it reads
    them as column references), so build an explicit DataFrame first."""
    d = pd.DataFrame({"x": list(x), "y": list(y)})
    kw = {}
    if color_seq is not None:
        kw["color_discrete_sequence"] = color_seq
    if color is not None:
        d["c"] = list(color)
        kw["color"] = "c"
        if color_map is not None:
            kw["color_discrete_map"] = color_map
        if color_scale is not None:
            kw["color_continuous_scale"] = color_scale
    f = px.bar(d, x="x", y="y", **kw)
    f.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle,
                    showlegend=False, coloraxis_showscale=False)
    if angle is not None:
        f.update_layout(xaxis_tickangle=angle)
    return f


def _pie(names, values, title=None):
    d = pd.DataFrame({"n": list(names), "v": list(values)})
    f = px.pie(d, names="n", values="v")
    f.update_layout(title=title)
    return f


def _hist_df() -> pd.DataFrame:
    return pd.DataFrame(st.session_state.history)


def _alive() -> List[BioHyperAgent]:
    return [p for p in st.session_state.population if p.alive]


def _small(fig, h: int = 250):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=34, b=8))
    return fig


# ----------------------------------------------------------------------------
# PANEL 1 — Observation Deck
# ----------------------------------------------------------------------------
def render_observation_deck() -> None:
    world = st.session_state.world
    population = st.session_state.population
    civ = st.session_state.civ
    hist = st.session_state.history
    alive = _alive()

    n_tribes = len([t for t in civ.tribes.values() if t.members(population)])
    traditions = st.session_state.narrative.traditions
    mean_fid = float(np.mean([t.fidelity() for t in traditions.values()])) if traditions else None
    epoch = bloom_epoch_label(st.session_state.tick, n_tribes, civ.tech_tree.number_of_nodes(), mean_fid)

    st.subheader(f"Bloom Epoch: {epoch}")
    st.caption("A narrative display heuristic, not a precise measurement — see masterplan §19.")

    c = st.columns(6)
    c[0].metric("Tick", st.session_state.tick)
    c[1].metric("Population", len(alive), help=f"Bounded to [{POP_FLOOR}, {POP_CEILING}]")
    c[2].metric("Tribes", n_tribes)
    c[3].metric("Tech nodes", civ.tech_tree.number_of_nodes())
    c[4].metric("B_tech", f"{civ.b_tech:.3f}", help="Technology multiplier, capped at 3.0")
    c[5].metric("Laureates", len(st.session_state.nobel.laureates))

    st.markdown("#### World state")
    w = st.columns(6)
    w[0].metric("Season", world.season_name.capitalize(), f"{world.season_phase:+.2f}")
    w[1].metric("Weather amp.", f"{world.weather_amplitude:.3f}")
    w[2].metric("Structures", len(world.structures))
    w[3].metric("Mega-resources", len(world.mega_resources))
    w[4].metric("Mean temp.", f"{world.heat_field.mean():.3f}")
    w[5].metric("Mean resource", f"{world.resource_grid.mean():.3f}")

    if alive:
        st.markdown("#### Vital signs")
        v = st.columns(6)
        v[0].metric("Mean energy", f"{np.mean([p.energy for p in alive]):.3f}")
        v[1].metric("Mean health", f"{np.mean([p.health for p in alive]):.3f}")
        v[2].metric("Mean age", f"{np.mean([p.age for p in alive]):.1f}")
        v[3].metric("Max generation", max(p.generation for p in alive))
        v[4].metric("Total discoveries", sum(len(p.discoveries) for p in alive))
        v[5].metric("Kuramoto r", f"{_order_parameter(alive):.3f}", help="Global phase synchrony")

    if len(hist["tick"]) >= 2:
        df = _hist_df()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["tick"], y=df["population"], name="Population",
                                 line=dict(color="#4fd1c5", width=2)))
        fig.add_hline(y=POP_FLOOR, line_dash="dot", line_color="#e53e3e", annotation_text="floor")
        fig.add_hline(y=POP_CEILING, line_dash="dot", line_color="#e53e3e", annotation_text="ceiling")
        fig.update_layout(title="Population trajectory", height=330, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        with a:
            f2 = go.Figure([go.Scatter(x=df["tick"], y=df["mean_phi"], line=dict(color="#f6ad55"))])
            f2.update_layout(title="Mean IIT Φ")
            st.plotly_chart(_small(f2, 260), width="stretch")
        with b:
            f3 = go.Figure([go.Scatter(x=df["tick"], y=df["b_tech"], line=dict(color="#9f7aea"))])
            f3.update_layout(title="Technology multiplier B_tech")
            st.plotly_chart(_small(f3, 260), width="stretch")

        a, b = st.columns(2)
        with a:
            f4 = go.Figure([go.Scatter(x=df["tick"], y=df["tech_nodes"], line=dict(color="#63b3ed"))])
            f4.update_layout(title="Cumulative inventions (tech tree size)")
            st.plotly_chart(_small(f4, 250), width="stretch")
        with b:
            growth = np.diff(np.asarray(hist["population"], dtype=float))
            f5 = go.Figure([go.Bar(x=df["tick"][1:], y=growth, marker_color="#4fd1c5")])
            f5.add_hline(y=0, line_color="#666")
            f5.update_layout(title="Net population change per tick")
            st.plotly_chart(_small(f5, 250), width="stretch")

    if alive:
        st.markdown("#### Live population distributions")
        d = st.columns(4)
        for col, (vals, title, color) in zip(d, [
            ([p.age for p in alive], "Age", "#4fd1c5"),
            ([p.energy for p in alive], "Energy", "#f6ad55"),
            ([p.health for p in alive], "Health", "#48bb78"),
            ([p.generation for p in alive], "Generation", "#9f7aea"),
        ]):
            with col:
                f = px.histogram(x=vals, nbins=18, title=title, color_discrete_sequence=[color])
                f.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
                st.plotly_chart(_small(f, 230), width="stretch")

        a, b = st.columns(2)
        with a:
            st.markdown("**Population mean emotional state**")
            means = [float(np.mean([p.hrc.emotion(e) for p in alive])) for e in EMOTION_NAMES]
            f = go.Figure(go.Barpolar(r=means + [means[0]], theta=EMOTION_NAMES + [EMOTION_NAMES[0]],
                                      marker_color="#f6ad55"))
            st.plotly_chart(_small(f, 320), width="stretch")
        with b:
            st.markdown("**Token economy (population-wide)**")
            tot = {k: sum(p.inventory.get(k, 0.0) for p in alive) for k in ("red", "green", "blue")}
            f = _bar(list(tot), list(tot.values()), color=list(tot),
                     color_map={"red": "#e53e3e", "green": "#48bb78", "blue": "#4299e1"},
                     ytitle="tokens")
            st.plotly_chart(_small(f, 320), width="stretch")

        st.markdown("#### Aggregate action frequencies (lifetime, whole population)")
        tally = {a_: sum(p.action_counts.get(a_, 0) for p in alive) for a_ in ACTION_NAMES}
        tally = {k: v for k, v in sorted(tally.items(), key=lambda kv: -kv[1]) if v > 0}
        f = _bar(list(tally), list(tally.values()), color_seq=["#4fd1c5"],
                 ytitle="times chosen", angle=-40)
        st.plotly_chart(_small(f, 320), width="stretch")


def _order_parameter(alive: List[BioHyperAgent]) -> float:
    if not alive:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * np.array([p.theta for p in alive])))))


# ----------------------------------------------------------------------------
# PANEL 2 — Biome Cartography
# ----------------------------------------------------------------------------
def render_biome_cartography() -> None:
    world = st.session_state.world
    A, B = st.session_state.morphogen_A, st.session_state.morphogen_B
    alive = _alive()

    st.markdown("#### Cultural Stigmergy Map (Meme Grid)")
    st.caption("The 8-channel meme grid agents deposit cultural signal into — RGB shows the "
               "first three channels (danger / resource / sacred).")
    meme = world.meme_grid[:, :, :3].astype(np.float64)
    meme = meme / (meme.max() + 1e-9)
    st.plotly_chart(_small(px.imshow(meme, origin="lower"), 430), width="stretch")

    st.markdown("#### Biogenic Bloom Map (Gray–Scott morphogenesis × elemental chemistry)")
    st.caption("R/G are real Turing reaction–diffusion morphogens; Blue is the locally dominant "
               "chemical element, projected from world resources through chemistry.py's real profile table.")
    st.plotly_chart(_small(px.imshow(biome_rgb(A, B, dominant_element_hue_map(world)), origin="lower"), 460),
                    width="stretch")

    st.markdown("#### Agent positions & density")
    a, b = st.columns(2)
    with a:
        if alive:
            f = px.scatter(x=[p.x for p in alive], y=[p.y for p in alive],
                           color=[p.role for p in alive], size=[3 + 4 * p.energy for p in alive],
                           title="Agents by caste")
            f.update_layout(xaxis_title=None, yaxis_title=None,
                            xaxis_range=[0, world.width], yaxis_range=[0, world.height])
            st.plotly_chart(_small(f, 380), width="stretch")
    with b:
        if alive:
            dens, _, _ = np.histogram2d([p.y for p in alive], [p.x for p in alive],
                                        bins=[16, 16], range=[[0, world.height], [0, world.width]])
            f = px.imshow(dens, origin="lower", color_continuous_scale="plasma", title="Population density")
            st.plotly_chart(_small(f, 380), width="stretch")

    st.markdown("#### Physical fields")
    a, b, c = st.columns(3)
    with a:
        st.plotly_chart(_small(px.imshow(world.heat_field, origin="lower",
                                         color_continuous_scale="inferno", title="Temperature"), 300),
                        width="stretch")
    with b:
        st.plotly_chart(_small(px.imshow(world.fertility, origin="lower",
                                         color_continuous_scale="YlGn", title="Terrain fertility"), 300),
                        width="stretch")
    with c:
        st.plotly_chart(_small(px.imshow(world.resource_grid.sum(axis=2), origin="lower",
                                         color_continuous_scale="viridis", title="Total resources"), 300),
                        width="stretch")

    st.markdown("#### Resource channels")
    cols = st.columns(4)
    for col, i in zip(cols, range(len(RESOURCE_NAMES))):
        with col:
            st.plotly_chart(_small(px.imshow(world.resource_grid[:, :, i], origin="lower",
                                             color_continuous_scale="viridis",
                                             title=RESOURCE_NAMES[i].capitalize()), 240), width="stretch")

    st.markdown("#### Morphogen fields (the two Gray–Scott species, separately)")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(_small(px.imshow(A, origin="lower", color_continuous_scale="magma",
                                         title="Morphogen A"), 300), width="stretch")
    with b:
        st.plotly_chart(_small(px.imshow(B, origin="lower", color_continuous_scale="magma",
                                         title="Morphogen B"), 300), width="stretch")

    with st.expander("Pheromone channels (16-channel stigmergic trail system)"):
        ch = st.selectbox("Channel", PHEROMONE_NAMES, index=0)
        i = PHEROMONE_NAMES.index(ch)
        st.plotly_chart(_small(px.imshow(world.pheromone_grid[:, :, i], origin="lower",
                                         color_continuous_scale="inferno"), 360), width="stretch")
        totals = world.pheromone_grid.sum(axis=(0, 1))
        f = _bar(PHEROMONE_NAMES, totals, title="Total signal per pheromone channel",
                 color_seq=["#f6ad55"], angle=-45)
        st.plotly_chart(_small(f, 280), width="stretch")

    with st.expander("Meme channels (8-channel cultural grid)"):
        totals = world.meme_grid.sum(axis=(0, 1))
        f = _bar(MEME_NAMES, totals, title="Total signal per meme channel",
                 color_seq=["#9f7aea"], angle=-45)
        st.plotly_chart(_small(f, 280), width="stretch")
        ch = st.selectbox("Meme channel map", MEME_NAMES, index=0)
        st.plotly_chart(_small(px.imshow(world.meme_grid[:, :, MEME_NAMES.index(ch)], origin="lower",
                                         color_continuous_scale="purples"), 340), width="stretch")


# ----------------------------------------------------------------------------
# PANEL 3 — Consciousness Inspector
# ----------------------------------------------------------------------------
def render_consciousness_inspector() -> None:
    alive = _alive()
    if not alive:
        st.warning("No living agents to inspect.")
        return

    ids = [p.agent_id for p in alive]
    sel = st.selectbox("Select an agent", ids, index=0)
    agent = next(p for p in alive if p.agent_id == sel)
    hrc = agent.hrc
    spec = hrc.spectral_summary()

    c = st.columns(6)
    c[0].metric("Role", agent.role)
    c[1].metric("Generation", agent.generation)
    c[2].metric("Age", agent.age)
    c[3].metric("Energy", f"{agent.energy:.3f}")
    c[4].metric("Health", f"{agent.health:.3f}")
    c[5].metric("Tribe", agent.tribe_id if agent.tribe_id is not None else "—")

    c = st.columns(6)
    c[0].metric("Φ (last)", f"{spec['phi_last']:.4f}")
    c[1].metric("Confidence", f"{spec['confidence']:.3f}")
    c[2].metric("Crystallised", spec["crystallized"], help="Hopfield attractors formed")
    c[3].metric("Discoveries", len(agent.discoveries))
    c[4].metric("Meta-inventions", agent.n_meta_inventions)
    c[5].metric("λ spread", f"{spec['lambda_spread']:.2f}")

    a, b = st.columns(2)
    with a:
        if hrc.phi_history:
            f = go.Figure([go.Scatter(y=hrc.phi_history[-250:], line=dict(color="#f6ad55"))])
            f.update_layout(title="Φ history (this agent)")
            st.plotly_chart(_small(f, 260), width="stretch")
    with b:
        if hrc.free_energy_history:
            f = go.Figure([go.Scatter(y=hrc.free_energy_history[-250:], line=dict(color="#63b3ed"))])
            f.update_layout(title="Variational free energy (prediction error)")
            st.plotly_chart(_small(f, 260), width="stretch")

    a, b = st.columns(2)
    with a:
        vals = [float(hrc.emotion(e)) for e in EMOTION_NAMES]
        f = go.Figure(go.Barpolar(r=vals + [vals[0]], theta=EMOTION_NAMES + [EMOTION_NAMES[0]],
                                  marker_color="#4fd1c5"))
        f.update_layout(title="Emotional state")
        st.plotly_chart(_small(f, 330), width="stretch")
    with b:
        _, lam = hrc._eig()
        f = go.Figure([go.Bar(y=lam, marker_color="#9f7aea")])
        f.update_layout(title="Hamiltonian eigenspectrum (64 cognitive modes)",
                        xaxis_title="mode", yaxis_title="λ")
        st.plotly_chart(_small(f, 330), width="stretch")

    st.markdown("#### Wavefunction ψ — the live cognitive state")
    a, b, c3 = st.columns(3)
    with a:
        f = go.Figure([go.Bar(y=np.abs(hrc.psi), marker_color="#4fd1c5")])
        f.update_layout(title="|ψ| amplitude per mode")
        st.plotly_chart(_small(f, 260), width="stretch")
    with b:
        f = go.Figure([go.Bar(y=np.angle(hrc.psi), marker_color="#f6ad55")])
        f.update_layout(title="arg(ψ) phase per mode")
        st.plotly_chart(_small(f, 260), width="stretch")
    with c3:
        f = go.Figure([go.Scatter(x=hrc.psi.real, y=hrc.psi.imag, mode="markers",
                                  marker=dict(size=6, color=np.arange(K_DIM), colorscale="turbo"))])
        f.update_layout(title="ψ in the complex plane", xaxis_title="Re", yaxis_title="Im")
        st.plotly_chart(_small(f, 260), width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Hamiltonian |H| magnitude**")
        st.plotly_chart(_small(px.imshow(np.abs(hrc.H), color_continuous_scale="viridis"), 330),
                        width="stretch")
    with b:
        st.markdown("**Meta-Hamiltonian |H_meta| (how this agent learns)**")
        st.plotly_chart(_small(px.imshow(np.abs(hrc.meta.H), color_continuous_scale="magma"), 330),
                        width="stretch")

    a, b = st.columns(2)
    with a:
        f = go.Figure([go.Bar(y=hrc.meta.learning_rate_profile(K_DIM), marker_color="#48bb78")])
        f.update_layout(title="Meta-modulated learning-rate profile µ")
        st.plotly_chart(_small(f, 260), width="stretch")
    with b:
        f = go.Figure([go.Bar(y=hrc.exploration_counts, marker_color="#ed8936")])
        f.update_layout(title="Eigenmode exploration counts (dark modes drive invention)")
        st.plotly_chart(_small(f, 260), width="stretch")

    st.markdown("#### Genome — read directly off this agent's Hamiltonian eigenbasis")
    fp = genome_fingerprint(hrc.eigenvectors())
    st.code(fp["dna"], language=None)
    a, b = st.columns([2, 1])
    with a:
        base_counts = {base: fp["dna"].count(base) for base in "ACGT"}
        f = _bar(list(base_counts), list(base_counts.values()), title="Nucleotide composition",
                 color=list(base_counts),
                 color_map={"A": "#e53e3e", "C": "#4299e1", "G": "#48bb78", "T": "#ed8936"})
        st.plotly_chart(_small(f, 260), width="stretch")
        gc = 100.0 * (base_counts["G"] + base_counts["C"]) / max(len(fp["dna"]), 1)
        st.caption(f"GC content: {gc:.1f}% — in real genomics this correlates with thermal stability.")
    with b:
        prot = pd.Series(fp["protein"]).value_counts()
        f = _bar(prot.index.tolist(), prot.values.tolist(), title="Amino acid counts",
                 color_seq=["#9f7aea"])
        st.plotly_chart(_small(f, 260), width="stretch")
    st.caption("Codons → " + " · ".join(fp["protein"]))

    a, b = st.columns(2)
    with a:
        st.markdown("**Game of Life scratchpad (seeded from this agent's DNA)**")
        f = px.imshow(agent.gol_grid, color_continuous_scale="greys", origin="lower")
        f.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_small(f, 250), width="stretch")
        st.caption(f"Live cell fraction: {agent.gol_alive_fraction:.3f}")
    with b:
        st.markdown("**Causal model — mean reward by action**")
        rows = [(k, float(np.mean(v))) for k, v in agent.causal_model.items() if v]
        if rows:
            rows.sort(key=lambda r: -r[1])
            f = _bar([r[0] for r in rows], [r[1] for r in rows], color_seq=["#4fd1c5"],
                     ytitle="mean reward", angle=-45)
            st.plotly_chart(_small(f, 250), width="stretch")

    if agent.trust:
        st.markdown("**Trust toward other agents**")
        items = sorted(agent.trust.items(), key=lambda kv: -kv[1])[:20]
        f = _bar([str(k) for k, _ in items], [v for _, v in items], color_seq=["#4fd1c5"],
                 xtitle="agent id", ytitle="trust")
        st.plotly_chart(_small(f, 250), width="stretch")

    if agent.discoveries:
        st.markdown("**Inventions by this agent**")
        st.dataframe(pd.DataFrame([
            {"Name": n, "Gödel number": g, "Program": " → ".join(prog)}
            for prog, g, n in agent.discoveries[-12:]
        ]), width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# PANEL 4 — Civilization
# ----------------------------------------------------------------------------
def render_civilization() -> None:
    civ: Civilization = st.session_state.civ
    population = st.session_state.population
    alive = _alive()

    active = {tid: t for tid, t in civ.tribes.items() if t.members(population)}
    if not active:
        st.info("No tribes have formed yet — advance the simulation.")
        return

    c = st.columns(5)
    c[0].metric("Active tribes", len(active))
    c[1].metric("Tech tree nodes", civ.tech_tree.number_of_nodes())
    c[2].metric("Tech tree edges", civ.tech_tree.number_of_edges())
    c[3].metric("B_tech", f"{civ.b_tech:.3f}")
    c[4].metric("Liquid Dunbar max", civ.liquid_dunbar_max(),
                help="Max tribe size, grows with technology: 12 + 3·|TechTree|")

    rows = []
    for tid, tribe in active.items():
        rows.append({"Tribe": tid, "Members": len(tribe.members(population)),
                     "Discoveries": tribe.discoveries, "Alliances": len(tribe.alliances),
                     "Trades": tribe.trade_count(population),
                     "Power": round(civ.compute_tribal_power(tid, population), 2),
                     "Colour": f"rgb{tribe.color}", "Founded": tribe.founded_tick})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Tribal power decomposition (the eight weighted terms of §3.16)")
    tid = st.selectbox("Tribe", list(active.keys()))
    members = active[tid].members(population)
    if members:
        terms = {
            "0.30·Σenergy": 0.30 * sum(p.energy for p in members),
            "0.20·Σhealth": 0.20 * sum(p.health for p in members),
            "2.5·discoveries": 2.5 * active[tid].discoveries,
            "0.55·size": 0.55 * len(members),
            "1.5·alliances": 1.5 * len(active[tid].alliances),
            "3.0·meta-spread": 3.0 * float(np.mean([np.std(p.hrc.meta.eigvals()) for p in members])),
            "5.0·mean Φ": 5.0 * float(np.mean([p.hrc.phi_history[-1] if p.hrc.phi_history else 0.0
                                               for p in members])),
            "0.10·trades": 0.10 * active[tid].trade_count(population),
        }
        f = _bar(list(terms), list(terms.values()), color_seq=["#4fd1c5"],
                 ytitle="contribution", angle=-40)
        st.plotly_chart(_small(f, 320), width="stretch")

    a, b = st.columns(2)
    with a:
        f = _pie([f"Tribe {t}" for t in active],
                 [len(x.members(population)) for x in active.values()], title="Membership share")
        st.plotly_chart(_small(f, 320), width="stretch")
    with b:
        powers = {t: civ.compute_tribal_power(t, population) for t in active}
        f = _bar([f"Tribe {t}" for t in powers], list(powers.values()),
                 title="Tribal power", color_seq=["#9f7aea"])
        st.plotly_chart(_small(f, 320), width="stretch")

    st.markdown("#### Technology")
    n = civ.tech_tree.number_of_nodes()
    if n:
        recent = list(civ.tech_tree.nodes)[-TECH_TREE_RENDER_LIMIT:]
        sub = civ.tech_tree.subgraph(recent)
        st.caption(f"Showing the most recent {len(recent)} of {n} tech-tree nodes — a force-directed "
                   f"layout of the full graph would be unreadable at scale.")
        pos = nx.spring_layout(sub, seed=1)
        ex, ey = [], []
        for u, v in sub.edges():
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
        deg = dict(sub.degree())
        f = go.Figure()
        f.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#555"), hoverinfo="none"))
        f.add_trace(go.Scatter(x=[pos[k][0] for k in sub.nodes()], y=[pos[k][1] for k in sub.nodes()],
                               mode="markers",
                               marker=dict(size=[6 + 3 * deg.get(k, 0) for k in sub.nodes()],
                                           color=[deg.get(k, 0) for k in sub.nodes()],
                                           colorscale="turbo", showscale=False),
                               text=[f"Gödel {k}" for k in sub.nodes()], hoverinfo="text"))
        f.update_layout(height=430, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(f, width="stretch")

        a, b = st.columns(2)
        with a:
            degs = list(dict(civ.tech_tree.degree()).values())
            f = px.histogram(x=degs, nbins=20, color_discrete_sequence=["#63b3ed"])
            f.update_layout(title="Tech-tree degree distribution", xaxis_title="degree", yaxis_title=None)
            st.plotly_chart(_small(f, 260), width="stretch")
        with b:
            ticks = [d.get("tick", 0) for _, d in civ.tech_tree.nodes(data=True)]
            f = px.histogram(x=ticks, nbins=30, color_discrete_sequence=["#48bb78"])
            f.update_layout(title="Invention rate over time", xaxis_title="tick", yaxis_title="inventions")
            st.plotly_chart(_small(f, 260), width="stretch")

    st.markdown("#### Alliance network")
    G = nx.Graph()
    for tid in active:
        G.add_node(tid)
    for tid, tribe in active.items():
        for other in tribe.alliances:
            if other in active:
                G.add_edge(tid, other)
    if G.number_of_edges():
        pos = nx.spring_layout(G, seed=3)
        ex, ey = [], []
        for u, v in G.edges():
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
        f = go.Figure()
        f.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#48bb78", width=2)))
        f.add_trace(go.Scatter(x=[pos[k][0] for k in G.nodes()], y=[pos[k][1] for k in G.nodes()],
                               mode="markers+text", text=[f"T{k}" for k in G.nodes()],
                               textposition="top center", marker=dict(size=18, color="#4fd1c5")))
        f.update_layout(height=320, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(f, width="stretch")
    else:
        st.caption("No alliances currently active. Note the documented finding in BUILD_STATUS.md: "
                   "tribal diversification does not occur under the current parameterisation, so "
                   "diplomacy has no second tribe to engage with.")

    if len(active) >= 2:
        st.markdown("#### Epistemic distance between tribes (schism risk)")
        ids = list(active)
        M = np.zeros((len(ids), len(ids)))
        for i, x in enumerate(ids):
            for j, y in enumerate(ids):
                if i != j:
                    d = civ.epistemic_distance(x, y, population)
                    M[i, j] = d if d is not None else 0.0
        f = px.imshow(M, x=[f"T{i}" for i in ids], y=[f"T{i}" for i in ids],
                      color_continuous_scale="RdYlGn_r")
        st.plotly_chart(_small(f, 320), width="stretch")

    if civ.events:
        st.markdown("#### Diplomacy log")
        for e in civ.events[-20:][::-1]:
            st.text(e)


# ----------------------------------------------------------------------------
# PANEL 5 — Narrative
# ----------------------------------------------------------------------------
def render_narrative() -> None:
    narrative: NarrativeEngine = st.session_state.narrative
    if not narrative.traditions:
        st.info("No traditions founded yet — a tribe needs to exist first.")
        return

    tid = st.selectbox("Tribe", list(narrative.traditions.keys()))
    tradition = narrative.traditions[tid]
    fid_history = narrative.fidelity_history.get(tid, [])

    c = st.columns(5)
    c[0].metric("Fidelity", f"{tradition.fidelity():.3f}")
    c[1].metric("Motifs drifted", f"{tradition.hamming_drift()}/{len(tradition.origin_sequence)}")
    c[2].metric("Generations", narrative.generations_tracked(tid))
    c[3].metric("Founded gen.", tradition.founded_generation)
    c[4].metric("Gödel number", f"{tradition.godel_number():.3g}")

    if narrative.milestones_reached.get(tid):
        st.success("✅ Milestone Reached: Stable Traditions")

    if len(fid_history) >= 2:
        f = go.Figure([go.Scatter(y=fid_history, line=dict(color="#f6ad55"), name="observed")])
        theo = [expected_retention_after(i) for i in range(len(fid_history))]
        f.add_trace(go.Scatter(y=theo, line=dict(color="#63b3ed", dash="dot"),
                               name="Swadesh expected retention"))
        f.add_hline(y=0.95, line_dash="dot", annotation_text="stability threshold")
        f.update_layout(title="Tradition persistence vs. the real glottochronological decay curve")
        st.plotly_chart(_small(f, 330), width="stretch")

    a, b = st.columns(2)
    with a:
        shared = max(1.0 - tradition.hamming_drift() / max(len(tradition.origin_sequence), 1), 1e-6)
        t_div = glottochronology_divergence_time(shared)
        st.metric("Implied divergence time", f"{t_div:.3f} units",
                  help="Swadesh's t = ln(c) / (2·ln(r)); one unit is the period the retention "
                       "constant r=0.805 was calibrated for (a millennium, in real linguistics).")
        drift_mask = [o != c_ for o, c_ in zip(tradition.origin_sequence, tradition.current_sequence)]
        f = px.imshow(np.array(drift_mask, dtype=int).reshape(1, -1),
                      color_continuous_scale=[[0, "#48bb78"], [1, "#e53e3e"]], origin="lower")
        f.update_layout(title="Per-motif drift map (red = changed)", coloraxis_showscale=False,
                        yaxis=dict(visible=False))
        st.plotly_chart(_small(f, 200), width="stretch")
    with b:
        counts = pd.Series([ALL_PRIMITIVES[i] for i in tradition.current_sequence]).value_counts()
        f = _bar(counts.index.tolist(), counts.values.tolist(),
                 title="Motif composition of the current myth", color_seq=["#9f7aea"], angle=-45)
        st.plotly_chart(_small(f, 300), width="stretch")

    if len(narrative.traditions) > 1:
        st.markdown("#### Cross-tribe myth fidelity")
        ids = list(narrative.traditions)
        f = _bar([f"T{i}" for i in ids], [narrative.traditions[i].fidelity() for i in ids],
                 color_seq=["#4fd1c5"], ytitle="fidelity")
        st.plotly_chart(_small(f, 280), width="stretch")

    st.markdown("#### Founding myth (Gödel-encoded with the same machinery as any invention)")
    st.code(" → ".join(ALL_PRIMITIVES[i] for i in tradition.origin_sequence), language=None)
    st.markdown("#### Current retelling")
    st.code(" → ".join(ALL_PRIMITIVES[i] for i in tradition.current_sequence), language=None)
    st.caption(f"Swadesh retention constant in use: r = {GLOTTOCHRONOLOGY_RETENTION_RATE} per generation, "
               f"boosted by local 'sacred' meme signal.")


# ----------------------------------------------------------------------------
# PANEL 6 — Nobel Committee
# ----------------------------------------------------------------------------
def render_nobel() -> None:
    nobel: NobelCommittee = st.session_state.nobel
    cats = ["Physics", "Chemistry", "Physiology or Medicine", "Literature", "Peace", "Economics"]

    counts = {c: 0 for c in cats}
    for l in nobel.laureates:
        counts[l.category] = counts.get(l.category, 0) + 1

    c = st.columns(6)
    for col, cat in zip(c, cats):
        col.metric(cat.split()[0], counts.get(cat, 0))

    a, b = st.columns(2)
    with a:
        f = _bar(list(counts), list(counts.values()), title="Laureates by category",
                 color=list(counts), angle=-30)
        st.plotly_chart(_small(f, 320), width="stretch")
    with b:
        if nobel.laureates:
            f = px.scatter(x=[l.tick for l in nobel.laureates],
                           y=[l.category for l in nobel.laureates],
                           color=[l.category for l in nobel.laureates])
            f.update_layout(title="Breakthroughs over time", showlegend=False,
                            xaxis_title="tick", yaxis_title=None)
            st.plotly_chart(_small(f, 320), width="stretch")
        else:
            st.info("No laureates yet — six categories are being watched.")

    c = st.columns(3)
    c[0].metric("Molecules known", len(nobel.discovery_log.known_formulas))
    c[1].metric("Deaths recorded", len(nobel.death_log.records))
    c[2].metric("Trade-efficiency record", f"{nobel.economics.record_efficiency:.3f}")

    if nobel.death_log.records:
        st.markdown("#### Lifespan record (feeds the Medicine category)")
        ages = [a for _, a in nobel.death_log.records]
        a, b = st.columns(2)
        with a:
            f = px.histogram(x=ages, nbins=25, color_discrete_sequence=["#ed8936"])
            f.update_layout(title="Lifespan distribution", xaxis_title="age at death", yaxis_title=None)
            st.plotly_chart(_small(f, 280), width="stretch")
        with b:
            by_group: Dict[int, List[float]] = {}
            for gid, age in nobel.death_log.records:
                by_group.setdefault(gid, []).append(age)
            gl = sorted(by_group)
            f = _bar([f"grp {g}" for g in gl], [float(np.mean(by_group[g])) for g in gl],
                     title="Mean lifespan by lineage", color_seq=["#4fd1c5"], ytitle="mean age")
            st.plotly_chart(_small(f, 280), width="stretch")

    if nobel.laureates:
        st.markdown("#### Laureate ledger")
        st.dataframe(pd.DataFrame([{"Category": l.category, "Tick": l.tick, "Detail": l.detail}
                                   for l in nobel.laureates[::-1]]), width="stretch", hide_index=True)


# ----------------------------------------------------------------------------
# PANEL 7 — Evolution & Phylogeny
# ----------------------------------------------------------------------------
def render_evolution() -> None:
    evo: EvolutionEngine = st.session_state.evo
    population = st.session_state.population
    hist = st.session_state.history
    alive = _alive()

    c = st.columns(5)
    c[0].metric("Clades", evo.phylo.n_clades)
    c[1].metric("Tradition verified", "Yes" if evo.tradition_verified else "No")
    c[2].metric("Cultural ratchet r",
                f"{evo.cultural_ratchet_history[-1]:.3f}" if evo.cultural_ratchet_history else "—",
                help="0.55 is the stated verification bar")
    c[3].metric("Max generation", max((p.generation for p in alive), default=0))
    c[4].metric("Mean meta-fitness",
                f"{np.mean(list(evo.meta_fitness.values())):.2f}" if evo.meta_fitness else "—")

    if evo.archetypes:
        a, b = st.columns(2)
        counts: Dict[str, int] = {}
        for v in evo.archetypes.values():
            counts[v] = counts.get(v, 0) + 1
        with a:
            f = _pie(list(counts), list(counts.values()),
                     title="Behavioural archetypes (KMeans on task-band eigenvalues)")
            st.plotly_chart(_small(f, 330), width="stretch")
        with b:
            roles: Dict[str, int] = {}
            for p in alive:
                roles[p.role] = roles.get(p.role, 0) + 1
            f = _pie(list(roles), list(roles.values()), title="Caste distribution")
            st.plotly_chart(_small(f, 330), width="stretch")

    valid = [r for r in hist["cultural_ratchet"] if not (isinstance(r, float) and np.isnan(r))]
    if len(valid) >= 2:
        f = go.Figure([go.Scatter(y=valid, line=dict(color="#4fd1c5"))])
        f.add_hline(y=0.55, line_dash="dot", annotation_text="verification bar")
        f.update_layout(title="Cultural ratchet r over time (founder vs. descendant action profiles)")
        st.plotly_chart(_small(f, 300), width="stretch")

    if alive:
        st.markdown("#### Genetic structure")
        sample = alive[:40]
        dnas = [genome_fingerprint(p.hrc.eigenvectors())["dna"] for p in sample]
        n = len(sample)
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    M[i, j] = genetic_distance(dnas[i], dnas[j])
        a, b = st.columns(2)
        with a:
            f = px.imshow(M, color_continuous_scale="viridis",
                          title=f"Pairwise genetic distance ({n} agents)")
            st.plotly_chart(_small(f, 340), width="stretch")
        with b:
            tri = M[np.triu_indices(n, 1)]
            f = px.histogram(x=tri, nbins=25, color_discrete_sequence=["#9f7aea"])
            f.update_layout(title="Genetic distance distribution", xaxis_title="fraction of bases differing",
                            yaxis_title=None)
            st.plotly_chart(_small(f, 340), width="stretch")
            st.caption(f"Mean pairwise distance: {tri.mean():.3f}")

        st.markdown("#### Cognitive diversity")
        a, b = st.columns(2)
        with a:
            spreads = [p.hrc.spectral_summary()["lambda_spread"] for p in alive]
            f = px.histogram(x=spreads, nbins=20, color_discrete_sequence=["#f6ad55"])
            f.update_layout(title="Eigenvalue spread (drives caste assignment)",
                            xaxis_title="λ spread", yaxis_title=None)
            st.plotly_chart(_small(f, 280), width="stretch")
        with b:
            phis = [p.hrc.phi_history[-1] for p in alive if p.hrc.phi_history]
            if phis:
                f = px.histogram(x=phis, nbins=20, color_discrete_sequence=["#48bb78"])
                f.update_layout(title="Φ distribution", xaxis_title="Φ", yaxis_title=None)
                st.plotly_chart(_small(f, 280), width="stretch")

        st.markdown("#### Kuramoto phase synchronisation")
        a, b = st.columns(2)
        with a:
            th = np.array([p.theta for p in alive])
            f = go.Figure([go.Scatterpolar(r=np.ones_like(th), theta=np.degrees(th), mode="markers",
                                           marker=dict(size=7, color="#4fd1c5"))])
            f.update_layout(title=f"Phase wheel — order parameter r = {_order_parameter(alive):.3f}",
                            polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(_small(f, 320), width="stretch")
        with b:
            f = px.scatter(x=[p.age for p in alive], y=[p.energy for p in alive],
                           color=[p.role for p in alive], size=[1 + len(p.discoveries) for p in alive],
                           title="Age vs. energy (size = discoveries)")
            f.update_layout(xaxis_title="age", yaxis_title="energy")
            st.plotly_chart(_small(f, 320), width="stretch")

        st.markdown("#### Meta-fitness leaderboard")
        if evo.meta_fitness:
            top = sorted(evo.meta_fitness.items(), key=lambda kv: -kv[1])[:15]
            f = _bar([str(k) for k, _ in top], [v for _, v in top], color_seq=["#63b3ed"],
                     xtitle="agent id", ytitle="meta-fitness")
            st.plotly_chart(_small(f, 280), width="stretch")


# ----------------------------------------------------------------------------
# PANEL 8 — Chemistry Lab  (new)
# ----------------------------------------------------------------------------
def render_chemistry() -> None:
    world = st.session_state.world

    st.markdown("#### The periodic subset in play")
    st.caption("Real IUPAC standard atomic weights and real Pauling-scale electronegativities — "
               "the same table chemistry.py uses for metabolism and molecule discovery.")
    df = pd.DataFrame({"Element": ELEMENTS,
                       "Atomic weight": [ATOMIC_WEIGHT[e] for e in ELEMENTS],
                       "Electronegativity": [ELECTRONEGATIVITY[e] for e in ELEMENTS]})
    a, b = st.columns(2)
    with a:
        f = px.bar(df, x="Element", y="Atomic weight", color_discrete_sequence=["#4fd1c5"])
        st.plotly_chart(_small(f, 280), width="stretch")
    with b:
        f = px.bar(df, x="Element", y="Electronegativity", color_discrete_sequence=["#ed8936"])
        st.plotly_chart(_small(f, 280), width="stretch")

    f = px.scatter(df, x="Atomic weight", y="Electronegativity", text="Element",
                   color="Electronegativity", color_continuous_scale="turbo",
                   title="Electronegativity vs. atomic weight")
    f.update_traces(textposition="top center", marker=dict(size=13))
    st.plotly_chart(_small(f, 380), width="stretch")

    st.markdown("#### Elemental composition of the world")
    M = _resource_profile_matrix(tuple(RESOURCE_NAMES))
    element_field = world.resource_grid.astype(np.float64) @ M
    totals = element_field.sum(axis=(0, 1))
    a, b = st.columns(2)
    with a:
        f = _bar(ELEMENTS, totals, title="Total elemental abundance",
                 color_seq=["#9f7aea"], ytitle="abundance")
        st.plotly_chart(_small(f, 300), width="stretch")
    with b:
        f = _pie(ELEMENTS, totals, title="Elemental share")
        st.plotly_chart(_small(f, 300), width="stretch")

    el = st.selectbox("Element distribution map", ELEMENTS, index=ELEMENTS.index("C"))
    st.plotly_chart(_small(px.imshow(element_field[:, :, ELEMENTS.index(el)], origin="lower",
                                     color_continuous_scale="viridis"), 360), width="stretch")

    st.markdown("#### Arrhenius kinetics — real temperature dependence")
    st.caption("k(T) = A·exp(−Eₐ/RT), evaluated across this world's actual temperature field, "
               "mapped onto a thermodynamic scale by chemistry.py's documented bridge.")
    heats = np.linspace(0.0, 2.0, 60)
    temps = np.array([heat_to_kelvin(h) for h in heats])
    a, b = st.columns(2)
    with a:
        f = go.Figure()
        for rxn, colr in ((RESPIRATION, "#e53e3e"), (PHOTOSYNTHESIS, "#48bb78")):
            f.add_trace(go.Scatter(x=temps, y=[arrhenius_rate_constant(rxn, t) for t in temps],
                                   name=rxn.name.split("(")[0].strip(), line=dict(color=colr)))
        f.update_layout(title="Rate constant vs. temperature", xaxis_title="T (K)",
                        yaxis_title="k", yaxis_type="log")
        st.plotly_chart(_small(f, 300), width="stretch")
    with b:
        k_field = np.vectorize(lambda h: arrhenius_rate_constant(RESPIRATION, heat_to_kelvin(h)))(world.heat_field)
        f = px.imshow(np.log10(k_field + 1e-12), origin="lower", color_continuous_scale="inferno")
        f.update_layout(title="log₁₀ respiration rate across the world")
        st.plotly_chart(_small(f, 300), width="stretch")

    st.markdown("#### Reference reactions (both verified mass-balanced)")
    rows = []
    for rxn in (RESPIRATION, PHOTOSYNTHESIS):
        rows.append({"Reaction": rxn.name.split("(")[0].strip(),
                     "ΔG (kJ/mol)": rxn.delta_g_kj_per_mol,
                     "Eₐ (J/mol)": rxn.activation_energy_j_per_mol,
                     "Mass balanced": "✅" if rxn.is_mass_balanced() else "❌",
                     "Spontaneous": "Yes (exergonic)" if rxn.delta_g_kj_per_mol < 0 else "No (endergonic)"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Molecules")
    mols = [GLUCOSE, WATER, CO2, O2]
    st.dataframe(pd.DataFrame([{"Name": m.name, "Formula (Hill)": m.hill_formula(),
                                "Molar mass (g/mol)": round(m.molar_mass(), 3)} for m in mols]),
                 width="stretch", hide_index=True)
    f = _bar([m.hill_formula() for m in mols], [m.molar_mass() for m in mols],
             title="Molar masses", color_seq=["#63b3ed"], ytitle="g/mol")
    st.plotly_chart(_small(f, 260), width="stretch")

    st.markdown("#### The standard genetic code")
    st.caption("The real 64-codon table — the same one biology.py translates agent genomes through, "
               "and the reason K = 64 = 4³ unifies cognition with genetics at all.")
    aa_counts = pd.Series(list(STANDARD_GENETIC_CODE.values())).value_counts()
    f = _bar(aa_counts.index.tolist(), aa_counts.values.tolist(),
             title="Codon degeneracy — codons encoding each amino acid",
             color_seq=["#9f7aea"], ytitle="codons")
    st.plotly_chart(_small(f, 300), width="stretch")
    st.caption("This redundancy is what makes neutral drift possible in real molecular evolution "
               "(Kimura's neutral theory) — several codons map to the same amino acid, so many "
               "mutations are phenotypically silent.")


# ----------------------------------------------------------------------------
# PANEL 9 — Geometry & Topology  (new)
# ----------------------------------------------------------------------------
def render_geometry() -> None:
    world = st.session_state.world
    A, B = st.session_state.morphogen_A, st.session_state.morphogen_B

    st.markdown("#### Algebraic topology of the living fields")
    st.caption("β₀ counts connected components; β₁ counts enclosed holes; the Euler characteristic "
               "χ = β₀ − β₁. These are real topological invariants, verified in geometry.py against "
               "known-correct test shapes.")

    field_name = st.selectbox("Field", ["Morphogen B", "Morphogen A", "Total resources",
                                        "Meme: danger", "Meme: sacred", "Temperature"])
    field = {
        "Morphogen B": B, "Morphogen A": A,
        "Total resources": world.resource_grid.sum(axis=2).astype(np.float64),
        "Meme: danger": world.meme_grid[:, :, MEME_NAMES.index("danger")].astype(np.float64),
        "Meme: sacred": world.meme_grid[:, :, MEME_NAMES.index("sacred")].astype(np.float64),
        "Temperature": world.heat_field,
    }[field_name]

    thr = st.slider("Binarisation threshold (percentile)", 10, 90, 50, 5)
    cutoff = np.percentile(field, thr)
    binary = (field > cutoff).astype(np.int8)
    summary = topology_summary(binary)

    c = st.columns(4)
    c[0].metric("β₀ (components)", summary["betti_0"])
    c[1].metric("β₁ (holes)", summary["betti_1"])
    c[2].metric("χ (Euler char.)", summary["euler_characteristic"])
    c[3].metric("Occupied fraction", f"{binary.mean():.3f}")

    a, b = st.columns(2)
    with a:
        f = px.imshow(field, origin="lower", color_continuous_scale="viridis")
        f.update_layout(title=f"{field_name} (continuous)")
        st.plotly_chart(_small(f, 340), width="stretch")
    with b:
        f = px.imshow(binary, origin="lower", color_continuous_scale="greys")
        f.update_layout(title=f"Binarised at the {thr}th percentile", coloraxis_showscale=False)
        st.plotly_chart(_small(f, 340), width="stretch")

    st.markdown("#### Topology across thresholds (a persistence-style sweep)")
    sweep = list(range(10, 95, 5))
    b0s, b1s = [], []
    for t in sweep:
        bin_t = (field > np.percentile(field, t)).astype(np.int8)
        b0s.append(betti_0(bin_t))
        b1s.append(betti_1(bin_t))
    f = go.Figure()
    f.add_trace(go.Scatter(x=sweep, y=b0s, name="β₀ components", line=dict(color="#4fd1c5")))
    f.add_trace(go.Scatter(x=sweep, y=b1s, name="β₁ holes", line=dict(color="#ed8936")))
    f.update_layout(title="Betti numbers vs. threshold", xaxis_title="percentile threshold",
                    yaxis_title="count")
    st.plotly_chart(_small(f, 320), width="stretch")

    st.markdown("#### French Flag developmental zoning")
    st.caption("Wolpert's (1969) positional-information model: a single smooth morphogen gradient, "
               "thresholded into three discrete zones — the simplest real mechanism known for "
               "turning a gradient into a body plan.")
    zones = french_flag_zones(A)
    a, b = st.columns(2)
    with a:
        f = px.imshow(zones, origin="lower", color_continuous_scale=[[0, "#e53e3e"], [0.5, "#f7fafc"],
                                                                     [1, "#4299e1"]])
        f.update_layout(title="Positional zones", coloraxis_showscale=False)
        st.plotly_chart(_small(f, 320), width="stretch")
    with b:
        counts = {f"Zone {z}": int((zones == z).sum()) for z in (0, 1, 2)}
        f = _bar(list(counts), list(counts.values()), title="Cells per zone",
                 color_seq=["#9f7aea"])
        st.plotly_chart(_small(f, 320), width="stretch")

    st.markdown("#### L-systems — real recursive growth grammars")
    st.caption("Lindenmayer (1968) string-rewriting grammars rendered by turtle graphics. "
               "These are the same grammars used to model real plant architecture.")
    a, b = st.columns(2)
    with a:
        sysname = st.selectbox("Grammar", ["Fractal plant", "Koch curve"])
    with b:
        iters = st.slider("Iterations", 1, 5, 4)

    lsys = FRACTAL_PLANT if sysname == "Fractal plant" else KOCH_CURVE
    expanded = lsys.expand(iters)
    segs = turtle_render(expanded, lsys.angle_degrees, step=1.0)

    c = st.columns(3)
    c[0].metric("Grammar length", f"{len(expanded):,} symbols")
    c[1].metric("Segments drawn", f"{len(segs):,}")
    bbox = bounding_box(segs)
    c[2].metric("Bounding box", f"{bbox[2]-bbox[0]:.1f} × {bbox[3]-bbox[1]:.1f}")

    if segs:
        xs, ys = [], []
        for x0, y0, x1, y1 in segs:
            xs += [x0, x1, None]
            ys += [y0, y1, None]
        f = go.Figure([go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#48bb78", width=1.4))])
        f.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"))
        st.plotly_chart(f, width="stretch")

    st.caption(f"Axiom: `{lsys.axiom}` · Rules: " +
               " · ".join(f"`{k} → {v}`" for k, v in lsys.rules.items()) +
               f" · Turn angle: {lsys.angle_degrees}°")
PANEL_RENDERERS = {
    PANELS[0]: render_observation_deck,
    PANELS[1]: render_biome_cartography,
    PANELS[2]: render_consciousness_inspector,
    PANELS[3]: render_civilization,
    PANELS[4]: render_narrative,
    PANELS[5]: render_nobel,
    PANELS[6]: render_evolution,
    PANELS[7]: render_chemistry,
    PANELS[8]: render_geometry,
}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    if not st.session_state.get("initialized", False):
        init_state()

    st.title("⬡ GeNeSIS V — OMNIGENESIS")
    st.caption("Quantum Biology · Chemical Genesis · Fractal Cosmogenesis")

    with st.sidebar:
        st.markdown("### Controls")
        n_ticks = st.slider("Ticks per advance", 1, 100, 10)
        if st.button("▶ Advance", width='stretch'):
            with st.spinner(f"Advancing {n_ticks} ticks..."):
                run_ticks(n_ticks)

        auto = st.checkbox("Auto-advance (small batches)", value=False)
        if auto:
            run_ticks(2)
            time.sleep(0.05)
            st.rerun()

        st.divider()
        if st.button("↺ Reset simulation", width='stretch'):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.divider()
        st.markdown("### Navigate")
        chosen = st.radio("Panel", PANELS, index=PANELS.index(st.session_state.active_panel), label_visibility="collapsed")
        st.session_state.active_panel = chosen

        st.divider()
        st.caption(f"World {WORLD_SIZE}×{WORLD_SIZE} · Population bounded [{POP_FLOOR}, {POP_CEILING}] · "
                   f"History capped at {HISTORY_CAP} ticks")

    # Lazy dispatch: only the selected panel's render function — and
    # therefore only its expensive plotting/computation — executes this run.
    PANEL_RENDERERS[st.session_state.active_panel]()


if __name__ == "__main__":
    main()
