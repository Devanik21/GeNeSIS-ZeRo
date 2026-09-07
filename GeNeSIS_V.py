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
)
from world import GenesisWorld, MEME_NAMES, PHEROMONE_NAMES, RESOURCE_NAMES
from agents import (
    BioHyperAgent, make_child, update_roles, spectral_resonance,
    MEME_ABSORPTION_EVERY, VIRAL_BROADCAST_EVERY, ACTION_NAMES,
)
from evolution import EvolutionEngine, POP_INITIAL, POP_FLOOR, POP_CEILING
from civilization import Civilization
from narrative import NarrativeEngine
from nobel import NobelCommittee
from chemistry import RESOURCE_ELEMENT_PROFILE, ELEMENTS, N_ELEMENTS

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
def render_observation_deck() -> None:
    world = st.session_state.world
    population = st.session_state.population
    civ = st.session_state.civ
    hist = st.session_state.history
    alive = [p for p in population if p.alive]

    n_tribes = len([t for t in civ.tribes.values() if t.members(population)])
    mean_fid = None
    traditions = st.session_state.narrative.traditions
    if traditions:
        mean_fid = float(np.mean([t.fidelity() for t in traditions.values()]))
    epoch = bloom_epoch_label(st.session_state.tick, n_tribes, civ.tech_tree.number_of_nodes(), mean_fid)

    st.subheader(f"Bloom Epoch: {epoch}")
    st.caption("A narrative display heuristic, not a precise measurement — see masterplan §19.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tick", st.session_state.tick)
    c2.metric("Population", f"{len(alive)}", help=f"Bounded to [{POP_FLOOR}, {POP_CEILING}] by evolution.py")
    c3.metric("Tribes", n_tribes)
    c4.metric("Tech Tree Nodes", civ.tech_tree.number_of_nodes())
    c5.metric("B_tech multiplier", f"{civ.b_tech:.3f}", help="Capped at 3.0")

    if len(hist["tick"]) >= 2:
        df = pd.DataFrame(hist)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["tick"], y=df["population"], name="Population", line=dict(color="#4fd1c5")))
        fig.update_layout(
            title="Population over time", xaxis_title="tick", yaxis_title="alive agents",
            height=320, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width='stretch')

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df["tick"], y=df["mean_phi"], name="Mean Φ", line=dict(color="#f6ad55")))
        fig2.update_layout(
            title="Mean IIT Φ across the population", xaxis_title="tick", yaxis_title="Φ",
            height=280, margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig2, width='stretch')
    else:
        st.info("Advance the simulation from the sidebar to see history accumulate.")


def render_biome_cartography() -> None:
    world = st.session_state.world
    A, B = st.session_state.morphogen_A, st.session_state.morphogen_B

    st.markdown("#### Cultural Stigmergy Map (Meme Grid)")
    st.caption("The original favourite — the 8-channel meme grid agents deposit cultural signal into, "
               "still here, still evolving the same way it always did.")
    meme_rgb = world.meme_grid[:, :, :3].astype(np.float64)
    meme_rgb = meme_rgb / (meme_rgb.max() + 1e-9)
    fig_meme = px.imshow(meme_rgb, origin="lower")
    fig_meme.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_meme, width='stretch')

    st.markdown("#### Biogenic Bloom Map (Gray–Scott Morphogenesis × Elemental Chemistry)")
    st.caption("V's elevation of the same idea: R/G channels are real Turing reaction-diffusion "
               "morphogens (Gray-Scott, 1983); Blue is which element dominates that cell's chemistry "
               "(chemistry.py's real periodic-table profile table, projected from world resources).")
    hue = dominant_element_hue_map(world)
    rgb = biome_rgb(A, B, hue)
    fig_bloom = px.imshow(rgb, origin="lower")
    fig_bloom.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_bloom, width='stretch')

    with st.expander("Pheromone channels (16-channel stigmergic trail system)"):
        channel = st.selectbox("Channel", PHEROMONE_NAMES, index=0)
        idx = PHEROMONE_NAMES.index(channel)
        fig_ph = px.imshow(world.pheromone_grid[:, :, idx], origin="lower", color_continuous_scale="inferno")
        fig_ph.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_ph, width='stretch')


def render_consciousness_inspector() -> None:
    population = st.session_state.population
    alive = [p for p in population if p.alive]
    if not alive:
        st.warning("No living agents to inspect.")
        return

    ids = [p.agent_id for p in alive]
    selected_id = st.selectbox("Select an agent", ids, index=0)
    agent = next(p for p in alive if p.agent_id == selected_id)
    hrc = agent.hrc

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Role", agent.role)
    c2.metric("Generation", agent.generation)
    c3.metric("Age", agent.age)
    c4.metric("Energy", f"{agent.energy:.2f}")

    spec = hrc.spectral_summary()
    st.markdown(f"**Φ (last):** {spec['phi_last']:.4f} &nbsp;&nbsp; "
                f"**Confidence:** {spec['confidence']:.3f} &nbsp;&nbsp; "
                f"**Crystallised attractors:** {spec['crystallized']}")

    if hrc.phi_history:
        fig_phi = go.Figure()
        fig_phi.add_trace(go.Scatter(y=hrc.phi_history[-200:], line=dict(color="#f6ad55")))
        fig_phi.update_layout(title="Φ history (this agent)", height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_phi, width='stretch')

    fig_emo = go.Figure(go.Barpolar(
        r=[float(hrc.emotion(e)) for e in EMOTION_NAMES] + [float(hrc.emotion(EMOTION_NAMES[0]))],
        theta=EMOTION_NAMES + [EMOTION_NAMES[0]],
        marker_color="#4fd1c5",
    ))
    fig_emo.update_layout(title="Emotional state", height=350, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_emo, width='stretch')

    st.markdown("#### Genome (read directly off this agent's Hamiltonian eigenbasis)")
    fp = genome_fingerprint(hrc.eigenvectors())
    st.code(fp["dna"], language=None)
    st.caption(f"{len(fp['codons'])} codons -> protein: {' - '.join(fp['protein'][:8])}"
               + (" ..." if len(fp["protein"]) > 8 else ""))

    st.markdown("#### Game of Life scratchpad (seeded from this agent's own DNA)")
    fig_gol = px.imshow(agent.gol_grid, color_continuous_scale="greys", origin="lower")
    fig_gol.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig_gol, width='stretch')


def render_civilization() -> None:
    civ: Civilization = st.session_state.civ
    population = st.session_state.population

    active_tribes = {tid: t for tid, t in civ.tribes.items() if t.members(population)}
    if not active_tribes:
        st.info("No tribes have formed yet — advance the simulation.")
        return

    rows = []
    for tid, tribe in active_tribes.items():
        power = civ.compute_tribal_power(tid, population)
        rows.append({
            "Tribe": tid, "Members": len(tribe.members(population)),
            "Discoveries": tribe.discoveries, "Alliances": len(tribe.alliances),
            "Power": round(power, 2), "Color": f"rgb{tribe.color}",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    st.markdown("#### Tech Tree")
    n_nodes = civ.tech_tree.number_of_nodes()
    if n_nodes == 0:
        st.info("No inventions registered yet.")
    else:
        recent_nodes = list(civ.tech_tree.nodes)[-TECH_TREE_RENDER_LIMIT:]
        sub = civ.tech_tree.subgraph(recent_nodes)
        st.caption(f"Showing the most recent {len(recent_nodes)} of {n_nodes} total tech-tree nodes "
                   f"(a force-directed layout of the full graph would be unreadable and slow at scale).")
        pos = nx.spring_layout(sub, seed=1)
        edge_x, edge_y = [], []
        for u, v in sub.edges():
            edge_x += [pos[u][0], pos[v][0], None]
            edge_y += [pos[u][1], pos[v][1], None]
        fig_tree = go.Figure()
        fig_tree.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="#555"), hoverinfo="none"))
        fig_tree.add_trace(go.Scatter(
            x=[pos[n][0] for n in sub.nodes()], y=[pos[n][1] for n in sub.nodes()],
            mode="markers", marker=dict(size=8, color="#4fd1c5"),
        ))
        fig_tree.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_tree, width='stretch')

    if civ.events:
        st.markdown("#### Diplomacy Log")
        for e in civ.events[-15:][::-1]:
            st.text(e)


def render_narrative() -> None:
    narrative: NarrativeEngine = st.session_state.narrative
    if not narrative.traditions:
        st.info("No traditions founded yet — a tribe needs to exist first.")
        return

    tribe_id = st.selectbox("Tribe", list(narrative.traditions.keys()))
    fid_history = narrative.fidelity_history.get(tribe_id, [])
    tradition = narrative.traditions[tribe_id]

    c1, c2, c3 = st.columns(3)
    c1.metric("Inter-Generational Fidelity", f"{tradition.fidelity():.3f}")
    c2.metric("Motifs drifted", f"{tradition.hamming_drift()} / {len(tradition.origin_sequence)}")
    c3.metric("Generations tracked", narrative.generations_tracked(tribe_id))

    if narrative.milestones_reached.get(tribe_id):
        st.success("✅ Milestone Reached: Stable Traditions")

    if len(fid_history) >= 2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=fid_history, line=dict(color="#f6ad55")))
        fig.add_hline(y=0.95, line_dash="dot", annotation_text="stability threshold")
        fig.update_layout(title="Tradition Persistence over generations", height=320,
                           margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Founding myth (Gödel-encoded, same machinery as any invention)")
    st.code(" → ".join(ALL_PRIMITIVES[i] for i in tradition.origin_sequence), language=None)
    st.markdown("#### Current retelling")
    st.code(" → ".join(ALL_PRIMITIVES[i] for i in tradition.current_sequence), language=None)


def render_nobel() -> None:
    nobel: NobelCommittee = st.session_state.nobel
    if not nobel.laureates:
        st.info("No laureates yet. Six categories are being watched: Physics, Chemistry, "
                 "Physiology or Medicine, Literature, Peace, Economics.")
        return

    rows = [{"Category": l.category, "Tick": l.tick, "Detail": l.detail} for l in nobel.laureates[::-1]]
    st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    counts = pd.Series([l.category for l in nobel.laureates]).value_counts()
    fig = px.bar(x=counts.index, y=counts.values, labels={"x": "Category", "y": "Laureates"})
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')


def render_evolution() -> None:
    evo: EvolutionEngine = st.session_state.evo
    population = st.session_state.population
    hist = st.session_state.history

    c1, c2, c3 = st.columns(3)
    c1.metric("Phylogenetic clades", evo.phylo.n_clades)
    c2.metric("Tradition verified", "Yes" if evo.tradition_verified else "No")
    c3.metric(
        "Cultural ratchet r",
        f"{evo.cultural_ratchet_history[-1]:.3f}" if evo.cultural_ratchet_history else "—",
        help="0.55 is the stated verification bar",
    )

    if evo.archetypes:
        counts: Dict[str, int] = {}
        for v in evo.archetypes.values():
            counts[v] = counts.get(v, 0) + 1
        fig = px.pie(names=list(counts.keys()), values=list(counts.values()),
                     title="Behavioral archetypes (KMeans on first 4 task-band eigenvalues)")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width='stretch')

    valid_r = [r for r in hist["cultural_ratchet"] if not (isinstance(r, float) and np.isnan(r))]
    if len(valid_r) >= 2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=valid_r, line=dict(color="#4fd1c5")))
        fig2.add_hline(y=0.55, line_dash="dot", annotation_text="verification bar")
        fig2.update_layout(title="Cultural ratchet r over time", height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, width='stretch')

    role_counts: Dict[str, int] = {}
    for p in population:
        if p.alive:
            role_counts[p.role] = role_counts.get(p.role, 0) + 1
    if role_counts:
        fig3 = px.bar(x=list(role_counts.keys()), y=list(role_counts.values()),
                      title="Caste distribution (Forager / Processor / Warrior / Queen)")
        fig3.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, width='stretch')


PANEL_RENDERERS = {
    PANELS[0]: render_observation_deck,
    PANELS[1]: render_biome_cartography,
    PANELS[2]: render_consciousness_inspector,
    PANELS[3]: render_civilization,
    PANELS[4]: render_narrative,
    PANELS[5]: render_nobel,
    PANELS[6]: render_evolution,
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
