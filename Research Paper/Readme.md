<div align="center">

# ⬡ GeNeSIS V — OMNIGENESIS

## A Coupled Mathematical Laboratory for Artificial Life

**Quantum-formalism cognition · Unified cognitive/genomic representation · Ecology · Chemistry · Evolution · Civilization · Statistical inference**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.35%2B-FF4B4B)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/numpy-1.26%2B-013243)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/scipy-1.12%2B-8CAAE6)](https://scipy.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://www.apache.org/licenses/LICENSE-2.0)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)]()

**GeNeSIS I → II → III → IV → V**

</div>

---

## Abstract

GeNeSIS V (OMNIGENESIS) is a from-scratch artificial-life simulation designed as a coupled mathematical system rather than a collection of independent game mechanics. The implementation combines a 64-dimensional complex cognitive state, a personal Hermitian Hamiltonian, Born-rule action selection, an integrated-information-inspired state-integration statistic, Hamiltonian-derived genomic encoding, Gray–Scott reaction–diffusion morphogenesis, a procedural ecological world, stigmergic pheromone and meme fields, rate-limited molecular chemistry, population evolution, social organization, technology formation, cultural transmission, topology, and a large statistical-analysis layer.

The principal architectural construction is the reuse of the same 64-dimensional spectral substrate for both cognition and genotype. The standard four-base codon alphabet has 4³ = 64 possible codons; GeNeSIS V maps the phase structure of the Hamiltonian eigenbasis into a 64-symbol nucleotide string and translates that string with the standard genetic code. This is an **architectural encoding**, not a claim that cognition and biological genotype are physically identical. The advantage is systems-level: the simulation does not need two independently mutable representations and a synchronization mechanism between them.

GeNeSIS V is deliberately engineered around a low-memory execution regime. The active frontend uses a 56 × 56 world, capped history buffers, bounded population dynamics, immediate pruning of deceased agents, procedural terrain generation, lazy panel execution, and a NumPy implementation of the frozen physics oracle. The analytical layer is correspondingly broad: it computes diversity, inequality, spatial statistics, time-series structure, distributions, information-theoretic measures, network metrics, ecological fits, inferential tests, signal-processing and causality diagnostics, statistical-mechanics proxies, and image/morphology descriptors.

The repository is a **research prototype**. The mathematical mechanisms are implemented and tested; this does not establish that the resulting artificial system is biologically realistic, physically quantum, phenomenally conscious, or open-ended in the strongest formal sense. Those distinctions are fundamental to the interpretation of the project.

**Keywords:** artificial life; agent-based modeling; Hilbert-space dynamics; quantum-formalism cognition; genomic encoding; reaction–diffusion; evolutionary computation; stigmergy; social dynamics; statistical inference; complex systems.

### Table of Contents

1. [Scientific Position and Scope](#1-scientific-position-and-scope)
2. [Central Construction: One Spectral Substrate](#2-central-construction-one-spectral-substrate)
3. [Mathematical Model](#3-mathematical-model)
4. [Quantitative Analysis Framework](#4-quantitative-analysis-framework)
5. [Integrated Simulation Cycle](#5-integrated-simulation-cycle)
6. [Software Architecture](#6-software-architecture)
7. [Interactive Research Interface](#7-interactive-research-interface)
8. [Research Measurements and Reproducibility](#8-research-measurements-and-reproducibility)
9. [Verification and Tests](#9-verification-and-tests)
10. [Engineering Constraints and Memory Discipline](#10-engineering-constraints-and-memory-discipline)
11. [Scientific Foundations and Adopted Formalisms](#11-scientific-foundations-and-adopted-formalisms)
12. [Project Lineage](#12-project-lineage)
13. [Installation](#13-installation)
14. [Suggested Experimental Protocol](#14-suggested-experimental-protocol)
15. [Known Limitations](#15-known-limitations)
16. [Current Research Directions](#16-current-research-directions)
17. [Module Reference](#17-module-reference)
18. [Citation / Reference List](#18-citation--reference-list)
19. [License](#19-license)
20. [Research Statement](#20-research-statement)

---

## 1. Scientific Position and Scope

### 1.1 What the system is

GeNeSIS V is a discrete-time, agent-based dynamical system in which the state at tick `t` can be written schematically as

```math
X_t = \left( W_t,\; \{A_i(t)\}_{i=1}^{N_t},\; C_t,\; E_t,\; M_t,\; T_t,\; R_t \right),
```

where:

- `W_t` denotes the environmental state: resources, temperature, pheromones, memes, structures, season and weather;
- `A_i(t)` denotes agent-level state;
- `C_t` denotes civilization state: tribes, alliances, memories and the technology graph;
- `E_t` denotes evolutionary state and population history;
- `M_t` denotes molecular/chemical state and chemical reference data;
- `T_t` denotes analytical time-series state and historical observables;
- `R_t` denotes the collection of derived diagnostics.

The global simulation is therefore not a single equation but a composition of coupled transition operators:

```math
X_{t+1} = \mathcal{G}\!\left(\mathcal{C}\!\left(\mathcal{E}\!\left(\mathcal{A}\!\left(\mathcal{W}(X_t)\right)\right)\right)\right),
```

with the important practical property that every operator operates on explicit state rather than hidden external services.

### 1.2 What the system is not

The use of quantum notation does not imply quantum hardware or microscopic quantum dynamics. The cognitive engine is executed on classical CPU hardware and uses Hilbert-space mathematics as a modeling formalism.

The quantity called `Φ` in the implementation is **not a full implementation of Integrated Information Theory**. It is a compact, reproducible integration proxy evaluated over a fixed partition of the state vector.

The chemistry engine implements standard atomic data and standard rate equations, but it is not a first-principles molecular dynamics simulation.

The frozen `PhysicsOracle` is a deterministic neural function whose role is to define a discoverable law for agents. It is not a physical theory of the simulated world.

The thermodynamic, causal and chaos panels compute diagnostics and formal proxies. They do not, by themselves, establish real-world thermodynamic equilibrium, causal identification, or mathematical chaos in the stronger theorem-level sense.

---

## 2. Central Construction: One Spectral Substrate

### 2.1 The 64-dimensional coincidence

Each agent has a cognitive state in

```math
\psi_i(t) \in \mathbb{C}^{64}, \qquad \lVert \psi_i(t) \rVert_2 = 1.
```

The standard genetic alphabet contains four nucleotides, and triplet coding produces

```math
4^3 = 64
```

possible codons.

GeNeSIS V therefore defines a deterministic map from the Hamiltonian eigenbasis to a nucleotide string:

```math
H_i v_{ik} = \lambda_{ik} v_{ik}, \qquad k=0,1,\ldots,63,
```

followed by phase-quadrant discretization of the dominant component of each eigenvector. The result is a 64-symbol sequence

```math
D_i = d_{i,0} d_{i,1} \cdots d_{i,63}, \qquad d_{i,k} \in \{A,C,G,T\}.
```

That sequence is partitioned into complete codons plus one remainder symbol:

```math
64 = 3\cdot21 + 1.
```

The first 63 nucleotides therefore define 21 codons, while the final nucleotide is retained as an unpaired remainder.

### 2.2 Architectural consequence

The mapping can be summarized as

```math
H_i \longrightarrow \{v_{ik}\}_{k=0}^{63}
\longrightarrow D_i
\longrightarrow \text{codons}
\longrightarrow \text{amino-acid symbols}.
```

Consequently, a perturbation to `H_i` can change both the future cognitive trajectory and the derived genotype. This coupling applies to learning, inherited variation and Hamiltonian-based developmental readout without requiring a second genome object that must be manually synchronized.

The correct interpretation is therefore **shared representation**, not biological identity.

---

## 3. Mathematical Model

## 3.1 Cognitive state and Hermitian dynamics

The cognitive core is a complex state vector normalized to unit Euclidean norm:

```math
\psi(t) \in \mathbb{C}^{K}, \qquad K=64, \qquad \psi^\dagger\psi=1.
```

Each agent owns a Hermitian Hamiltonian `H`, satisfying

```math
H = H^\dagger.
```

Its eigendecomposition is

```math
H = V\,\operatorname{diag}(\lambda_0,\lambda_1,\ldots,\lambda_{K-1})\,V^\dagger.
```

The implementation propagates the state by applying the exact matrix exponential represented in the eigenbasis:

```math
\psi(t+\Delta t)
=V\,\operatorname{diag}\!\left(e^{-i\lambda_0\Delta t},\ldots,e^{-i\lambda_{K-1}\Delta t}\right)V^\dagger\psi(t).
```

Because each factor has unit modulus and `V` is unitary, the update is norm-preserving up to floating-point error.

### 3.1.1 Observation-conditioned action selection

The environment supplies a finite-dimensional observation vector. The cognitive engine transforms the observation through a Fourier representation and uses the transformed signal to construct candidate action states from the eigenbasis.

For candidate basis states `b_a`, action likelihood is proportional to a Born-style overlap:

```math
P(a\mid\psi,o) \propto \left|\langle b_a(o),\psi\rangle\right|^2.
```

The implementation then applies a curiosity-dependent temperature to modulate the sharpness of the action distribution. The use of the Born rule here is formal: it is a stochastic policy parameterization expressed in the language of quantum probability.

### 3.1.2 Entropic cognitive cost

The code uses the Shannon entropy of the current measurement distribution as an information-processing signal and also exposes a von Neumann entropy routine for density matrices. The measurement entropy is

```math
S_{\mathrm{meas}}(\psi) = -\sum_{k=0}^{K-1} p_k\ln p_k,
\qquad p_k = |\psi_k|^2.
```

A learning step incurs a Landauer-inspired computational cost proportional to an entropy change:

```math
c_{\mathrm{learn}} = \kappa\,|\Delta S|.
```

The implementation treats this as a **model-level information cost**, not as a direct thermodynamic measurement of software.

### 3.1.3 Hermitian learning update

Learning modifies the Hamiltonian through a reward-signed, meta-modulated low-rank perturbation. Abstractly, the update has the form

```math
\Delta H = \eta\,m(r,\psi,\text{meta})\,\left(uu^\dagger\right),
```

where `u` is an observation/action-derived direction, `η` is the learning scale and the scalar factor `m(·)` carries the reward and meta-cognitive modulation. Hermitian symmetrization is enforced where required:

```math
H \leftarrow \frac{H+H^\dagger}{2}.
```

This makes Hermiticity a maintained invariant rather than a visual assumption.

---

## 3.2 Integrated-information-inspired integration proxy

The current implementation partitions the 64-dimensional state into a 24-dimensional task subspace and a 40-dimensional complement:

```math
\psi = \begin{bmatrix}\psi_{\mathrm{task}}\\\psi_{\mathrm{comp}}\end{bmatrix},
\qquad
\dim(\psi_{\mathrm{task}})=24,
\qquad
\dim(\psi_{\mathrm{comp}})=40.
```

For any state block `x`, define

```math
I(x)=\log_2\!\left(1+\operatorname{Var}(|x|)\right).
```

The implemented integration statistic is

```math
\Phi = \max\!\left(0,\;I(\psi_{\mathrm{task}})+I(\psi_{\mathrm{comp}})-I(\psi)\right).
```

A sustained rise is flagged when the trailing ten-step mean exceeds `1.2` times a reference mean from the earlier 40–50-step window. This provides a reproducible longitudinal statistic while remaining explicitly distinct from a validated measure of phenomenal consciousness.

---

## 3.3 Meta-cognition, Gödel encoding and novelty

Behavior is represented over a fixed primitive vocabulary. Programs are Gödel-encoded as unique integers, allowing symbolic behavioral structures to be compared without assigning meaning to the integer itself.

A generic prime-product encoding can be written as

```math
G(P)=\prod_{j=1}^{L} q_j^{e_j},
```

where `q_j` are fixed primes and `e_j` encode the primitive/position structure. Program distance is subsequently computed from the integer representation rather than from an arbitrary neural embedding alone.

Novelty is evaluated against a memory of prior programs. The system uses novelty as a reward-shaping signal and can promote a discovery to breakthrough status under its configured statistical threshold.

The architecture also contains a 40-dimensional meta-cognitive band. Its purpose is not to replace the task state but to model parameters governing how learning, exploration and self-reference evolve.

---

## 3.4 Genetic readout and developmental representation

The genome is not independently stored. It is obtained from the Hamiltonian eigenbasis as described above. The same string is then passed through the standard genetic code mapping to obtain translated codons.

Genetic similarity is computed from the derived representation, allowing reproduction and population analysis to operate over a phenotype/genotype bridge that is already present in the cognitive state.

A second biological substrate is a binary 8 × 8 Conway's Game of Life scratchpad seeded directly from the DNA readout. Bases `G` and `T` are mapped to active cells; the remaining bases map to inactive cells.

The Game of Life transition is

```math
s_{t+1}(x,y)=\begin{cases}
1, & n_t(x,y)=3,\\
1, & s_t(x,y)=1\ \text{and}\ n_t(x,y)=2,\\
0, & \text{otherwise},
\end{cases}
```

with toroidal boundary conditions.

This does not claim that biological genomes perform cellular automata. It is an internal computational substrate designed to expose another transformation of the same derived sequence.

---

## 3.5 Gray–Scott morphogenesis

The spatial pattern generator follows the Gray–Scott reaction–diffusion equations:

```math
\frac{\partial A}{\partial t}
= D_A\nabla^2 A - AB^2 + f(1-A),
```

```math
\frac{\partial B}{\partial t}
= D_B\nabla^2 B + AB^2 - (f+k)B.
```

The frontend advances the morphogen field three times per simulation tick. The spatial operator is a five-point discrete Laplacian on a toroidal grid:

```math
\nabla^2 u_{i,j}
\approx u_{i+1,j}+u_{i-1,j}+u_{i,j+1}+u_{i,j-1}-4u_{i,j}.
```

The same discrete operator is reused for environmental heat diffusion. This is a deliberate implementation reuse: one numerically simple spatial operator serves multiple coupled fields.

The Gray–Scott output is visualized alongside a chemistry-derived dominant-element field, making the Biome Cartography panel a compositional projection rather than a purely aesthetic texture generator.

---

## 3.6 World physics: procedural terrain, heat and resources

The world is a toroidal lattice. The active frontend uses

```math
H_W=W_W=56.
```

Terrain is generated procedurally from a deterministic integer hash followed by interpolated value noise. A multi-octave field is formed as

```math
F(x,y)=\frac{\sum_{o=0}^{O-1} a_o\,V_o(x,y)}{\sum_{o=0}^{O-1} a_o},
\qquad a_o=a_0\,p^o,
```

with default persistence `p = 0.5` and lacunarity `2.0`.

The procedural generator is deterministic for fixed `(seed, x, y)`. It therefore avoids materializing a global terrain volume.

### 3.6.1 Heat diffusion

The temperature field evolves according to an explicit discrete diffusion step with diffusivity `0.12` and a seasonal forcing term:

```math
T_{t+1}=T_t+\alpha\nabla^2T_t+0.03\,\sin\!\left(\frac{2\pi t}{200}\right),
\qquad \alpha=0.12.
```

The result is bounded to the configured interval `[0,2]`.

### 3.6.2 Seasons and ecological regeneration

Season phase is

```math
s(t)=\sin\!\left(\frac{2\pi t}{200}\right).
```

The resource update combines weather amplitude, seasonality, fertility, temperature and population-density damping. In conceptual form,

```math
\Delta R
=0.04\;A_w\;\frac{1}{1+\rho_N}\;\left(0.5+0.5T\right)\;\left(1+0.3s(t)\right)\;F(x,y),
```

where `A_w` is the weather amplitude, `ρ_N` is local population density and `F` is the procedural fertility field.

Resource channels are clipped to the interval `[0,5]`.

### 3.6.3 Stigmergic fields

The environment contains sixteen pheromone channels and eight meme channels. Their generic update follows

```math
S_{t+1}=\operatorname{clip}\!\left(\delta S_t+\gamma\nabla^2S_t,\;0,\;10\right).
```

The configured decay/diffusion constants are:

| Field | Channels | Decay | Diffusion |
|---|---:|---:|---:|
| Pheromones | 16 | 0.97 | 0.10 |
| Memes | 8 | 0.995 | 0.05 |

The slower meme decay intentionally gives culture a longer temporal memory than volatile local scent.

The sixteen pheromone channels are `trail`, `alarm`, `food`, `water`, `mate`, `territory`, `help`, `danger`, `gathering`, `migration`, `nest`, `hunt`, `rest`, `play`, `mourning`, and `celebration`. The eight meme channels are `danger`, `resource`, `sacred`, `territory`, `curiosity`, `authority`, `kinship`, and `mystery`.

### 3.6.4 Environmental observation vector

A local observation consists of four resource means, sixteen pheromone channels, eight meme channels and local heat:

```math
\dim(o_t)=4+16+8+1=29.
```

The same world exposes an eight-dimensional feature vector to the frozen PhysicsOracle: the four resources, heat, season phase, weather amplitude and a reserved zero feature.

---

## 3.7 Frozen PhysicsOracle

The PhysicsOracle is a deterministic, never-trained three-layer NumPy multilayer perceptron:

```math
x\in\mathbb{R}^{8}
\xrightarrow{\tanh,W_1,b_1}
\mathbb{R}^{16}
\xrightarrow{\tanh,W_2,b_2}
\mathbb{R}^{16}
\xrightarrow{\tanh,W_3,b_3}
\mathbb{R}^{4}.
```

Formally,

```math
h_1=\tanh(xW_1+b_1),
```

```math
h_2=\tanh(h_1W_2+b_2),
```

```math
y=\tanh(h_2W_3+b_3).
```

Weights are seeded and never trained. An agent or civilization may therefore attempt to infer a stable mapping from observable environmental quantities to the latent oracle outputs. The scientific interpretation is that the oracle supplies a fixed latent law for system-identification experiments.

---

## 3.8 Embodied agent lifecycle

The current agent algebra contains twenty actions:

`move_n`, `move_ne`, `move_e`, `move_se`, `move_s`, `move_sw`, `move_w`, `move_nw`, `eat`, `attack`, `communicate`, `reproduce`, `invent`, `rest`, `build_artifact`, `absorb_artifact`, `meta_invent`, `compose_action`, `trade`, `punish`.

The nominal fixed energy costs include

```math
c_{\mathrm{move}}=0.0005,
\quad
c_{\mathrm{attack}}=0.20,
\quad
c_{\mathrm{invent}}=0.15,
\quad
c_{\mathrm{build}}=0.05,
```

```math
c_{\mathrm{meta}}=0.06,
\quad
c_{\mathrm{compose}}=0.02,
\quad
c_{\mathrm{trade}}=0.005,
\quad
c_{\mathrm{punish}}=0.10.
```

A basal metabolic cost is charged every tick:

```math
c_{\mathrm{basal}}=0.010.
```

The survival override activates below

```math
E<0.4,
```

and forces an `eat` action. A reproduction cooldown of 25 ticks prevents unrestricted immediate rebirth.

### 3.8.1 Mode bias

Emotion and integration state modulate the action sampled from the cognitive policy. In particular:

- severe fear redirects the policy toward rest;
- critical hunger overrides other drives and selects eating;
- sufficiently integrated or discovery-rich agents redirect an attack impulse toward invention;
- highly angry, low-integration agents with diffuse action probabilities can be biased toward attack.

These rules are explicit state-dependent transformations, not hidden reward terms.

### 3.8.2 Role allocation and artifacts

Roles are assigned from population and spectral state. A `Queen` is eligible when the population exceeds 80 agents and the candidate energy exceeds 1.5. Remaining agents are partitioned into behavioral roles using tertiles of Hamiltonian eigenvalue spread.

Construction maps role state to explicit world structures:

| Role | Primary artifact | World effect |
|---|---|---|
| Warrior | Trap | damage parameter = 0.3 |
| Processor | Battery | storage capacity = 10 |
| Forager / Queen | Cultivator | radius = 3, resource boost = 1.5 |

The world also contains cooperative mega-resources. A harvest is permitted only when at least two agents are co-located, making cooperation an explicit physical precondition rather than a reward label.

### 3.8.2 Reproduction

The reproduction cost grows with population size:

```math
C_{\mathrm{rep}}=0.35\left(1+0.5\left(\frac{N}{128}\right)^2\right).
```

A compatible partner requires a minimum spectral resonance. Child state inheritance uses a sampled parental weight

```math
\alpha\sim U(0.35,0.65),
```

with cognitive-state mixing

```math
\psi_c
=\alpha\psi_a+(1-\alpha)\psi_b+\epsilon_\psi,
```

followed by normalization, and Hamiltonian inheritance of the form

```math
H_c
=\alpha H_a+(1-\alpha)H_b+\epsilon_H,
```

followed by Hermitian projection. The meta-Hamiltonian receives a separate weighted combination with additional perturbation.

### 3.8.4 Death and inherited memory

Death does not simply discard state. A spectral death packet contains spectral information, a meta-Hamiltonian fragment, discoveries, a soul fragment and high-value causal-action information. Neighboring agents can absorb this information before the deceased object is pruned from the live population.

This produces a computational analogue of cultural/spectral inheritance while preserving the memory bound by removing the full dead-agent state from the active population list.

---

## 3.9 Spectral resonance and synchronization

For two agents `a` and `b`, the resonance statistic uses the first task-band eigenvalues. With eigenvalue vectors `λ_a` and `λ_b`,

```math
\rho(a,b)=\max\!\left(0.015,\frac{1+\operatorname{cos}(\lambda_a,\lambda_b)}{2}\right).
```

Communication uses resonance-weighted state blending. If `σ_s=ψ_s\odot\omega_s` denotes the sender's soul-modulated broadcast, the receiver state is updated as

```math
\psi_r'
=\frac{(1-0.07\rho)\psi_r+0.07\rho\,\sigma_s}
{\left\|(1-0.07\rho)\psi_r+0.07\rho\,\sigma_s\right\|}.
```

### Kuramoto coupling

The oscillator phase evolves every five ticks. For neighbors `N_i`,

```math
\dot{\theta}_i
=\omega_i+\frac{\kappa}{|N_i|}\sum_{j\in N_i}\sin(\theta_j-\theta_i),
```

with configured

```math
\kappa=0.5,\qquad \Delta t=0.1.
```

Population synchronization is summarized by the complex order parameter

```math
R=\left|\frac{1}{N}\sum_{j=1}^{N}e^{i\theta_j}\right|.
```

`R ≈ 1` corresponds to strong phase alignment, while small `R` indicates broad phase dispersion.

---

## 3.10 Evolution, population control and behavioral archetypes

Population dynamics are explicitly bounded. The configured floor and ceiling are 28 and 128 agents, respectively. Meta-fitness-weighted immigration and culling maintain these limits.

The model also computes behavioral archetypes using KMeans over the first four task-band spectral eigenvalues. The current labels are:

| Archetype | Interpretation |
|---|---|
| Explorer | mobility / information seeking |
| Builder | construction / artifact activity |
| Fighter | aggressive / defensive action profile |
| Thinker | cognition / invention-heavy profile |

The labels are descriptive clusters, not ontological species.

### Cultural-ratchet statistic

The evolution layer compares founder and descendant action-frequency profiles using Pearson correlation. For vectors `x` and `y`,

```math
r_{xy}
=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}
{\sqrt{\sum_i(x_i-\bar{x})^2}\sqrt{\sum_i(y_i-\bar{y})^2}}.
```

High positive values indicate persistence of behavioral structure across generations under the implemented measure.

---

## 3.11 Civilization, tribes, diplomacy and technology

### Tribe assignment

An unaffiliated agent samples up to six members of each existing tribe and computes mean spectral resonance. The agent joins the highest-scoring tribe when

```math
\bar{\rho}>0.08;
```

otherwise a new tribe is founded.

A founder's spectral RGB color is derived from the minimum, median and maximum Hamiltonian eigenvalues and normalized into a fixed 0–255 range.

### Tribal power

The exact implemented eight-term power function is

```math
P_T
=0.30\sum_i E_i
+0.20\sum_i H_i
+2.5D
+0.55N
+1.5A
+3.0\,\overline{\sigma_{\mathrm{meta}}}
+5.0\,\overline{\Phi}
+0.10Q,
```

where:

- `E_i` is member energy;
- `H_i` is member health;
- `D` is tribe discovery count;
- `N` is current membership;
- `A` is alliance count;
- `σ_meta` is meta-eigenspectrum spread;
- `Φ` is mean integration proxy;
- `Q` is aggregate trade volume.

### Diplomacy

Alliance rolls occur when the power ratio lies inside the interval

```math
0.5<\frac{P_A}{P_B}<2.0,
```

with a configured roll probability of `0.60`. War rolls require an extreme disparity,

```math
\frac{P_A}{P_B}>3
\qquad\text{or}\qquad
\frac{P_A}{P_B}<\frac13,
```

with a configured probability of `0.02`.

Diplomacy is evaluated every 20 ticks.

### Epistemic schism

Each tribe has a mean meta-Hamiltonian. Two tribes have epistemic distance

```math
D_{AB}=\left\|\lambda(H_A^{\mathrm{meta}})-\lambda(H_B^{\mathrm{meta}})\right\|_2.
```

An alliance is eligible for dissolution once

```math
D_{AB}>48.
```

Cultural assimilation continuously moves members toward their tribe's average meta-Hamiltonian:

```math
H_i' = 0.95H_i+0.05H_T,
```

followed by Hermitian projection.

### Technology graph

Each invention becomes a node in a directed technology graph. The new node is linked to its nearest existing node by Gödel-distance before being stored. The global technology multiplier grows as

```math
B_{t+1}=\min(3.0,\;1.003\,B_t).
```

A singularity override promotes a discovery when both conditions hold:

```math
n_{\mathrm{novelty}}>0.55
```

and

```math
|V_{\mathrm{tech}}|\ge 15.
```

This is a repository-specific breakthrough criterion, not a claim of technological singularity in the physical world.

### Liquid Dunbar bound

Civilizational social capacity is linked to technology through

```math
N_{\max}^{\mathrm{tribe}}=12+3|V_{\mathrm{tech}}|.
```

---

## 3.12 Oral tradition and cultural drift

A founding myth is a sequence of 16 symbols drawn from the same behavioral primitive vocabulary used elsewhere in the system.

Under the implemented Swadesh-style retention model, each motif has a per-generation retention probability `r`. The nominal value is

```math
r_0=0.805,
```

and sacred local meme influence can increase it toward

```math
r_{\mathrm{sacred}}=0.955.
```

For `L` motifs, expected retained motif count after `g` generations is

```math
\mathbb{E}[L_g]=Lr^g.
```

Fidelity is evaluated using Pearson correlation between founding and current motif representations. The model therefore separates **transmission persistence** from mere symbolic identity.

---

## 3.13 Computable Breakthrough Evaluation

The evaluation layer contains six Nobel-inspired categories. They are deterministic computational criteria, not claims that simulated events are equivalent to the real Nobel Prizes.

| Category | Implemented trigger concept |
|---|---|
| Physics | local causal model demonstrates out-of-distribution generalization of the frozen world oracle |
| Chemistry | sufficiently novel thermodynamically exergonic molecule or reaction |
| Physiology or Medicine | lineage longevity is a statistical outlier relative to other lineages |
| Literature | myth changes measurably while retaining recognizable structure |
| Peace | alliance persists beyond the configured temporal criterion |
| Economics | trade-network efficiency improves by a meaningful margin subject to a minimum interval |

The layer converts qualitative milestones into explicit, inspectable state transitions.

---

## 3.14 Chemistry and reaction kinetics

The chemistry layer uses atomic data and a molecular-species abstraction. It distinguishes between:

1. a bulk elemental-composition field used for environmental visualization; and
2. an explicit molecular-species pool used for reactions.

The reaction-rate law is the Arrhenius equation:

```math
k(T)=A\exp\!\left(-\frac{E_a}{RT}\right).
```

Per-step conversion is gated by the limiting reactant, and built-in tests verify atom conservation for the reference reactions.

For a reaction converting reactants to products, the implementation checks equality of total elemental counts before and after the transition:

```math
\sum_j n_{j,e}^{(\mathrm{reactants})}
=
\sum_j n_{j,e}^{(\mathrm{products})}
\qquad\forall e.
```

The chemistry reference layer contains fifteen biologically relevant elements, IUPAC atomic weights, Pauling electronegativities and common molecular constants including glucose, water, carbon dioxide and oxygen.

---

## 3.15 Geometry and topology

The geometry subsystem contains Lindenmayer-system string rewriting, turtle rendering, and discrete topological invariants.

For a structured binary field, the first two Betti numbers are interpreted as

```math
\beta_0=\text{number of connected components},
```

```math
\beta_1=\text{number of enclosed one-dimensional holes}.
```

The Euler characteristic is

```math
\chi=\beta_0-\beta_1.
```

The implementation validates canonical cases including one connected loop with one enclosed hole and two disconnected regions.

The morphology extension adds convex-hull solidity, circularity, aspect ratio, orientation coherence and local-binary-pattern texture entropy.

Circularity is based on

```math
C=\frac{4\pi A}{P^2},
```

where `A` is area and `P` is the perimeter estimator. Solidity is

```math
S=\frac{A}{A_{\mathrm{hull}}}.
```

---

# 4. Quantitative Analysis Framework

The current `analytics.py` module is a standalone quantitative-analysis library designed to remain safe on empty and degenerate inputs: undefined statistics return `NaN` rather than crashing a live dashboard.

The analytical stack is organized into three conceptual tiers.

## 4.1 Tier I — Core diversity, spatial, temporal, information and network structure

### Diversity

For counts `c_i` with total `C`,

```math
p_i=\frac{c_i}{\sum_j c_j}.
```

Shannon entropy:

```math
H=-\sum_i p_i\ln p_i.
```

Simpson concentration:

```math
D=\sum_i p_i^2.
```

Inverse Simpson:

```math
{}^2D=\frac{1}{D}.
```

Pielou evenness:

```math
J'=\frac{H}{\ln S}.
```

Hill number:

```math
{}^qD
=\left(\sum_i p_i^q\right)^{1/(1-q)}.
```

The `q → 1` limit is `exp(H)` and `q = 2` recovers inverse Simpson.

### Inequality

The Gini coefficient uses the sorted-rank estimator

```math
G
=\frac{2\sum_{i=1}^{n} i x_{(i)}}{n\sum_i x_i}-\frac{n+1}{n}.
```

The Lorenz curve is the cumulative population/resource share mapping.

Theil's T index is

```math
T=\frac{1}{n}\sum_i\frac{x_i}{\mu}\ln\!\left(\frac{x_i}{\mu}\right).
```

The analysis layer additionally computes Berger–Parker dominance, Hoover, Palma, Atkinson and HHI measures.

### Spatial statistics

Moran's I is calculated using rook adjacency and standardized deviations from the mean. The nearest-neighbor index uses the Clark–Evans expectation

```math
R=\frac{\bar d_{\mathrm{obs}}}{0.5/\sqrt{n/A}}.
```

Ripley's K estimator follows

```math
K(r)=\frac{N_{\mathrm{pairs}}(d\le r)}{n\lambda},
\qquad
\lambda=\frac{n}{A},
```

with

```math
L(r)=\sqrt{\frac{K(r)}{\pi}}-r.
```

Quadrat dispersion is the variance-to-mean ratio of cell counts.

### Time-series structure

The module computes autocorrelation, Hurst R/S exponent, spectral entropy, permutation entropy, Lyapunov-style divergence, detrended fluctuation analysis, Fano factor, coefficient of variation, Kendall trend, Allan variance, autocorrelation time and burstiness.

For a spectral distribution `q_k`, normalized spectral entropy is

```math
H_{\mathrm{spec}}
=-\frac{\sum_k q_k\ln q_k}{\ln M}.
```

Burstiness is

```math
B=\frac{\sigma-\mu}{\sigma+\mu}.
```

### Distributions and survival

The power-law maximum-likelihood estimator for `x ≥ x_min` is

```math
\hat\alpha
=1+\frac{n}{\sum_i\ln(x_i/x_{\min})}.
```

The module couples the MLE with a Kolmogorov–Smirnov distance rather than reporting a fitted exponent alone.

Zipf scaling is estimated from a linear fit in log-rank/log-frequency space. A Gompertz–Makeham hazard fit uses

```math
h(t)=\lambda+\alpha e^{\beta t}.
```

Kaplan–Meier survival analysis is implemented directly from observed ages, with a median-survival guard to avoid false precision on very small populations.

### Information theory

The analysis stack provides Shannon/joint entropy, mutual information, KL divergence, Jensen–Shannon divergence and a normalized compression distance:

```math
\operatorname{NCD}(x,y)
=\frac{C(xy)-\min(C(x),C(y))}{\max(C(x),C(y))}.
```

The implemented transfer-entropy routine is explicitly a **proxy** based on mutual information between lagged source and target variables rather than full conditional transfer entropy.

### Quantum-formalism diagnostics

For a density matrix with eigenvalues `p_i`, von Neumann entropy is

```math
S(\rho)=-\operatorname{Tr}(\rho\ln\rho)=-\sum_i p_i\ln p_i.
```

For a pure state, this is zero. The separate measurement-entropy routine uses Born probabilities `|ψ_i|²`.

The module also computes purity

```math
\operatorname{Purity}(\rho)=\operatorname{Tr}(\rho^2),
```

state fidelity

```math
F(\psi,\phi)=|\langle\psi,\phi\rangle|^2,
```

trace distance proxy

```math
D_{\mathrm{tr}}=\sqrt{1-F},
```

Bures angle

```math
\theta_B=\arccos(\sqrt{F}),
```

participation ratio

```math
PR=\frac{\left(\sum_i |\psi_i|^2\right)^2}{\sum_i|\psi_i|^4},
```

and inverse participation ratio `IPR = 1/PR`.

Level-spacing ratios are compared against standard spectral-statistics reference values used by the test suite, including Poisson, GOE and GUE regimes.

### Fractal and network metrics

The module computes box-counting dimension, lacunarity, degree and betweenness summaries, clustering, assortativity, components, path statistics, modularity proxies, small-world sigma, global efficiency, degree power-law behavior and centrality summaries.

---

## 4.2 Tier II — Additional ecology, complexity and structure

The second tier extends the analysis with:

| Domain | Implemented diagnostics |
|---|---|
| Diversity | Margalef, Menhinick, Chao1, Rényi, Tsallis |
| Similarity | Bray–Curtis, Jaccard, Sørensen, Whittaker beta |
| Inequality | Atkinson, Palma, Hoover, HHI |
| Complexity | Sample entropy, approximate entropy, Lempel–Ziv, Higuchi dimension |
| Recurrence | Recurrence rate, autocorrelation time |
| Spatial | Geary's C, patch statistics, contagion, centre-of-mass drift, radius of gyration |
| Demography | Doubling time, quasi-extinction risk, survivorship curve type |
| Distribution | Skewness, kurtosis, Jarque–Bera-style diagnostic, Benford deviation, mode/median/mean |

Chao1 is implemented as a richness estimator using singleton/doubleton counts. The canonical branch is

```math
\hat S_{\mathrm{Chao1}}
=S_{\mathrm{obs}}+\frac{F_1^2}{2F_2},
```

with a finite-sample correction where the doubleton count is zero.

Rényi entropy is

```math
H_\alpha
=\frac{1}{1-\alpha}\ln\!\left(\sum_i p_i^\alpha\right),
```

and Tsallis entropy is

```math
T_q
=\frac{1-\sum_i p_i^q}{q-1}.
```

For nonnegative abundance vectors, Bray–Curtis dissimilarity is

```math
BC(x,y)=\frac{\sum_i|x_i-y_i|}{\sum_i(x_i+y_i)}.
```

---

## 4.3 Tier IIIa — Statistical inference

The current frontend exposes a dedicated **⟒ Statistical Inference** panel. It is designed around inferential questions that can be answered from the current living population while still remaining explicit about sample-size guards.

### Role-wise energy inference

For at least two populated role groups, the panel reports one-way ANOVA:

```math
F=\frac{MS_{\mathrm{between}}}{MS_{\mathrm{within}}}.
```

For two groups it additionally reports Cohen's d,

```math
d=\frac{\bar x_1-\bar x_2}{s_p},
```

where `s_p` is the pooled standard deviation, Hedges' g, Cliff's delta, permutation p-value and Mann–Whitney p-value.

### Bootstrap confidence intervals

For energy, health, age and discoveries, the dashboard computes percentile bootstrap intervals from 2,000 resamples with a fixed RNG seed for reproducibility.

### Association and dependence

The panel computes Pearson and Spearman coefficients. Pearson's correlation is

```math
r=\frac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i(x_i-\bar x)^2\sum_i(y_i-\bar y)^2}}.
```

It also computes a role-versus-tribe chi-square test whenever there are at least ten tribed agents across at least two tribes:

```math
\chi^2=\sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}.
```

### Young-vs-old distribution comparison

The current implementation contains an explicit population distribution comparison that was added to the Statistical Inference panel.

To avoid the empty-group pathology caused by a strict `<= median` versus `> median` split when many agents have the same age, the live population is partitioned by **age rank**. For `N ≥ 10`, the youngest half and oldest half are selected after a stable sort by age.

For the two cohorts, the panel reports the two-sample Kolmogorov–Smirnov statistic and Welch's t statistic:

```math
D_{\mathrm{KS}}
=\sup_x |F_{\mathrm{young}}(x)-F_{\mathrm{old}}(x)|.
```

```math
t_W
=\frac{\bar x_1-\bar x_2}
{\sqrt{s_1^2/n_1+s_2^2/n_2}}.
```

The visualization overlays the **energy distributions** of the younger and older halves as two histograms and reports cohort median ages and the global age range. The test asks a precise descriptive/inferential question—whether energy distributions differ between the age-ranked cohorts—without assuming that the answer must be positive.

### Inferential safeguards

The implementation deliberately refuses to fabricate statistics when sample structure is inadequate. Typical guards include:

- at least 8 living agents for the overall inference panel;
- at least 2 observations per role for ANOVA groups;
- at least 10 living agents for the age-ranked cohort comparison;
- at least 10 tribed agents across at least two tribes for role-versus-tribe independence;
- finite-data checks inside the analytics library for every metric.

---

## 4.4 Tier IIIb — Signal processing, stationarity, causality and chaos

The third tier includes cross-correlation, lag selection, spectral coherence, Granger-style causality, ADF-like stationarity diagnostics, a 0–1 chaos test and correlation dimension.

Cross-correlation is evaluated over a symmetric lag window and normalized to enable lag comparison.

Spectral coherence follows a Welch-style decomposition and estimates the normalized frequency-domain coupling between two signals.

The Granger-style test compares restricted and unrestricted autoregressive residual sums of squares using the nested-model F statistic:

```math
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/(T-k)}.
```

The implementation uses a default lag order of two for the dashboard diagnostic.

The ADF-like statistic is used comparatively to distinguish integrated-looking random walks from stationary noise. It is labeled as an ADF-style implementation rather than presented as a full econometric test suite.

The 0–1 chaos diagnostic maps trajectory statistics toward values near zero for regular signals and near one for chaotic signals. The library self-test verifies a periodic reference and the fully chaotic logistic map at `r=4`.

Correlation dimension is estimated from delay-embedded states using a correlation-sum scaling relationship of the form

```math
C(r)\propto r^D.
```

These statistics are **diagnostics**, not causal proofs.

---

## 4.5 Tier IIIc — Statistical mechanics and morphology

The ⟟ Thermodynamics & Chaos panel treats agent energy as a kinetic-like variable and reports statistical-mechanics quantities in natural units (`k_B=1`).

### Maxwell–Boltzmann fit

For a two-dimensional speed-like variable `v`, the model is

```math
f(v)=\frac{v}{\sigma^2}\exp\!\left(-\frac{v^2}{2\sigma^2}\right).
```

The fitted scale parameter is interpreted as an effective temperature proxy.

### Equipartition

For `dof` effective degrees of freedom,

```math
T_{\mathrm{eq}}=\frac{2\langle E\rangle}{dof\,k_B}.
```

### Boltzmann entropy

For microstate count `\Omega`,

```math
S_B=k_B\ln\Omega.
```

### Gibbs entropy

For canonical state probabilities `p_i`,

```math
S_G=-k_B\sum_i p_i\ln p_i.
```

### Helmholtz free-energy proxy

Using a partition function over observed energy levels,

```math
F=-k_BT\ln Z,
```

with numerical stabilization applied before exponentiation.

### Entropy production

The dashboard reports the mean per-step entropy increment

```math
\dot S_{\mathrm{proxy}}
=\frac{1}{T-1}\sum_{t=1}^{T-1}(S_{t+1}-S_t).
```

### Virial proxy

The dimensionless virial ratio is

```math
\mathcal{V}=\frac{2\langle K\rangle}{|\langle U\rangle|}.
```

Again, these are population-level analogues, not measurements of a physical gas.

### Morphological descriptors

The morphology layer computes convex-hull solidity, circularity, moment-based aspect ratio, field orientation coherence and local-binary-pattern texture entropy.

These descriptors convert the simulated spatial fields into quantitative shape statistics so that visual structure can be analyzed rather than only displayed.

---

# 5. Integrated Simulation Cycle

A simulation tick can be represented by the following high-level sequence:

```text
World fields advance
        │
        ▼
Agent senses local resources / pheromones / memes / heat
        │
        ▼
Fourier-conditioned cognitive state → Born-rule action policy
        │
        ▼
Mode bias (hunger / fear / anger / integration)
        │
        ▼
Embodied action execution
        │
        ├── movement / eating / attack / communication
        ├── reproduction / invention / construction
        ├── trade / punishment / meta-invention
        └── artifact and cultural interactions
        │
        ▼
Reward and information-costed learning
        │
        ▼
Hamiltonian / meta-state evolution
        │
        ├── Kuramoto synchronization (cadence: 5 ticks)
        ├── role update + Game of Life (cadence: 10 ticks)
        ├── meme absorption (cadence: 20 ticks)
        └── viral broadcast (cadence: 50 ticks)
        │
        ▼
EvolutionEngine
        │
        ├── population bounds
        ├── meta-fitness
        ├── cultural-ratchet statistic
        └── phylogenetic clustering
        │
        ▼
Civilization
        │
        ├── tribe assignment
        ├── assimilation
        ├── diplomacy / schism
        └── technology graph
        │
        ▼
Narrative + Nobel-style evaluation + analytics history
        │
        ▼
Gray–Scott morphogenesis advances three substeps
```

This ordering matters because the simulation is not a static visualization. State accumulated in one tick alters the probability landscape of subsequent ticks.

---

# 6. Software Architecture

```text
GeNeSIS-V/
├── GeNeSIS_V.py          Streamlit frontend and lazy panel dispatcher
├── metacognition.py      Gödel encoding, associative memory, novelty, meta-cognition
├── consciousness.py      64D complex cognitive dynamics, Born policy, Φ proxy
├── biology.py            Hamiltonian-derived codons, genetic code, Gray–Scott, development
├── world.py              Procedural world, heat, resources, pheromones, memes, oracle
├── agents.py             BioHyperAgent lifecycle and embodied actions
├── evolution.py          Population regulation, meta-fitness, phylogeny, archetypes
├── chemistry.py          Atomic data, molecules, reactions, Arrhenius kinetics
├── civilization.py       Tribes, power, diplomacy, technology graph, schism
├── geometry.py           L-systems, topology, morphology
├── narrative.py          Myth generation and transmission
├── nobel.py              Six-category breakthrough evaluation
├── analytics.py          Quantitative analysis and statistical inference
├── test_smoke.py         Cross-module integration test
├── requirements.txt      Runtime dependencies
└── BUILD_STATUS.md       Dated engineering/calibration record
```

The current frontend uses explicit panel dispatch rather than `st.tabs()`. This is a deliberate performance architecture: Streamlit reruns the script on interaction, so only the selected panel body should execute its expensive rendering logic.

A lightweight analytics wrapper also protects the UI from missing or malformed metric calls. Functions with structured return types have shaped `NaN` fallbacks so a single unavailable statistic cannot turn into a secondary unpacking/subscriptability error in the dashboard.

---

# 7. Interactive Research Interface

The current GeNeSIS V frontend contains **14 panels**:

| Panel | Scientific/technical role |
|---|---|
| ⬡ Observation Deck | Population, tribes, technology, mean integration and long-horizon history |
| ⌬ Biome Cartography | Meme/pheromone ecology, Gray–Scott patterning, chemistry-derived field projection |
| ✦ Consciousness Inspector | Per-agent `Φ`, emotional state, DNA/codon readout, Game of Life |
| ⬢ Civilization | Tribal structure, tribal power, technology graph and diplomacy |
| ⟡ Narrative | Myth transmission and cultural fidelity |
| ✧ Nobel Committee | Six-category breakthrough ledger |
| ⏣ Evolution & Phylogeny | Archetypes, cultural ratchet, clades and population structure |
| ⎔ Chemistry Lab | Elemental data, Arrhenius kinetics, reactions and molecular properties |
| ⬠ Geometry & Topology | Betti numbers, Euler characteristic, French Flag fields, L-systems, morphology |
| ⟁ Complexity & Dynamics | Entropy, fractals, growth, recurrence, compression and temporal complexity |
| ⬣ Ecology & Inequality | Diversity, spatial statistics, distributions, survival and inequality |
| ◈ Cognitive Spectra | Spectral statistics, coherence, participation, fidelity, genome statistics |
| ⟒ Statistical Inference | ANOVA, effect sizes, bootstrap CI, correlations, KS/Welch and young-vs-old distributions |
| ⟟ Thermodynamics & Chaos | Maxwell–Boltzmann, entropy, virial diagnostics, chaos, causality and morphology |

The interface is therefore not simply a simulator display. It is an experimental instrument for interrogating the current state from multiple mathematical perspectives.

The Observation Deck's named "bloom epoch" is explicitly a display heuristic derived from technology count, tribe count and mean cultural fidelity. It is not a formally calibrated phase-transition detector.

---

# 8. Research Measurements and Reproducibility

## 8.1 Histories

The primary frontend history buffer records:

```text
tick
population
tribes
tech_nodes
b_tech
mean_phi
cultural_ratchet
```

The history cap is 400 entries. This provides sufficient longitudinal analysis for dashboard diagnostics without allowing interactive sessions to accumulate unbounded time-series memory.

## 8.2 Deterministic seeding

World terrain and the primary simulation RNG are seeded. The frozen PhysicsOracle also has a fixed seed. Analytics routines that require resampling use explicit fixed seeds where reproducibility is important, including the bootstrap confidence interval routine.

Deterministic seeding therefore gives a reproducible experimental baseline, while the stochastic action policy and evolutionary processes can still be varied deliberately by seed.

## 8.3 Statistical interpretation protocol

A dashboard statistic should be interpreted in the following order:

1. verify sample size and finite-data guards;
2. identify whether the metric is descriptive, inferential or model-dependent;
3. inspect the corresponding distributional or temporal diagnostic;
4. distinguish correlation from the directional causality diagnostics;
5. treat proxy quantities as internal diagnostics rather than real-world measurements.

This is particularly important for `Φ`, thermodynamic analogues, Granger-style tests and the 0–1 chaos statistic.

---

# 9. Verification and Tests

The project includes executable self-tests and cross-module verification. The current supplied revision was syntax-checked successfully for the updated `agents.py`, `analytics.py`, `civilization.py`, `world.py` and `GeNeSIS_V.py`.

The full `analytics.py` self-test also completed successfully on the current revision. Representative checks include:

| Test family | Verification performed |
|---|---|
| Diversity | Exact Shannon, Simpson, Gini and related reference cases |
| Spatial | Known clustered / regular field behavior for Moran and neighbor diagnostics |
| Power laws | Synthetic MLE + KS recovery |
| Survival | Kaplan–Meier / median-survival guards |
| Statistics | Cohen's d, Hedges g, Cliff's delta, bootstrap CI, permutation, ANOVA, chi-square, Pearson, Spearman, KS, Mann–Whitney, Welch |
| Signal processing | Planted lag recovery and spectral coherence |
| Causality | Known directional Granger-style process |
| Stationarity | Random walk vs stationary noise |
| Chaos | Periodic signal versus logistic map at `r=4` |
| Correlation dimension | Low-dimensional deterministic map versus noise |
| Statistical mechanics | Maxwell–Boltzmann scale, equipartition, Boltzmann entropy, Gibbs/Helmholtz limits |
| Thermodynamic proxies | Positive/zero entropy-production references and virial identity |
| Morphology | Circularity, solidity, aspect ratio, orientation and texture ordering |
| Degenerate input | Undefined metrics return `NaN` or empty arrays cleanly |

Representative self-test values from the current analytics implementation include:

```text
Cohen's d for a 5-sigma shift:       -4.966  (theory: -5.0)
Bootstrap 95% CI for mean 50:        [49.28, 50.17]
Permutation test, identical groups:   p = 0.991
Permutation test, shifted groups:     p < 0.0001
Pearson linear-reference r:           1.0000
Spearman linear-reference rho:        1.0000
Planted cross-correlation lag:        -5
Self spectral coherence:              1.000
Logistic-map 0–1 chaos K:             0.998
Maxwell–Boltzmann sigma:              2.994  (truth: 3.0)
Equipartition temperature:            3.982  (truth: 4.0)
Boltzmann entropy shift:              ln(2)
Virial ratio reference:               1.0
```

These values are **test fixtures**, not empirical claims about emergent civilization behavior.

---

# 10. Engineering Constraints and Memory Discipline

GeNeSIS V was designed around a 2 GB RAM ceiling. The implementation therefore treats memory behavior as part of the model architecture.

### 10.1 Procedural terrain

Terrain is a function of coordinates and seed rather than a permanently stored global map.

### 10.2 Bounded population

The live population is constrained to the configured interval `[28,128]`.

### 10.3 Bounded history

Historical arrays are capped at 400 entries.

### 10.4 Immediate dead-agent pruning

After death handling and spectral wisdom transfer, deceased agents are removed from the live list. This prevents cumulative growth from retaining full Hamiltonian objects that no longer participate in the simulation.

### 10.5 Lazy analytics

The frontend executes only the active panel's expensive renderer. This matters because plotting routines may invoke numerous NumPy/SciPy computations that have no value when the user is viewing a different panel.

### 10.6 Reuse of numerical operators

The same discrete Laplacian serves both the reaction–diffusion morphology and environmental heat diffusion. The same Gödel machinery serves inventions and myths. The same Hamiltonian substrate supplies cognition and genomic readout.

### 10.7 Computational complexity

With cognitive dimension fixed at `K=64`, dense eigendecomposition is an `O(K^3)` operation with a fixed small constant relative to population-scale interactions. Agent-to-agent radius searches can be `O(N)` per agent in the straightforward implementation, giving `O(N^2)` worst-case interaction scanning per tick. Spatial field updates scale with grid area and channel count, approximately

```math
O(HW\,C),
```

where `C` is the number of active field channels.

The architecture intentionally trades asymptotic elegance for bounded, inspectable state and practical low-memory execution.

---

# 11. Scientific Foundations and Adopted Formalisms

The project adopts established mathematical mechanisms where they are useful as computational components.

| Component | Adopted formalism | Primary reference |
|---|---|---|
| Quantum-formalism cognition | Hilbert-space state evolution and Born probability | Quantum probability / mathematical cognition literature |
| Integrated-information proxy | IIT-inspired integration statistic | Tononi (2004) |
| Reaction–diffusion morphology | Gray–Scott system | Gray & Scott (1983) |
| Morphogen interpretation | Positional-information / French Flag model | Wolpert (1969) |
| Reaction rates | Arrhenius equation | Arrhenius (1889) |
| Cultural retention | Swadesh-style glottochronology | Swadesh (1952) |
| Predator–prey interpretation | Lotka–Volterra dynamics | Lotka (1925); Volterra (1926) |
| Phase synchronization | Kuramoto oscillators | Kuramoto (1975) |
| Associative memory | Hopfield-style attractor memory | Hopfield (1982) |
| Formal encoding | Gödel numbering | Gödel (1931) |
| Neutral drift | Neutral theory as conceptual basis | Kimura (1968) |
| Island biogeography | Species–area / founder-effect motivation | MacArthur & Wilson (1967) |
| Red Queen dynamics | Host–pathogen coevolution motivation | Van Valen (1973) |
| Sexual selection | Fisherian and handicap-principle motivation | Fisher (1930); Zahavi (1975); Lande (1981) |
| Autocatalytic systems | RAF-set motivation | Kauffman (1993); Hordijk & Steel (2004) |
| Open-ended search | Novelty search / coevolution literature | Lehman & Stanley (2011); Brant & Stanley (2017); Wang et al. (2019) |
| Growth grammars | L-systems | Lindenmayer (1968) |
| Spectral statistics | Random-matrix level-spacing references | Standard RMT literature |
| Survival analysis | Kaplan–Meier methodology | Kaplan & Meier (1958) |
| Non-parametric rank tests | Mann–Whitney / KS procedures | Mann & Whitney (1947); Kolmogorov–Smirnov literature |
| Entropy measures | Shannon / Rényi / Tsallis | Shannon (1948); Rényi (1961); Tsallis (1988) |

These references are used to identify the mathematical lineage of mechanisms. Their presence does not imply that GeNeSIS V is equivalent to the corresponding natural or physical process.

---

# 12. Project Lineage

GeNeSIS V is the fifth iteration of an evolving architecture.

| Version | Main architectural result |
|---|---|
| **GeNeSIS I** | Causal-emergence and self-modeling experiments |
| **GeNeSIS II** | Evolutionary self-organization and inherited latent structure |
| **GeNeSIS III** | World-model-based imaginative planning |
| **GeNeSIS IV** | 64D Hilbert-space cognitive architecture, Born-rule choice and Φ-inspired diagnostics |
| **GeNeSIS V** | Coupling cognition, derived genotype, ecology, chemistry, evolution, civilization and quantitative inference |

The principal inheritance from GeNeSIS IV is the 64-dimensional cognitive substrate. GeNeSIS V's central construction is that this representation is also the source of the genomic encoding.

---

# 13. Installation

## Requirements

```text
Python 3.11+
numpy>=1.26
scipy>=1.12
streamlit>=1.35
plotly>=5.20
pandas>=2.0
scikit-learn>=1.4
networkx>=3.2
```

PyTorch is deliberately absent from the runtime dependency set. The PhysicsOracle role that an earlier specification associated with a small PyTorch network is implemented here as a fixed NumPy MLP. This avoids importing a heavyweight framework for a network that is never trained.

## Install

```bash
git clone <this-repository>
cd GeNeSIS-V
python -m pip install -r requirements.txt
```

## Run the application

```bash
streamlit run GeNeSIS_V.py
```

## Run the quantitative library self-test

```bash
python analytics.py
```

The project modules also expose executable self-tests where appropriate. The cross-module integration suite can be run through the repository's test entry point.

---

# 14. Suggested Experimental Protocol

A reproducible experiment should specify at minimum:

```text
seed
initial population
number of ticks
world dimensions
population floor / ceiling
analysis panel and metric family
whether the run is descriptive or inferential
```

For comparative studies, repeat each condition across multiple seeds and report distributions of summary statistics rather than a single trajectory.

For example, an age-distribution study can define the two cohorts by the exact current implementation:

```math
Y=\operatorname{bottom}_{\lfloor N/2\rfloor}\{\text{agents sorted by age}\},
```

```math
O=\operatorname{top}_{\lfloor N/2\rfloor}\{\text{agents sorted by age}\}.
```

Then compare energy using both the distribution-free KS statistic and Welch's unequal-variance t statistic. This is preferable to deciding a priori that one age cohort “must” differ from the other.

For causal analysis, report the lag window, autoregressive order, sample length, test statistic and p-value together. For chaos analysis, report the diagnostic and the reference system used for calibration.

---

# 15. Known Limitations

The following observations are part of the scientific record of the current architecture rather than issues hidden behind the interface.

## 15.1 Tribal diversification remains structurally limited

Current independently seeded agents can exhibit very high spectral resonance because the underlying Hamiltonian construction shares a common ordered prime structure. In prior testing, resonance has remained in the approximate `0.986–0.9996` range, well above the tribe-join threshold of `0.08`.

The consequence is that the system can remain effectively single-tribe under normal conditions, suppressing the natural triggering rate of alliances, wars and schisms. This is not being disguised as successful emergent pluralism; it is a known architectural behavior.

A future redesign could deliberately introduce stronger per-agent spectral permutation while preserving the shared K=64 substrate, but that would change the model and should therefore be evaluated as a new experiment rather than silently applied.

## 15.2 Long-horizon emergence is not established

Current verification focuses on module-level tests and interactive trajectories in the hundreds-of-ticks regime. Multi-thousand-tick civilization-scale behavior has not been established as a scientific result.

Correctness of local mechanisms is therefore not equivalent to evidence of long-run open-endedness.

## 15.3 Chemistry is intentionally separated from the live ecological loop

The chemistry module contains a richer molecular-species and reaction-kinetics framework than is required for the live frontend. The dashboard currently uses lightweight chemistry projections to remain responsive, while the full reaction machinery is available for analysis and validation.

## 15.4 Open-endedness claims are deliberately narrow

GeNeSIS V uses novelty scoring and novelty-shaped behavioral search. It does not implement the full environment-agent coevolution mechanisms of POET or minimal-criterion coevolution. Accordingly, the project should be described as **open-endedness-oriented**, not as a proof of open-ended evolution.

## 15.5 Physics and thermodynamics are computational analogues

The frozen PhysicsOracle is intentionally synthetic. The thermodynamic panel treats simulation variables as analogues of physical quantities. A correct fit to Maxwell–Boltzmann statistics, for example, would show consistency of a statistical model with the observed synthetic distribution; it would not turn the agent population into a physical gas.

## 15.6 Consciousness is not established

The cognitive system implements Hilbert-space dynamics, Born-rule policy selection and an integration proxy. These provide a rigorous computational object for experiments, but they do not establish sentience or phenomenal experience.

---

# 16. Current Research Directions

The next scientifically meaningful extensions are:

- controlled spectral diversity experiments to test whether pluralistic tribes emerge when eigenbasis degeneracy is deliberately broken;
- multi-seed, multi-thousand-tick experiments with preregistered summary statistics;
- causal ablations of novelty shaping, cultural assimilation, communication and reproduction;
- a full live connection between molecular reaction kinetics and agent resource metabolism;
- agent-triggered L-system structure growth in the world;
- persistent save/load for exact experiment continuation;
- stronger statistical protocols including effect-size distributions across independent seeds rather than single-run significance;
- explicit environment–agent coevolution mechanisms for a more defensible open-endedness study.

A particularly important experimental principle is **ablation**: every major mechanism should be removable while preserving the rest of the simulator, allowing the investigator to ask whether an observed effect actually depends on the mechanism claimed to cause it.

---

# 17. Module Reference

### `metacognition.py`

Gödel encoding, `CivilizationMemory`, novelty scoring, meta-cognitive state dynamics and phylogenetic clustering.

### `consciousness.py`

The 64-dimensional complex cognitive state, Hermitian Hamiltonian, spectral evolution, observation-conditioned action selection, entropy/cost machinery, Φ proxy, self-reference diagnostics and invention substrate.

### `biology.py`

Hamiltonian-derived codon readout, standard genetic-code translation, Gray–Scott reaction–diffusion, discrete Laplacian support and developmental/French-Flag style fields.

### `world.py`

Procedural terrain, resource ecology, seasonal heat dynamics, pheromone and meme fields, structures, cooperative mega-resources and frozen PhysicsOracle.

### `agents.py`

The twenty-action embodied lifecycle, metabolic cost, action bias, interaction radius, reproduction, death, learning, communication, Kuramoto synchronization, Game of Life and spectral wisdom transfer.

### `evolution.py`

Population regulation, meta-fitness, cultural-ratchet verification, behavioral archetypes, clade structure and evolutionary history.

### `chemistry.py`

Atomic constants, molecular representation, reaction kinetics, conservation checks, thermodynamic quantities and reference chemistry.

### `civilization.py`

Tribe formation, spectral identity, eight-term tribal power, diplomacy, assimilation, epistemic schism, technology graph and singularity-override logic.

### `geometry.py`

L-systems, turtle rendering, connected-component/hole topology and morphological descriptors.

### `narrative.py`

Myth generation, motif transmission, Swadesh-style retention and fidelity statistics.

### `nobel.py`

Computable breakthrough detection across Physics, Chemistry, Physiology or Medicine, Literature, Peace and Economics categories.

### `analytics.py`

The quantitative analysis layer: diversity, inequality, spatial statistics, time-series complexity, distributions, survival, information theory, spectral measures, networks, ecology, statistical inference, signal processing, chaos, statistical mechanics and morphology.

### `GeNeSIS_V.py`

The Streamlit research interface, simulation loop, state history, lazy panel dispatcher and visualization layer. The current revision exposes fourteen analytical panels, including the dedicated Statistical Inference and Thermodynamics & Chaos panels.

---

# 18. Citation / Reference List

1. Arrhenius, S. (1889). Über die Reaktionsgeschwindigkeit bei der Inversion von Rohrzucker durch Säuren. *Zeitschrift für Physikalische Chemie*.
2. Brant, J. C., & Stanley, K. O. (2017). Minimal criterion coevolution: A new approach to open-ended search. *Proceedings of GECCO*.
3. Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). Power-law distributions in empirical data. *SIAM Review*.
4. Fisher, R. A. (1930). *The Genetical Theory of Natural Selection*. Clarendon Press.
5. Gödel, K. (1931). Über formal unentscheidbare Sätze der Principia Mathematica und verwandter Systeme I. *Monatshefte für Mathematik und Physik*.
6. Gray, P., & Scott, S. K. (1983). Autocatalytic reactions in the isothermal, continuous stirred tank reactor. *Chemical Engineering Science*.
7. Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*.
8. Grassberger, P., & Procaccia, I. (1983). Measuring the strangeness of strange attractors. *Physica D*.
9. Hordijk, W., & Steel, M. (2004). Detecting autocatalytic, self-sustaining sets in chemical reaction systems. *Journal of Theoretical Biology*.
10. Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *Proceedings of the National Academy of Sciences*.
11. Jaccard, P. (1912). The distribution of the flora in the alpine zone. *New Phytologist*.
12. Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *Journal of the American Statistical Association*.
13. Kimura, M. (1968). Evolutionary rate at the molecular level. *Nature*.
14. Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators. In *International Symposium on Mathematical Problems in Theoretical Physics*.
15. Lande, R. (1981). Models of speciation by sexual selection on polygenic traits. *Proceedings of the National Academy of Sciences*.
16. Lehman, J., & Stanley, K. O. (2011). Abandoning objectives: Evolution through the search for novelty alone. *Evolutionary Computation*.
17. Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences. *IEEE Transactions on Information Theory*.
18. Lindenmayer, A. (1968). Mathematical models for cellular interaction in development. *Journal of Theoretical Biology*.
19. Lotka, A. J. (1925). *Elements of Physical Biology*. Williams & Wilkins.
20. MacArthur, R. H., & Wilson, E. O. (1967). *The Theory of Island Biogeography*. Princeton University Press.
21. Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*.
22. Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. *Biometrika*.
23. Ojala, T., Pietikäinen, M., & Mäenpää, T. (2002). Multiresolution gray-scale and rotation invariant texture classification with local binary patterns. *IEEE Transactions on Pattern Analysis and Machine Intelligence*.
24. Rényi, A. (1961). On measures of entropy and information. In *Proceedings of the Fourth Berkeley Symposium on Mathematical Statistics and Probability*.
25. Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology*.
26. Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. *Physica D*.
27. Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*.
28. Sørensen, T. (1948). A method of establishing groups of equal amplitude in plant sociology based on similarity of species content. *Biologiske Skrifter*.
29. Swadesh, M. (1952). Lexico-statistic dating of prehistoric ethnic contacts. *Proceedings of the American Philosophical Society*.
30. Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*.
31. Tsallis, C. (1988). Possible generalization of Boltzmann–Gibbs statistics. *Journal of Statistical Physics*.
32. Turing, A. M. (1952). The chemical basis of morphogenesis. *Philosophical Transactions of the Royal Society B*.
33. Van Valen, L. (1973). A new evolutionary law. *Evolutionary Theory*.
34. Volterra, V. (1926). Variazioni e fluttuazioni del numero d'individui in specie animali conviventi. *Memorie della Reale Accademia Nazionale dei Lincei*.
35. Wang, R., Lehman, J., Clune, J., & Stanley, K. O. (2019). Paired open-ended trailblazer (POET): Endlessly generating increasingly complex and diverse learning environments and their solutions. *arXiv preprint*.
36. Welch, B. L. (1947). The generalization of “Student's” problem when several different population variances are involved. *Biometrika*.
37. Wolpert, L. (1969). Positional information and the spatial pattern of cellular differentiation. *Journal of Theoretical Biology*.
38. Zahavi, A. (1975). Mate selection—a selection for a handicap. *Journal of Theoretical Biology*.

---

# 19. License

Copyright 2026 Devanik — GeNeSIS V (OMNIGENESIS).

Licensed under the **Apache License, Version 2.0**.

See `LICENSE` for the complete license text.

---

# 20. Research Statement

GeNeSIS V should be read as a **computational laboratory for constructing, perturbing and measuring coupled artificial-life dynamics**.

Its strongest contribution is architectural: cognitive dynamics, genomic representation, local ecology, culture and civilization are placed on mutually observable mathematical state variables instead of being isolated demonstrations. Its second contribution is methodological: the project exposes a large collection of quantitative diagnostics so that emergent behavior can be inspected with effect sizes, confidence intervals, distribution tests, spectral statistics, information measures, causality diagnostics and statistical-mechanics proxies rather than only with visual narratives.

The project deliberately leaves the strongest questions open. Whether the coupled system produces robust multi-tribal civilizations, genuinely open-ended evolutionary innovation, long-lived cultural attractors, or qualitatively new collective regimes is an empirical question for controlled experiments—not a conclusion encoded in the README.

That distinction is the basis on which GeNeSIS V is intended to be extended: **construct the mechanism, expose the observable, test the invariant, quantify the effect, and report the limitation.**

<div align="center">

*GeNeSIS V — OMNIGENESIS*

*One substrate. Many scales. Explicit mathematics. Testable dynamics.*

</div>
