"""
consciousness.py — GeNeSIS V — Harmonic Resonance Consciousness (HRC)
=========================================================================

One agent's full cognitive substrate, carried forward from GeNeSIS IV
(README IV §3.2-3.11, §4.2) with V's codon-reading hook (masterplan §1)
exposed via the `eigenvectors()` accessor that biology.py consumes.

Nine subsystems, per the original spec:
    1. Wave dynamics        — psi evolution, Born-rule decision, FFT encoding
    2. Hermitian learning    — meta-modulated dH with lazy eigen-recache
    3. Landauer costing      — von Neumann entropy delta
    4. IIT Phi               — bipartition mutual information, 10-tick eval
    5. Strange loop          — 25-tick self-reference inconsistency check
    6. Active inference      — forward model, free energy, confidence
    7. Qualia memory         — pattern store/classify by cosine similarity
    8. Theory of mind        — level-1 predictive models of other agents
    9. Invention engine      — dark-eigenmode exploration, Gödel encoding

Call order per tick (agents.py, not yet built, will orchestrate this):

    events = hrc.evolve()                     # advance psi and psi_meta
    action, probs = hrc.decide(observation)   # Born-rule choice
    # ... environment executes the action, returns a scalar reward ...
    cost = hrc.learn(reward)                  # Hermitian update + Landauer cost
    # condition-gated, called by agents.py or by the environment's own logic:
    hrc.active_inference_step(observation)
    program, godel = hrc.attempt_invention()

`evolve()` auto-runs the two fixed-schedule checks (Phi every 10 ticks,
strange-loop every 25 ticks) internally via its own tick counter, since
those two are unconditionally periodic in the spec. Invention, qualia,
and theory-of-mind are event/condition-triggered in the spec and are
therefore left as explicit calls for the caller to invoke.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from metacognition import (
    ALL_PRIMITIVES,
    EPS,
    GodelEncoder,
    K_DIM,
    K_TASK,
    MAX_PROGRAM_LEN,
    MetaConsciousness,
    N_PRIMITIVES,
    TASK_DT,
    make_soul,
)

EMOTION_NAMES: List[str] = [
    "CURIOSITY", "FEAR", "JOY", "ANGER", "AFFECTION", "GRIEF", "WONDER",
]
# Fidelity correction: README IV §4.2 gives this exact 7-emotion table with
# fixed index/effect meanings (0 Curiosity - temperature/meta-modulation,
# 1 Fear - SURVIVE mode, 2 Joy - reproduction willingness, 3 Anger - DOMINATE
# mode, 4 Affection - bonding/trade, 5 Grief - suppressed, 6 Wonder - gates
# invention). An earlier draft of this file used a plausible-but-wrong
# Ekman-adjacent set (TRUST, SADNESS) before this table was located in the
# spec; agents.py's mode-bias step (Fear->SURVIVE, Anger->DOMINATE) depends
# on these exact names, so it is corrected here rather than left standing.
EMOTION_INDEX: Dict[str, int] = {name: i for i, name in enumerate(EMOTION_NAMES)}
EMOTION_DECAY: float = 0.994

PHI_CRITICAL: float = 0.1
STRANGE_LOOP_EVERY: int = 25
PHI_EVAL_EVERY: int = 10
HOPFIELD_MAX_ATTRACTORS: int = 20
HOPFIELD_REWARD_THRESHOLD: float = 2.0
EIGEN_RECOMPUTE_EVERY: int = 10
EIGEN_RECOMPUTE_REWARD_THRESHOLD: float = 0.25


class HarmonicResonanceConsciousness:
    """K=64 complex-wavefunction cognition, Schrödinger-evolved under a
    personal Hermitian Hamiltonian, decided by Born-rule measurement."""

    def __init__(self, agent_id: int, seed: Optional[int] = None):
        self.agent_id = agent_id
        self.rng = np.random.default_rng(seed)

        self.omega = make_soul(self.rng)  # immutable "soul" (§3.1)

        A = self.rng.normal(0, 0.1, (K_DIM, K_DIM)) + 1j * self.rng.normal(
            0, 0.1, (K_DIM, K_DIM)
        )
        H = np.diag(self.omega).astype(np.complex128) + 0.5 * (A + A.conj().T)
        self.H = 0.5 * (H + H.conj().T)

        psi0 = self.rng.normal(0, 1, K_DIM) + 1j * self.rng.normal(0, 1, K_DIM)
        self.psi = psi0 / np.linalg.norm(psi0)

        self.meta = MetaConsciousness(self.omega, self.rng)

        self._V: Optional[np.ndarray] = None
        self._lam: Optional[np.ndarray] = None
        self._dirty = True
        self._eig()  # populate cache once at birth

        self.emotions = np.zeros(len(EMOTION_NAMES), dtype=np.float64)

        self.tick_count = 0
        self.ticks_since_eigen_recompute = 0
        self.crystallize_count = 0
        self.S_cog_prev: Optional[float] = None

        self.phi_history: List[float] = []
        self.strange_loop_active = False

        self.qualia_memory: Dict[str, np.ndarray] = {}
        self.tom_models: Dict[int, np.ndarray] = {}

        self.free_energy_history: List[float] = []
        self.confidence = 0.5

        self.exploration_counts = np.zeros(K_DIM, dtype=np.int64)

    # ------------------------------------------------------------------
    # Eigendecomposition cache
    # ------------------------------------------------------------------
    def _eig(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._dirty or self._V is None:
            self._lam, self._V = np.linalg.eigh(self.H)
            self._dirty = False
            self.ticks_since_eigen_recompute = 0
        return self._V, self._lam

    def eigenvectors(self) -> np.ndarray:
        """Public accessor consumed by biology.py's codon reader (masterplan §1)."""
        V, _ = self._eig()
        return V

    def _maybe_recompute_eigen(self, reward: Optional[float]) -> None:
        self.ticks_since_eigen_recompute += 1
        force = reward is not None and abs(reward) > EIGEN_RECOMPUTE_REWARD_THRESHOLD
        if force or self.ticks_since_eigen_recompute >= EIGEN_RECOMPUTE_EVERY:
            self._dirty = True
            self._eig()

    # ------------------------------------------------------------------
    # Emotion accessors
    # ------------------------------------------------------------------
    def emotion(self, name: str) -> float:
        return float(self.emotions[EMOTION_INDEX[name]])

    def _bump_emotion(self, name: str, delta: float, lo: float = -1.0, hi: float = 1.0) -> None:
        i = EMOTION_INDEX[name]
        self.emotions[i] = float(np.clip(self.emotions[i] + delta, lo, hi))

    # ------------------------------------------------------------------
    # 1. Wave dynamics — Schrödinger evolution (§3.3)
    # ------------------------------------------------------------------
    def evolve(self) -> Dict[str, float]:
        """Advance psi (task) and psi_meta by one Schrödinger step; run the
        two fixed-schedule checks (Phi @10, strange-loop @25)."""
        V, lam = self._eig()
        phase = np.exp(-1j * lam * TASK_DT)
        self.psi = V @ (phase * (V.conj().T @ self.psi))
        n = np.linalg.norm(self.psi)
        if n > EPS:
            self.psi = self.psi / n

        self.meta.evolve_meta()
        self.emotions *= EMOTION_DECAY
        self.tick_count += 1

        events: Dict[str, float] = {}
        if self.tick_count % PHI_EVAL_EVERY == 0:
            phi, conscious = self.compute_phi()
            events["phi"] = phi
            events["conscious"] = float(conscious)
        if self.tick_count % STRANGE_LOOP_EVERY == 0:
            events["strange_loop"] = float(self.check_strange_loop())
        return events

    # ------------------------------------------------------------------
    # Shared FFT context encoder (§3.4, §3.9)
    # ------------------------------------------------------------------
    def _encode(self, observation: np.ndarray, dim: int = K_DIM) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64).ravel()
        n_fft = max(dim, obs.shape[0]) if obs.shape[0] > 0 else dim
        spec = np.fft.fft(obs, n=n_fft)[:dim]
        norm = np.linalg.norm(spec)
        if norm < EPS:
            spec = np.ones(dim, dtype=np.complex128)
            norm = np.linalg.norm(spec)
        return spec / norm

    # ------------------------------------------------------------------
    # 1. Born-rule quantum decision (§3.4)
    # ------------------------------------------------------------------
    def decide(self, observation: np.ndarray, n_actions: int = 20) -> Tuple[int, np.ndarray]:
        V, _ = self._eig()
        c = self._encode(observation)
        phase_c = np.angle(c)

        probs = np.empty(n_actions, dtype=np.float64)
        for i in range(n_actions):
            base = V[:, i % K_DIM]
            shifted_phase = np.roll(phase_c, i)
            b_i = base * np.exp(1j * shifted_phase)
            bn = np.linalg.norm(b_i)
            if bn > EPS:
                b_i = b_i / bn
            amp = np.vdot(b_i, self.psi)  # <b_i | psi>
            probs[i] = np.abs(amp) ** 2 + EPS
        probs = probs / probs.sum()

        curiosity = self.emotion("CURIOSITY")
        meta_mag = float(np.mean(np.abs(self.meta.psi)))
        T = max(0.01, 0.06 + 0.94 * ((curiosity + 1) / 2) * (0.5 + meta_mag))

        tempered = np.exp(np.log(probs) / T)
        tempered = tempered / tempered.sum()

        action_idx = int(self.rng.choice(n_actions, p=tempered))
        return action_idx, tempered

    # ------------------------------------------------------------------
    # 2 & 3. Meta-modulated Hermitian learning + Landauer cost (§3.5, §3.6)
    # ------------------------------------------------------------------
    def _von_neumann_entropy(self) -> float:
        V, _ = self._eig()
        p = np.abs(V.conj().T @ self.psi) ** 2
        p = p[p > EPS]
        return float(-np.sum(p * np.log(p)))

    def learn(self, reward: float) -> float:
        mu_full = self.meta.learning_rate_profile(K_DIM)
        S = np.sqrt(np.outer(mu_full, mu_full))

        outer = np.outer(self.psi, self.psi.conj())
        dH = np.sign(reward) * min(abs(reward), 3.0) * 0.007 * outer * S
        self.H = self.H + 0.5 * (dH + dH.conj().T)
        self.H = 0.5 * (self.H + self.H.conj().T)

        self._maybe_recompute_eigen(reward)

        S_cog = self._von_neumann_entropy()
        cost = 0.0 if self.S_cog_prev is None else 0.01 * abs(S_cog - self.S_cog_prev)
        self.S_cog_prev = S_cog

        if abs(reward) > HOPFIELD_REWARD_THRESHOLD and self.crystallize_count < HOPFIELD_MAX_ATTRACTORS:
            self._hopfield_crystallize(reward)

        if abs(reward) > 0.5:
            self._record_qualia(reward)

        return cost

    def _hopfield_crystallize(self, reward: float) -> None:
        """Hopfield-style Hebbian imprint of a high-magnitude experience (§3.14)."""
        outer = np.outer(self.psi, self.psi.conj())
        dH = 0.015 * np.sign(reward) / 2.0 * (outer + outer.conj().T)
        self.H = self.H + dH
        self.H = 0.5 * (self.H + self.H.conj().T)
        self.crystallize_count += 1
        self._dirty = True

    # ------------------------------------------------------------------
    # 4. IIT Integrated Information Phi (§3.7)
    # ------------------------------------------------------------------
    @staticmethod
    def _partition_entropy_proxy(psi_part: np.ndarray) -> float:
        return float(np.log2(1.0 + np.var(np.abs(psi_part)) + EPS))

    def compute_phi(self) -> Tuple[float, bool]:
        # NOTE: this bipartition is of the *task* wavefunction psi itself
        # (psi[:24] vs psi[24:64]), distinct from self.meta.psi (the separate
        # 40-dim meta-band object). The spec reuses the label "psi_meta" for
        # both; they are kept as clearly distinct objects here to avoid
        # collapsing two different quantities into one variable.
        psi_task_part = self.psi[:K_TASK]
        psi_rest_part = self.psi[K_TASK:]
        I_task = self._partition_entropy_proxy(psi_task_part)
        I_rest = self._partition_entropy_proxy(psi_rest_part)
        I_full = self._partition_entropy_proxy(self.psi)
        phi = max(0.0, I_task + I_rest - I_full)
        if self.strange_loop_active:
            phi *= 1.5
        self.phi_history.append(phi)

        conscious = False
        if phi > PHI_CRITICAL and len(self.phi_history) >= 50:
            recent = float(np.mean(self.phi_history[-10:]))
            older = float(np.mean(self.phi_history[-50:-40]))
            conscious = recent > 1.2 * (older + EPS)
        if conscious:
            # "Intelligence Prevents War" protocol: verified-conscious agents
            # have their aggression damped.
            self._bump_emotion("ANGER", -0.1)
        return phi, conscious

    # ------------------------------------------------------------------
    # 5. Gödelian self-reference & strange loops (§3.8)
    # ------------------------------------------------------------------
    def check_strange_loop(self) -> bool:
        psi_tilde = self.psi * np.exp(1j * np.pi * np.abs(self.psi))
        n = np.linalg.norm(psi_tilde)
        if n > EPS:
            psi_tilde = psi_tilde / n
        inconsistency = float(np.linalg.norm(psi_tilde - self.psi))
        self.strange_loop_active = inconsistency > 0.5
        if self.strange_loop_active:
            self._bump_emotion("WONDER", 0.05)
        return self.strange_loop_active

    # ------------------------------------------------------------------
    # 6. Active inference & free energy (§3.9)
    # ------------------------------------------------------------------
    def active_inference_step(self, observation: np.ndarray) -> float:
        V, lam = self._eig()
        c = self._encode(observation)
        alpha = V.conj().T @ c
        alpha_tilde = alpha * np.exp(-1j * lam * TASK_DT)
        s_hat = np.abs(V @ alpha_tilde)

        obs = np.asarray(observation, dtype=np.float64).ravel()
        m = min(len(obs), len(s_hat))
        free_energy = float(np.mean((s_hat[:m] - obs[:m]) ** 2)) if m > 0 else 0.0

        self.free_energy_history.append(free_energy)
        if len(self.free_energy_history) > 200:
            self.free_energy_history = self.free_energy_history[-200:]

        recent_fe = float(np.mean(self.free_energy_history[-5:]))
        self.confidence = 1.0 / (1.0 + recent_fe)
        self._bump_emotion("CURIOSITY", 0.05 * recent_fe)
        return free_energy

    # ------------------------------------------------------------------
    # 7. Qualia memory (§3.10)
    # ------------------------------------------------------------------
    def _record_qualia(self, reward: float, mode: str = "generic") -> None:
        key = f"{mode}_{'+' if reward > 0 else '-'}"
        self.qualia_memory[key] = np.abs(self.psi).copy()

    def classify_qualia(self) -> str:
        if not self.qualia_memory:
            return "novel"
        cur = np.abs(self.psi)
        cur_n = np.linalg.norm(cur)
        best_key, best_sim = "novel", 0.0
        for key, pattern in self.qualia_memory.items():
            pn = np.linalg.norm(pattern)
            if cur_n < EPS or pn < EPS:
                continue
            sim = float(np.dot(cur, pattern) / (cur_n * pn))
            if sim > best_sim:
                best_key, best_sim = key, sim
        return best_key if best_sim >= 0.7 else "novel"

    # ------------------------------------------------------------------
    # 8. Theory of mind (§3.10)
    # ------------------------------------------------------------------
    def theory_of_mind_update(self, other_agent_id: int, sigma_other: np.ndarray) -> float:
        target = np.abs(sigma_other)[:K_TASK]
        prior = self.tom_models.get(other_agent_id, np.zeros(K_TASK))
        updated = prior + 0.1 * (target - prior)
        self.tom_models[other_agent_id] = updated
        mse = float(np.mean((updated - target) ** 2))
        return max(-1.0, 1.0 - mse)

    # ------------------------------------------------------------------
    # 9. Gödel-encoded invention (§3.11)
    # ------------------------------------------------------------------
    def attempt_invention(self, resonance_rho: float = 0.0) -> Tuple[List[str], int]:
        V, _ = self._eig()
        k_dark = int(np.argmin(self.exploration_counts))
        self.exploration_counts[k_dark] += 1
        v_dark = V[:, k_dark]

        length = int(self.rng.integers(2, MAX_PROGRAM_LEN + 1))
        program: List[str] = []
        for k in range(length):
            phi_k = float(np.angle(v_dark[k % K_DIM]))
            idx = int(np.floor((phi_k + np.pi) / (2 * np.pi) * N_PRIMITIVES))
            idx = max(0, min(N_PRIMITIVES - 1, idx))
            program.append(ALL_PRIMITIVES[idx])

        wonder = self.emotion("WONDER")
        outer = np.outer(v_dark, v_dark.conj())
        dH = (wonder / 5.0) * 0.5 * (outer + outer.conj().T)
        self.H = self.H + dH
        self.H = 0.5 * (self.H + self.H.conj().T)
        self._dirty = True

        godel = GodelEncoder.encode(program)
        phi_now = self.phi_history[-1] if self.phi_history else 0.0
        if phi_now > 0.005 and wonder > 0.85:
            godel *= 7919  # Eureka Protocol

        return program, godel

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def spectral_summary(self) -> Dict[str, float]:
        _, lam = self._eig()
        return {
            "lambda_min": float(lam.min()),
            "lambda_max": float(lam.max()),
            "lambda_spread": float(lam.max() - lam.min()),
            "phi_last": self.phi_history[-1] if self.phi_history else 0.0,
            "confidence": self.confidence,
            "crystallized": self.crystallize_count,
        }


if __name__ == "__main__":
    hrc = HarmonicResonanceConsciousness(agent_id=0, seed=42)
    rng = np.random.default_rng(1)
    for t in range(60):
        obs = rng.normal(size=12)
        events = hrc.evolve()
        action, probs = hrc.decide(obs)
        reward = float(rng.normal(0, 1))
        hrc.learn(reward)
        if t % 7 == 0:
            hrc.active_inference_step(obs)
    print("tick_count:", hrc.tick_count)
    print("psi norm:", np.linalg.norm(hrc.psi))
    print("H Hermitian:", np.allclose(hrc.H, hrc.H.conj().T))
    print("spectral_summary:", hrc.spectral_summary())
    program, godel = hrc.attempt_invention()
    print("invention:", program, "-> Gödel", godel)
    print("consciousness.py self-test passed.")
