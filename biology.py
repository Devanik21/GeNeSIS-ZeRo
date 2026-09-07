"""
biology.py — GeNeSIS V — Codon Reader & Morphogenesis Engine
================================================================

The V-specific unification (masterplan §1): the same K=64 eigenbasis
that drives an agent's cognition in consciousness.py is *read*, here,
as a 64-symbol DNA string — 21 full codons plus one spare base, since
64 = 4**3 is exactly the dimension of a 3-position, 4-letter codon
space. No new per-agent state is allocated: this module only
*interprets* the Hamiltonian eigenvectors consciousness.py already
computes and caches.

Also provides the Gray-Scott reaction-diffusion engine used both for:
    - the Cultural Replicator "stigmergy" morphogen maps (masterplan §5.1)
    - developmental body-plan patterning via the French Flag model (§13)
so that one PDE solver serves both roles, per the memory-budget design
law in the master plan (§2: reuse over duplication).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

BASES: List[str] = ["A", "C", "G", "T"]

# The real, standard genetic code (64 codons -> 20 amino acids + 3 stops).
# This is a universally known, public-domain scientific fact table (the
# same table appears in every molecular biology textbook), not a
# copyrighted expressive work, so it is reproduced here in full.
STANDARD_GENETIC_CODE: Dict[str, str] = {
    "TTT": "Phe", "TTC": "Phe", "TTA": "Leu", "TTG": "Leu",
    "CTT": "Leu", "CTC": "Leu", "CTA": "Leu", "CTG": "Leu",
    "ATT": "Ile", "ATC": "Ile", "ATA": "Ile", "ATG": "Met",
    "GTT": "Val", "GTC": "Val", "GTA": "Val", "GTG": "Val",
    "TCT": "Ser", "TCC": "Ser", "TCA": "Ser", "TCG": "Ser",
    "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
    "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
    "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
    "TAT": "Tyr", "TAC": "Tyr", "TAA": "Stop", "TAG": "Stop",
    "CAT": "His", "CAC": "His", "CAA": "Gln", "CAG": "Gln",
    "AAT": "Asn", "AAC": "Asn", "AAA": "Lys", "AAG": "Lys",
    "GAT": "Asp", "GAC": "Asp", "GAA": "Glu", "GAG": "Glu",
    "TGT": "Cys", "TGC": "Cys", "TGA": "Stop", "TGG": "Trp",
    "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
    "AGT": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
    "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
}
assert len(STANDARD_GENETIC_CODE) == 64


# ---------------------------------------------------------------------------
# Codon reader — the central V unification (masterplan §1)
# ---------------------------------------------------------------------------
def _dominant_component_phase(eigenvector: np.ndarray) -> float:
    """Phase of an eigenvector's largest-magnitude component."""
    k = int(np.argmax(np.abs(eigenvector)))
    return float(np.angle(eigenvector[k]))


def read_codon_string(eigenvectors: np.ndarray) -> str:
    """
    Read a 64-character DNA string directly off an agent's Hamiltonian
    eigenbasis (masterplan §1: K=64 == 4**3, the codon-space dimension).

    Parameters
    ----------
    eigenvectors : (64, 64) complex ndarray
        Column k is the k-th eigenvector of the agent's Hamiltonian H,
        exactly as returned by HarmonicResonanceConsciousness.eigenvectors().

    Returns
    -------
    A 64-character string over {A, C, G, T}: base k is derived from the
    phase quadrant of eigenmode k's dominant component:
        [-pi, -pi/2) -> A    [-pi/2, 0) -> C    [0, pi/2) -> G    [pi/2, pi] -> T

    # design choice: the spec fixes the K=64 <-> 4**3 dimensional
    # coincidence but does not itself define a phase-to-base mapping —
    # this quadrant convention is the simplest deterministic, reproducible
    # choice, and it is the only free parameter in this function.
    """
    n = eigenvectors.shape[1]
    bases = []
    for k in range(n):
        phase = _dominant_component_phase(eigenvectors[:, k])
        if phase < -np.pi / 2:
            bases.append("A")
        elif phase < 0:
            bases.append("C")
        elif phase < np.pi / 2:
            bases.append("G")
        else:
            bases.append("T")
    return "".join(bases)


def codon_split(dna: str) -> List[str]:
    """Split a DNA string into codons of 3, dropping any trailing partial codon."""
    return [dna[i : i + 3] for i in range(0, len(dna) - 2, 3)]


def translate(dna: str) -> List[str]:
    """Standard-genetic-code translation of a codon string into amino acids."""
    return [STANDARD_GENETIC_CODE[c] for c in codon_split(dna)]


def genome_fingerprint(eigenvectors: np.ndarray) -> Dict[str, object]:
    """Convenience bundle: raw DNA string, its codons, and the translated protein."""
    dna = read_codon_string(eigenvectors)
    codons = codon_split(dna)
    protein = translate(dna)
    return {"dna": dna, "codons": codons, "protein": protein}


def genetic_distance(dna_a: str, dna_b: str) -> float:
    """Fraction of mismatched bases between two equal-length DNA strings —
    the real-biology analogue used for speciation thresholds (masterplan §3.3)."""
    if len(dna_a) != len(dna_b):
        raise ValueError("DNA strings must be equal length to compare")
    mismatches = sum(1 for a, b in zip(dna_a, dna_b) if a != b)
    return mismatches / len(dna_a)


# ---------------------------------------------------------------------------
# Gray-Scott reaction-diffusion  (masterplan §3.3, §13; Turing, 1952)
# ---------------------------------------------------------------------------
def laplacian(Z: np.ndarray) -> np.ndarray:
    """5-point discrete Laplacian on a toroidal (wraparound) grid."""
    return (
        np.roll(Z, 1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, 1, axis=1)
        + np.roll(Z, -1, axis=1)
        - 4.0 * Z
    )


def gray_scott_step(
    A: np.ndarray,
    B: np.ndarray,
    Da: float = 0.16,
    Db: float = 0.08,
    feed: float = 0.035,
    kill: float = 0.065,
    dt: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One explicit-Euler step of the Gray-Scott reaction-diffusion system:

        dA/dt = Da * lap(A) - A*B^2 + feed*(1-A)
        dB/dt = Db * lap(B) + A*B^2 - (feed+kill)*B

    The default (Da, Db, feed, kill) sit in the well-documented "coral
    growth" regime — the classic parameter set known to produce the
    spot/stripe/coral Turing patterns this module exists to generate.
    """
    reaction = A * B * B
    A_new = A + dt * (Da * laplacian(A) - reaction + feed * (1.0 - A))
    B_new = B + dt * (Db * laplacian(B) + reaction - (feed + kill) * B)
    return np.clip(A_new, 0.0, 1.0), np.clip(B_new, 0.0, 1.0)


def init_grid(height: int, width: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """A=1/B=0 background with a few small B seed patches — the standard
    Gray-Scott initial condition that reliably kicks off pattern formation."""
    rng = np.random.default_rng(seed)
    A = np.ones((height, width), dtype=np.float64)
    B = np.zeros((height, width), dtype=np.float64)
    n_seeds = max(1, (height * width) // 400)
    for _ in range(n_seeds):
        cy, cx = int(rng.integers(0, height)), int(rng.integers(0, width))
        r = 3
        y0, y1 = max(0, cy - r), min(height, cy + r)
        x0, x1 = max(0, cx - r), min(width, cx + r)
        B[y0:y1, x0:x1] = 1.0
        A[y0:y1, x0:x1] = 0.5
    return A, B


def biome_rgb(A: np.ndarray, B: np.ndarray, element_hue: np.ndarray) -> np.ndarray:
    """
    Compose the elevated Biome Cartography RGB frame (masterplan §5.1):
    R = morphogen A, G = morphogen B, Blue = dominant local element hue.
    Each channel is independently normalised to [0, 1].
    """

    def norm(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

    return np.stack([norm(A), norm(B), norm(element_hue)], axis=-1)


# ---------------------------------------------------------------------------
# French-Flag body-plan patterning (masterplan §13; Wolpert, 1969)
# ---------------------------------------------------------------------------
def french_flag_zones(
    morphogen: np.ndarray, low_threshold: float = 0.33, high_threshold: float = 0.66
) -> np.ndarray:
    """
    Wolpert's French Flag model: a single morphogen gradient, thresholded
    into three discrete zones (0, 1, 2), the simplest known real mechanism
    for turning a smooth gradient into a discrete body plan.
    """
    m = morphogen
    lo, hi = float(m.min()), float(m.max())
    normed = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
    zones = np.zeros_like(normed, dtype=np.int8)
    zones[normed >= low_threshold] = 1
    zones[normed >= high_threshold] = 2
    return zones


if __name__ == "__main__":
    from consciousness import HarmonicResonanceConsciousness

    hrc_a = HarmonicResonanceConsciousness(agent_id=0, seed=1)
    hrc_b = HarmonicResonanceConsciousness(agent_id=1, seed=2)

    fp_a = genome_fingerprint(hrc_a.eigenvectors())
    fp_b = genome_fingerprint(hrc_b.eigenvectors())
    print("agent 0 DNA:", fp_a["dna"])
    print("agent 0 codons (first 5):", fp_a["codons"][:5])
    print("agent 0 protein (first 5):", fp_a["protein"][:5])
    print("genetic distance(0, 1) =", genetic_distance(fp_a["dna"], fp_b["dna"]))

    A, B = init_grid(48, 48, seed=7)
    for _ in range(400):
        A, B = gray_scott_step(A, B)
    assert np.isfinite(A).all() and np.isfinite(B).all()
    print("Gray-Scott after 400 steps -> A range", A.min(), A.max(), "| B range", B.min(), B.max())

    element_hue = np.random.default_rng(3).uniform(0, 1, size=A.shape)
    rgb = biome_rgb(A, B, element_hue)
    print("biome_rgb shape:", rgb.shape, "range:", rgb.min(), rgb.max())

    zones = french_flag_zones(A)
    print("French Flag zone counts:", {z: int((zones == z).sum()) for z in (0, 1, 2)})
    print("biology.py self-test passed.")
