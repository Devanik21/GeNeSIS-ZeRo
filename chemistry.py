"""
chemistry.py — GeNeSIS V — Elemental Composition & Reaction Kinetics
========================================================================

Masterplan §3.2. Unlike consciousness.py/metacognition.py, this module
has no IV precedent to be faithful to — chemistry is V's own addition.
What it must be faithful to instead is real chemistry: real atomic
weights, real electronegativities, real mass-balanced reactions, and
the real Arrhenius equation. Where the sim necessarily simplifies (a
single-step "respiration" reaction standing in for a 10-step metabolic
pathway), that simplification is stated plainly rather than dressed up
as lab-accurate kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

EPS = 1e-12
GAS_CONSTANT_R: float = 8.314  # J / (mol . K), real physical constant

# ----------------------------------------------------------------------------
# The periodic subset  (real atomic data: IUPAC standard atomic weights;
# Pauling-scale electronegativities)
# ----------------------------------------------------------------------------
ELEMENTS: List[str] = ["H", "C", "N", "O", "Na", "Mg", "P", "S", "Cl", "K", "Ca", "Fe", "Zn", "Cu", "I"]
N_ELEMENTS: int = len(ELEMENTS)
ELEMENT_INDEX: Dict[str, int] = {e: i for i, e in enumerate(ELEMENTS)}

ATOMIC_WEIGHT: Dict[str, float] = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "Na": 22.990,
    "Mg": 24.305, "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098,
    "Ca": 40.078, "Fe": 55.845, "Zn": 65.38, "Cu": 63.546, "I": 126.904,
}
ELECTRONEGATIVITY: Dict[str, float] = {  # Pauling scale
    "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "Na": 0.93,
    "Mg": 1.31, "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82,
    "Ca": 1.00, "Fe": 1.83, "Zn": 1.65, "Cu": 1.90, "I": 2.66,
}
ATOMIC_WEIGHT_VEC = np.array([ATOMIC_WEIGHT[e] for e in ELEMENTS])


# ----------------------------------------------------------------------------
# Molecule: a composition vector over ELEMENTS
# ----------------------------------------------------------------------------
@dataclass
class Molecule:
    name: str
    composition: Dict[str, int]  # element -> atom count

    def __hash__(self) -> int:
        # Molecules are used as dict keys in Reaction stoichiometry; identity
        # here is by name (each named molecule constant is meant to be a
        # singleton), since a plain dict field isn't hashable on its own.
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Molecule) and self.name == other.name

    def vector(self) -> np.ndarray:
        v = np.zeros(N_ELEMENTS, dtype=np.float64)
        for el, n in self.composition.items():
            v[ELEMENT_INDEX[el]] = n
        return v

    def molar_mass(self) -> float:
        return float(sum(ATOMIC_WEIGHT[el] * n for el, n in self.composition.items()))

    def hill_formula(self) -> str:
        """Hill notation: C first, H second, then remaining elements alphabetically —
        the standard chemistry convention for writing molecular formulas."""
        parts = []
        comp = self.composition
        if "C" in comp:
            parts.append(f"C{comp['C']}" if comp["C"] != 1 else "C")
        if "H" in comp:
            parts.append(f"H{comp['H']}" if comp["H"] != 1 else "H")
        for el in sorted(k for k in comp if k not in ("C", "H")):
            n = comp[el]
            parts.append(f"{el}{n}" if n != 1 else el)
        return "".join(parts)


# Real, well-known molecules used by the two built-in reactions below.
GLUCOSE = Molecule("glucose", {"C": 6, "H": 12, "O": 6})
WATER = Molecule("water", {"H": 2, "O": 1})
CO2 = Molecule("carbon dioxide", {"C": 1, "O": 2})
O2 = Molecule("oxygen", {"O": 2})

# Sanity-checkable against any chemistry reference: glucose's real molar mass
# is ~180.16 g/mol; GLUCOSE.molar_mass() is asserted against this in the
# self-test below, using only the ATOMIC_WEIGHT table above.

MOLECULES: List[Molecule] = [GLUCOSE, WATER, CO2, O2]
N_MOLECULES: int = len(MOLECULES)
MOLECULE_INDEX: Dict[str, int] = {m.name: i for i, m in enumerate(MOLECULES)}


# ----------------------------------------------------------------------------
# Reaction: stoichiometrically balanced, with an approximate real Delta-G
# ----------------------------------------------------------------------------
@dataclass
class Reaction:
    name: str
    reactants: Dict[Molecule, int]  # molecule -> stoichiometric coefficient
    products: Dict[Molecule, int]
    delta_g_kj_per_mol: float  # standard free energy change (approximate, illustrative)
    activation_energy_j_per_mol: float  # Ea
    pre_exponential_factor: float  # A, in the same rate units the caller wants back

    def is_mass_balanced(self) -> bool:
        """Real conservation-of-atoms check: total element counts must match
        on both sides. This is the correctness test real chemistry gives us
        for free — a reaction that fails this is simply wrong."""
        lhs = np.zeros(N_ELEMENTS)
        rhs = np.zeros(N_ELEMENTS)
        for mol, coeff in self.reactants.items():
            lhs += coeff * mol.vector()
        for mol, coeff in self.products.items():
            rhs += coeff * mol.vector()
        return bool(np.allclose(lhs, rhs))


PHOTOSYNTHESIS = Reaction(
    name="photosynthesis (simplified, light-energy input not atom-balanced)",
    reactants={CO2: 6, WATER: 6},
    products={GLUCOSE: 1, O2: 6},
    delta_g_kj_per_mol=2870.0,  # endergonic: real ballpark magnitude, opposite sign of respiration
    activation_energy_j_per_mol=60000.0,
    pre_exponential_factor=1.0e6,
)

RESPIRATION = Reaction(
    name="cellular respiration (simplified, single-step proxy for glycolysis + Krebs + ETC)",
    reactants={GLUCOSE: 1, O2: 6},
    products={CO2: 6, WATER: 6},
    delta_g_kj_per_mol=-2870.0,  # real, well-known ballpark for glucose oxidation
    activation_energy_j_per_mol=45000.0,
    pre_exponential_factor=1.0e8,
)


# ----------------------------------------------------------------------------
# Arrhenius kinetics  (real equation: k = A * exp(-Ea / RT))
# ----------------------------------------------------------------------------
def arrhenius_rate_constant(reaction: Reaction, temperature_kelvin: float) -> float:
    T = max(temperature_kelvin, 1.0)  # guard against non-physical T <= 0
    return reaction.pre_exponential_factor * np.exp(
        -reaction.activation_energy_j_per_mol / (GAS_CONSTANT_R * T)
    )


def heat_to_kelvin(heat_value: float) -> float:
    """
    Bridges world.py's internal heat_field units (roughly [0, 2], no
    physical scale of its own) to a real thermodynamic temperature.

    # design choice: world.py's heat field has no inherent unit — this
    # affine map (250 K to 450 K over the field's [0, 2] range) puts
    # biologically-plausible temperatures at typical field values without
    # claiming the underlying field was ever meant to be literal Kelvin.
    """
    return 250.0 + 100.0 * float(heat_value)


# ----------------------------------------------------------------------------
# Resource <-> element bridge (world.py's 4 abstract resource types don't
# carry chemistry on their own; this profile table is how they get some)
# ----------------------------------------------------------------------------
# design choice: fractional element profile per world.py resource type,
# each summing to 1.0. "energy" is treated as pre-formed ATP-equivalent and
# is deliberately left with a thin, mostly-CHO profile since it represents
# already-processed chemical energy rather than raw biomass.
RESOURCE_ELEMENT_PROFILE: Dict[str, Dict[str, float]] = {
    "water": {"H": 2 / 3, "O": 1 / 3},
    "food": {"C": 0.45, "H": 0.35, "O": 0.15, "N": 0.05},
    "minerals": {
        "Ca": 0.20, "Fe": 0.15, "Mg": 0.15, "K": 0.15, "Na": 0.10,
        "P": 0.10, "S": 0.10, "Zn": 0.03, "Cu": 0.01, "I": 0.01,
    },
    "energy": {"C": 0.5, "H": 0.4, "O": 0.1},
}


def resource_to_element_vector(resource_amounts: np.ndarray, resource_names: List[str]) -> np.ndarray:
    """Projects world.py's 4-channel resource vector into a 15-element vector."""
    out = np.zeros(N_ELEMENTS, dtype=np.float64)
    for amount, name in zip(resource_amounts, resource_names):
        profile = RESOURCE_ELEMENT_PROFILE.get(name)
        if profile is None:
            continue
        for el, frac in profile.items():
            out[ELEMENT_INDEX[el]] += amount * frac
    return out


# ----------------------------------------------------------------------------
# ChemistryField: per-cell element composition grid over the world
# ----------------------------------------------------------------------------
class ChemistryField:
    def __init__(self, height: int, width: int, resource_grid: np.ndarray, resource_names: List[str], seed: int = 0):
        self.height = height
        self.width = width

        # Bulk elemental composition — raw material stock, bridged from
        # world.py's abstract resources. Reactions never change this grid
        # (atoms are conserved); it exists for pH and toxicity, which are
        # properties of *what a cell is made of*, not what it's currently
        # metabolising.
        self.grid = np.zeros((height, width, N_ELEMENTS), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                self.grid[y, x, :] = resource_to_element_vector(resource_grid[y, x, :], resource_names)

        h_fracs = self.grid[:, :, ELEMENT_INDEX["H"]] / (self.grid.sum(axis=2) + EPS)
        # design choice: pH is self-calibrated to this world's *own* mean and
        # spread of hydrogen fraction rather than an absolute "pure water"
        # reference — a fixed absolute baseline saturated at pH 14 for every
        # realistic mixed-resource cell the first time this was tested, since
        # no mixed cell ever resembles pure water's H fraction. Centering on
        # the grid's own distribution is what actually produces a graded,
        # useful signal instead of a constant that pins to one extreme.
        self._ph_baseline = float(h_fracs.mean())
        self._ph_scale = 3.5 / max(float(h_fracs.std()), EPS)

        # Molecular species pools — what reactions actually act on. Distinct
        # from the element grid above: two cells can have identical bulk
        # elemental composition while holding that carbon/hydrogen/oxygen in
        # completely different molecules (all-glucose vs. all-CO2+water),
        # and it is exactly that distinction reaction kinetics needs to see.
        rng = np.random.default_rng(seed)
        self.molecule_grid = np.zeros((height, width, N_MOLECULES), dtype=np.float32)
        food = resource_grid[:, :, resource_names.index("food")] if "food" in resource_names else np.zeros((height, width))
        water_res = resource_grid[:, :, resource_names.index("water")] if "water" in resource_names else np.zeros((height, width))
        self.molecule_grid[:, :, MOLECULE_INDEX["glucose"]] = 0.3 * food
        self.molecule_grid[:, :, MOLECULE_INDEX["water"]] = 0.5 * water_res
        self.molecule_grid[:, :, MOLECULE_INDEX["oxygen"]] = 0.5 + 0.1 * rng.uniform(-1, 1, (height, width))  # ambient-air proxy
        self.molecule_grid[:, :, MOLECULE_INDEX["carbon dioxide"]] = 0.05  # trace ambient
        np.clip(self.molecule_grid, 0.0, None, out=self.molecule_grid)

    def local_composition(self, x: int, y: int) -> np.ndarray:
        return self.grid[y, x, :].astype(np.float64)

    def local_molecules(self, x: int, y: int) -> Dict[str, float]:
        return {m.name: float(self.molecule_grid[y, x, i]) for i, m in enumerate(MOLECULES)}

    def total_element_mass(self, x: int, y: int) -> np.ndarray:
        """Reconstructs elemental totals implied by the current molecule
        pools — used to verify conservation of atoms across a reaction step,
        independent of the (separate, static) bulk element grid above."""
        out = np.zeros(N_ELEMENTS, dtype=np.float64)
        for i, mol in enumerate(MOLECULES):
            out += self.molecule_grid[y, x, i] * mol.vector()
        return out

    def local_ph(self, x: int, y: int) -> float:
        """
        # design choice: a real pH is -log10[H+]; there is no literal proton
        # concentration in this model, so local hydrogen mass fraction is
        # used as a monotone proxy, self-calibrated (see __init__) to this
        # world's own composition distribution. This is a flavour model, not
        # a rigorous acid-base equilibrium calculation.
        """
        comp = self.local_composition(x, y)
        total = comp.sum()
        if total < EPS:
            return 7.0
        h_fraction = comp[ELEMENT_INDEX["H"]] / total
        return float(np.clip(7.0 - self._ph_scale * (h_fraction - self._ph_baseline), 0.0, 14.0))

    def toxicity(self, x: int, y: int, tolerance: np.ndarray) -> float:
        """
        tolerance: a length-N_ELEMENTS vector of an agent's preferred local
        composition (e.g. a running average of compositions it has safely
        eaten). Toxicity is the normalised deviation from that tolerance —
        an element ratio outside an agent's tolerance band imposes a real
        metabolic penalty (masterplan §3.2), computed here as a plain
        Euclidean deviation, normalised to a comparable scale.
        """
        comp = self.local_composition(x, y)
        c_sum, t_sum = comp.sum(), tolerance.sum()
        if c_sum < EPS or t_sum < EPS:
            return 0.0
        c_norm, t_norm = comp / c_sum, tolerance / t_sum
        return float(np.linalg.norm(c_norm - t_norm))

    def apply_reaction_step(self, reaction: Reaction, temperature_field: np.ndarray, dt: float = 1.0) -> None:
        """
        One explicit step of single-step mass-action kinetics, operating on
        MOLECULE pools (not raw elements — see the ChemistryField docstring
        above for why that distinction is the one that makes this function
        capable of doing anything at all). Reactant molecules are consumed
        and product molecules are produced at a rate set by the real
        Arrhenius equation and the local temperature, limited by whichever
        reactant is scarcest in that cell (the limiting reagent).
        """
        reactant_idx = {MOLECULE_INDEX[m.name]: coeff for m, coeff in reaction.reactants.items()}
        product_idx = {MOLECULE_INDEX[m.name]: coeff for m, coeff in reaction.products.items()}

        r_indices = np.array(list(reactant_idx.keys()))
        r_coeffs = np.array(list(reactant_idx.values()), dtype=np.float64)
        p_indices = np.array(list(product_idx.keys()))
        p_coeffs = np.array(list(product_idx.values()), dtype=np.float64)

        for y in range(self.height):
            for x in range(self.width):
                available = self.molecule_grid[y, x, r_indices]
                extent = float(np.min(available / (r_coeffs + EPS)))
                if extent <= EPS:
                    continue
                k = arrhenius_rate_constant(reaction, heat_to_kelvin(temperature_field[y, x]))
                rate = min(extent, k * dt) if k * dt < extent else extent
                self.molecule_grid[y, x, r_indices] -= rate * r_coeffs
                self.molecule_grid[y, x, p_indices] += rate * p_coeffs
        np.clip(self.molecule_grid, 0.0, None, out=self.molecule_grid)


# ----------------------------------------------------------------------------
# Molecule discovery log  (feeds the future nobel.py Chemistry category)
# ----------------------------------------------------------------------------
class MoleculeDiscoveryLog:
    def __init__(self):
        self.known_formulas: set = set()

    def check_and_register(self, molecule: Molecule) -> Tuple[bool, str]:
        formula = molecule.hill_formula()
        is_novel = formula not in self.known_formulas
        self.known_formulas.add(formula)
        return is_novel, formula


if __name__ == "__main__":
    print("Glucose molar mass:", round(GLUCOSE.molar_mass(), 3), "g/mol (real value: ~180.156)")
    assert abs(GLUCOSE.molar_mass() - 180.156) < 0.05
    print("Glucose Hill formula:", GLUCOSE.hill_formula())
    assert GLUCOSE.hill_formula() == "C6H12O6"

    assert PHOTOSYNTHESIS.is_mass_balanced(), "photosynthesis must conserve atoms"
    assert RESPIRATION.is_mass_balanced(), "respiration must conserve atoms"
    print("Photosynthesis and respiration are both mass-balanced (real conservation of atoms).")

    k_cold = arrhenius_rate_constant(RESPIRATION, heat_to_kelvin(0.0))
    k_hot = arrhenius_rate_constant(RESPIRATION, heat_to_kelvin(2.0))
    assert k_hot > k_cold, "Arrhenius rate must increase with temperature"
    print(f"Arrhenius sanity: k(cold)={k_cold:.6g}  k(hot)={k_hot:.6g}  (hot > cold: {k_hot > k_cold})")

    resource_names = ["water", "food", "minerals", "energy"]
    resource_grid = np.random.default_rng(0).uniform(0.2, 1.0, size=(20, 20, 4)).astype(np.float32)
    field = ChemistryField(20, 20, resource_grid, resource_names)
    assert np.isfinite(field.grid).all()
    print("ChemistryField initial grid sum:", float(field.grid.sum()))

    temp_field = np.random.default_rng(1).uniform(0.0, 2.0, size=(20, 20))
    before = field.total_element_mass(5, 5)
    before_glucose = field.local_molecules(5, 5)["glucose"]
    for _ in range(20):
        field.apply_reaction_step(RESPIRATION, temp_field)
    after = field.total_element_mass(5, 5)
    after_glucose = field.local_molecules(5, 5)["glucose"]

    assert np.isfinite(field.molecule_grid).all()
    assert (field.molecule_grid >= 0).all()
    assert after_glucose < before_glucose, (
        "respiration must actually consume glucose — this is exactly the check "
        "that would have caught the original element-space bug immediately"
    )
    assert np.allclose(before, after, atol=1e-4), (
        f"atoms must be conserved through a reaction: before={before} after={after}"
    )
    print(f"Respiration ran for real: glucose {before_glucose:.4f} -> {after_glucose:.4f} "
          f"(consumed), while total element mass stayed conserved (before==after within 1e-4).")

    ph = field.local_ph(5, 5)
    print("local pH at (5,5):", round(ph, 2))
    assert 0.0 <= ph <= 14.0

    # confirm pH is graded, not saturated, across the grid (the original bug's symptom)
    ph_samples = [field.local_ph(x, y) for x in range(0, 20, 4) for y in range(0, 20, 4)]
    assert len(set(round(p, 1) for p in ph_samples)) > 1, "pH should vary across cells, not saturate to one value"
    print(f"pH sampled across grid varies (not saturated): min={min(ph_samples):.2f} max={max(ph_samples):.2f}")

    log = MoleculeDiscoveryLog()
    novel1, f1 = log.check_and_register(GLUCOSE)
    novel2, f2 = log.check_and_register(GLUCOSE)
    assert novel1 is True and novel2 is False and f1 == f2
    print("MoleculeDiscoveryLog: first sighting novel, repeat not novel. OK.")

    print("chemistry.py self-test passed.")
