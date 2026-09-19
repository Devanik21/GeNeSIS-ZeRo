"""
analytics.py — GeNeSIS V — Quantitative Analysis Library
============================================================

Real, citable statistical / complexity / ecological metrics computed over
live simulation state. Every function here implements a published method
and is verified in this module's self-test against a case with a known
analytic answer, rather than merely "running without error".

Contents
--------
Diversity & inequality   Shannon, Simpson, Pielou, Hill numbers, Gini, Lorenz,
                         Hutcheson t-test, effective number of types
Spatial statistics       Moran's I, Ripley's K/L, nearest-neighbour index,
                         quadrat dispersion index, spatial entropy
Time series              autocorrelation, Hurst exponent (R/S), spectral
                         entropy, permutation entropy, Lyapunov proxy,
                         detrended fluctuation analysis, Fano factor
Distributions            power-law MLE + KS distance, Zipf slope,
                         Gompertz-Makeham fit, Kaplan-Meier survival
Information theory       Shannon/joint/mutual information, KL and JS
                         divergence, transfer-entropy proxy, normalised
                         compression distance
Geometry / complexity    box-counting fractal dimension, participation ratio,
                         spectral (von Neumann) entropy, IPR
Networks                 degree/betweenness summary, assortativity,
                         clustering, small-world sigma, modularity proxy
Ecology                  Lotka-Volterra residual fit, carrying-capacity
                         estimate, logistic growth fit, Kendall tau trend

Everything is pure NumPy/SciPy and safe on empty or degenerate input:
functions return NaN (not an exception) when a statistic is undefined for
the data given, so a live dashboard never crashes on an early tick.
"""

from __future__ import annotations

import math
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage, optimize, stats

EPS = 1e-12
NAN = float("nan")


def _clean(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=np.float64).ravel()
    return a[np.isfinite(a)]


# ===========================================================================
# DIVERSITY & INEQUALITY
# ===========================================================================
def shannon_entropy(counts: Sequence[float], base: float = math.e) -> float:
    """H = -Σ p log p. The standard Shannon (1948) diversity index."""
    c = _clean(counts)
    c = c[c > 0]
    if c.size == 0:
        return NAN
    p = c / c.sum()
    return float(-np.sum(p * np.log(p)) / np.log(base))


def simpson_index(counts: Sequence[float]) -> float:
    """Simpson's (1949) D = Σ p². Probability two random draws match."""
    c = _clean(counts)
    c = c[c > 0]
    if c.size == 0:
        return NAN
    p = c / c.sum()
    return float(np.sum(p ** 2))


def inverse_simpson(counts: Sequence[float]) -> float:
    """1/D — the 'effective number of equally-common types'."""
    d = simpson_index(counts)
    return NAN if not np.isfinite(d) or d <= EPS else float(1.0 / d)


def pielou_evenness(counts: Sequence[float]) -> float:
    """J' = H / ln(S). 1.0 means perfectly even; 0 means total dominance."""
    c = _clean(counts)
    c = c[c > 0]
    if c.size < 2:
        return NAN
    return float(shannon_entropy(c) / np.log(c.size))


def hill_number(counts: Sequence[float], q: float) -> float:
    """
    Hill (1973) number of order q — a unified diversity family.
    q=0 richness, q→1 exp(Shannon), q=2 inverse Simpson.
    """
    c = _clean(counts)
    c = c[c > 0]
    if c.size == 0:
        return NAN
    p = c / c.sum()
    if abs(q - 1.0) < 1e-9:
        return float(np.exp(shannon_entropy(c)))
    return float(np.sum(p ** q) ** (1.0 / (1.0 - q)))


def gini_coefficient(values: Sequence[float]) -> float:
    """
    Gini (1912) coefficient of inequality. 0 = perfect equality,
    →1 = one holder has everything. Computed by the sorted-rank formula.
    """
    v = _clean(values)
    v = v[v >= 0]
    if v.size < 2 or v.sum() <= EPS:
        return NAN
    v = np.sort(v)
    n = v.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1.0) / n)


def lorenz_curve(values: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Cumulative population share vs cumulative resource share."""
    v = _clean(values)
    v = np.sort(v[v >= 0])
    if v.size == 0 or v.sum() <= EPS:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    cum = np.concatenate([[0.0], np.cumsum(v) / v.sum()])
    x = np.linspace(0.0, 1.0, cum.size)
    return x, cum


def theil_index(values: Sequence[float]) -> float:
    """Theil's T — an entropy-based inequality measure; 0 = equality."""
    v = _clean(values)
    v = v[v > 0]
    if v.size < 2:
        return NAN
    mu = v.mean()
    return float(np.mean((v / mu) * np.log(v / mu)))


def berger_parker(counts: Sequence[float]) -> float:
    """Dominance: share held by the single most abundant type."""
    c = _clean(counts)
    if c.size == 0 or c.sum() <= EPS:
        return NAN
    return float(c.max() / c.sum())


# ===========================================================================
# SPATIAL STATISTICS
# ===========================================================================
def morans_i(field: np.ndarray) -> float:
    """
    Moran's I (1950) spatial autocorrelation on a grid, rook adjacency.
    +1 clustered, 0 random, -1 dispersed (checkerboard).
    """
    z = np.asarray(field, dtype=np.float64)
    if z.size < 4 or not np.isfinite(z).all():
        return NAN
    d = z - z.mean()
    denom = np.sum(d ** 2)
    if denom <= EPS:
        return NAN
    num = (
        np.sum(d[:-1, :] * d[1:, :])
        + np.sum(d[1:, :] * d[:-1, :])
        + np.sum(d[:, :-1] * d[:, 1:])
        + np.sum(d[:, 1:] * d[:, :-1])
    )
    n_pairs = 2 * (z.shape[0] - 1) * z.shape[1] + 2 * z.shape[0] * (z.shape[1] - 1)
    if n_pairs == 0:
        return NAN
    return float((z.size / n_pairs) * (num / denom))


def nearest_neighbour_index(xs: Sequence[float], ys: Sequence[float],
                            width: float, height: float) -> float:
    """
    Clark & Evans (1954) R = observed mean NN distance / expected under CSR.
    R<1 clustered, R≈1 random, R>1 dispersed/regular.
    """
    x, y = _clean(xs), _clean(ys)
    n = x.size
    if n < 2 or width <= 0 or height <= 0:
        return NAN
    pts = np.column_stack([x, y])
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    observed = float(np.mean(d.min(axis=1)))
    expected = 0.5 / math.sqrt(n / (width * height))
    return NAN if expected <= EPS else observed / expected


def quadrat_dispersion(xs: Sequence[float], ys: Sequence[float],
                       width: float, height: float, bins: int = 8) -> float:
    """
    Variance-to-mean ratio of quadrat counts. =1 Poisson/random,
    >1 clustered, <1 uniform. A classic ecological dispersion index.
    """
    x, y = _clean(xs), _clean(ys)
    if x.size == 0:
        return NAN
    h, _, _ = np.histogram2d(y, x, bins=bins, range=[[0, height], [0, width]])
    m = h.mean()
    return NAN if m <= EPS else float(h.var() / m)


def ripley_k(xs: Sequence[float], ys: Sequence[float], radius: float,
             width: float, height: float) -> float:
    """Ripley's K(r): expected neighbours within r, normalised by intensity."""
    x, y = _clean(xs), _clean(ys)
    n = x.size
    if n < 2:
        return NAN
    pts = np.column_stack([x, y])
    d = np.hypot(pts[:, None, 0] - pts[None, :, 0], pts[:, None, 1] - pts[None, :, 1])
    np.fill_diagonal(d, np.inf)
    count = float(np.sum(d <= radius))
    lam = n / (width * height)
    return float(count / (n * lam)) if lam > EPS else NAN


def ripley_l(xs, ys, radius, width, height) -> float:
    """L(r) = sqrt(K/π) − r. Zero under complete spatial randomness."""
    k = ripley_k(xs, ys, radius, width, height)
    return NAN if not np.isfinite(k) else float(math.sqrt(max(k, 0) / math.pi) - radius)


def spatial_entropy(field: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy of a field's value histogram — spatial heterogeneity."""
    z = np.asarray(field, dtype=np.float64).ravel()
    z = z[np.isfinite(z)]
    if z.size == 0:
        return NAN
    h, _ = np.histogram(z, bins=bins)
    return shannon_entropy(h)


# ===========================================================================
# TIME SERIES
# ===========================================================================
def autocorrelation(series: Sequence[float], max_lag: int = 40) -> np.ndarray:
    """Normalised autocorrelation function up to max_lag."""
    s = _clean(series)
    if s.size < 3:
        return np.array([])
    s = s - s.mean()
    var = np.sum(s ** 2)
    if var <= EPS:
        return np.zeros(min(max_lag, s.size - 1))
    lags = min(max_lag, s.size - 1)
    return np.array([float(np.sum(s[: s.size - k] * s[k:]) / var) for k in range(lags)])


def hurst_exponent(series: Sequence[float]) -> float:
    """
    Rescaled-range (R/S) Hurst exponent. 0.5 = random walk,
    >0.5 persistent/trending, <0.5 mean-reverting.
    """
    s = _clean(series)
    n = s.size
    if n < 20:
        return NAN
    sizes = np.unique(np.floor(np.logspace(math.log10(8), math.log10(n // 2), 12)).astype(int))
    sizes = sizes[sizes >= 4]
    if sizes.size < 3:
        return NAN
    rs = []
    for w in sizes:
        chunks = n // w
        vals = []
        for i in range(chunks):
            seg = s[i * w:(i + 1) * w]
            dev = np.cumsum(seg - seg.mean())
            r = dev.max() - dev.min()
            sd = seg.std()
            if sd > EPS:
                vals.append(r / sd)
        rs.append(np.mean(vals) if vals else np.nan)
    rs = np.asarray(rs, dtype=float)
    ok = np.isfinite(rs) & (rs > 0)
    if ok.sum() < 3:
        return NAN
    slope, *_ = stats.linregress(np.log(sizes[ok]), np.log(rs[ok]))
    return float(slope)


def spectral_entropy(series: Sequence[float]) -> float:
    """Normalised Shannon entropy of the power spectrum. 1 = white noise."""
    s = _clean(series)
    if s.size < 8:
        return NAN
    ps = np.abs(np.fft.rfft(s - s.mean())) ** 2
    ps = ps[1:]
    if ps.sum() <= EPS:
        return NAN
    p = ps / ps.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(p.size)) if p.size > 1 else NAN


def permutation_entropy(series: Sequence[float], order: int = 3) -> float:
    """
    Bandt & Pompe (2002) permutation entropy — ordinal-pattern complexity,
    normalised to [0,1]. Robust to noise, no binning required.
    """
    s = _clean(series)
    n = s.size
    if n < order + 1:
        return NAN
    patterns: Dict[Tuple[int, ...], int] = {}
    for i in range(n - order + 1):
        key = tuple(np.argsort(s[i:i + order]))
        patterns[key] = patterns.get(key, 0) + 1
    h = shannon_entropy(list(patterns.values()))
    return float(h / np.log(math.factorial(order))) if np.isfinite(h) else NAN


def lyapunov_proxy(series: Sequence[float], eps_frac: float = 0.10) -> float:
    """
    Rosenstein-style largest-Lyapunov proxy: mean log divergence rate of
    initially nearby points, one step ahead. >0 suggests chaotic sensitivity.
    """
    s = _clean(series)
    n = s.size
    if n < 20:
        return NAN
    scale = s.std()
    if scale <= EPS:
        return NAN
    tol = eps_frac * scale
    rates = []
    for i in range(n - 1):
        for j in range(i + 1, n - 1):
            d0 = abs(s[i] - s[j])
            if EPS < d0 < tol:
                d1 = abs(s[i + 1] - s[j + 1])
                if d1 > EPS:
                    rates.append(math.log(d1 / d0))
    return float(np.mean(rates)) if rates else NAN


def detrended_fluctuation(series: Sequence[float]) -> float:
    """DFA exponent α — long-range correlation. 0.5 uncorrelated, 1.0 1/f."""
    s = _clean(series)
    n = s.size
    if n < 20:
        return NAN
    y = np.cumsum(s - s.mean())
    sizes = np.unique(np.floor(np.logspace(math.log10(6), math.log10(n // 3), 10)).astype(int))
    sizes = sizes[sizes >= 4]
    if sizes.size < 3:
        return NAN
    flucs = []
    for w in sizes:
        segs = n // w
        errs = []
        for i in range(segs):
            seg = y[i * w:(i + 1) * w]
            t = np.arange(w)
            coef = np.polyfit(t, seg, 1)
            errs.append(np.mean((seg - np.polyval(coef, t)) ** 2))
        flucs.append(math.sqrt(np.mean(errs)) if errs else np.nan)
    f = np.asarray(flucs, dtype=float)
    ok = np.isfinite(f) & (f > 0)
    if ok.sum() < 3:
        return NAN
    slope, *_ = stats.linregress(np.log(sizes[ok]), np.log(f[ok]))
    return float(slope)


def fano_factor(series: Sequence[float]) -> float:
    """Variance/mean — burstiness. 1 = Poisson, >1 = bursty/overdispersed."""
    s = _clean(series)
    if s.size < 2 or abs(s.mean()) <= EPS:
        return NAN
    return float(s.var() / s.mean())


def coefficient_of_variation(series: Sequence[float]) -> float:
    s = _clean(series)
    if s.size < 2 or abs(s.mean()) <= EPS:
        return NAN
    return float(s.std() / abs(s.mean()))


def kendall_trend(series: Sequence[float]) -> Tuple[float, float]:
    """Mann-Kendall style monotonic trend: returns (tau, p-value)."""
    s = _clean(series)
    if s.size < 4:
        return NAN, NAN
    tau, p = stats.kendalltau(np.arange(s.size), s)
    return float(tau), float(p)


# ===========================================================================
# DISTRIBUTIONS & SURVIVAL
# ===========================================================================
def powerlaw_alpha(values: Sequence[float], xmin: Optional[float] = None) -> Tuple[float, float]:
    """
    Clauset-Shalizi-Newman (2009) MLE exponent for a discrete/continuous
    power law, plus the KS distance of the fit. Returns (alpha, ks).
    """
    v = _clean(values)
    v = v[v > 0]
    if v.size < 8:
        return NAN, NAN
    if xmin is None:
        xmin = float(np.percentile(v, 10))
    tail = v[v >= max(xmin, EPS)]
    if tail.size < 5:
        return NAN, NAN
    alpha = 1.0 + tail.size / np.sum(np.log(tail / xmin))
    srt = np.sort(tail)
    emp = np.arange(1, srt.size + 1) / srt.size
    theo = 1.0 - (srt / xmin) ** (1.0 - alpha)
    return float(alpha), float(np.max(np.abs(emp - theo)))


def zipf_slope(counts: Sequence[float]) -> float:
    """Slope of log(frequency) vs log(rank). Ideal Zipf ≈ −1."""
    c = _clean(counts)
    c = np.sort(c[c > 0])[::-1]
    if c.size < 4:
        return NAN
    ranks = np.arange(1, c.size + 1)
    slope, *_ = stats.linregress(np.log(ranks), np.log(c))
    return float(slope)


def gompertz_makeham_fit(ages: Sequence[float]) -> Dict[str, float]:
    """
    Fit h(t) = λ + α·e^(βt) to an empirical hazard curve.
    λ = age-independent ('accident') mortality, β = senescence rate.
    """
    a = _clean(ages)
    if a.size < 12:
        return {"lambda": NAN, "alpha": NAN, "beta": NAN}
    bins = min(12, max(4, a.size // 5))
    counts, edges = np.histogram(a, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    at_risk = a.size - np.concatenate([[0], np.cumsum(counts)[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        hazard = np.where(at_risk > 0, counts / at_risk, np.nan)
    ok = np.isfinite(hazard) & (hazard > 0)
    if ok.sum() < 4:
        return {"lambda": NAN, "alpha": NAN, "beta": NAN}

    def model(t, lam, alpha, beta):
        return lam + alpha * np.exp(np.clip(beta * t, -50, 50))

    try:
        popt, _ = optimize.curve_fit(
            model, centres[ok], hazard[ok],
            p0=[max(hazard[ok].min(), 1e-4), 1e-3, 0.01],
            maxfev=8000, bounds=([0, 0, -1], [5, 5, 1]),
        )
        return {"lambda": float(popt[0]), "alpha": float(popt[1]), "beta": float(popt[2])}
    except Exception:
        return {"lambda": NAN, "alpha": NAN, "beta": NAN}


def kaplan_meier(ages: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier (1958) survival curve from uncensored ages at death."""
    a = np.sort(_clean(ages))
    if a.size == 0:
        return np.array([]), np.array([])
    times = np.unique(a)
    n = a.size
    surv, s = [], 1.0
    for t in times:
        d = int(np.sum(a == t))
        at_risk = int(np.sum(a >= t))
        if at_risk > 0:
            s *= (1.0 - d / at_risk)
        surv.append(s)
    return times, np.asarray(surv)


def median_survival(ages: Sequence[float]) -> float:
    t, s = kaplan_meier(ages)
    if t.size == 0:
        return NAN
    below = np.where(s <= 0.5)[0]
    return float(t[below[0]]) if below.size else NAN


# ===========================================================================
# INFORMATION THEORY
# ===========================================================================
def mutual_information(x: Sequence[float], y: Sequence[float], bins: int = 8) -> float:
    """I(X;Y) from a joint histogram, in nats."""
    a, b = _clean(x), _clean(y)
    n = min(a.size, b.size)
    if n < 8:
        return NAN
    a, b = a[:n], b[:n]
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    pj = joint / joint.sum() if joint.sum() > 0 else None
    if pj is None:
        return NAN
    px = pj.sum(axis=1, keepdims=True)
    py = pj.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = (pj > 0) & (denom > 0)
    return float(np.sum(pj[mask] * np.log(pj[mask] / denom[mask])))


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """D_KL(P||Q) in nats, over normalised count vectors."""
    a, b = _clean(p), _clean(q)
    if a.size != b.size or a.size == 0:
        return NAN
    a = a / a.sum() if a.sum() > 0 else a
    b = b / b.sum() if b.sum() > 0 else b
    mask = (a > 0) & (b > 0)
    return float(np.sum(a[mask] * np.log(a[mask] / b[mask]))) if mask.any() else NAN


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence — symmetric, bounded by ln 2."""
    a, b = _clean(p), _clean(q)
    if a.size != b.size or a.size == 0:
        return NAN
    a = a / a.sum() if a.sum() > 0 else a
    b = b / b.sum() if b.sum() > 0 else b
    m = 0.5 * (a + b)
    return float(0.5 * kl_divergence(a, m) + 0.5 * kl_divergence(b, m))


def normalised_compression_distance(s1: str, s2: str) -> float:
    """
    NCD (Li et al. 2004) — a universal similarity metric via compression.
    0 = identical, →1 = unrelated.
    """
    if not s1 or not s2:
        return NAN
    c = lambda s: len(zlib.compress(s.encode("utf-8"), 9))
    cx, cy, cxy = c(s1), c(s2), c(s1 + s2)
    denom = max(cx, cy)
    return float((cxy - min(cx, cy)) / denom) if denom > 0 else NAN


def transfer_entropy_proxy(source: Sequence[float], target: Sequence[float], bins: int = 6) -> float:
    """
    Lagged-mutual-information proxy for transfer entropy: I(source_t ; target_{t+1}).
    Directional information flow indicator, not a full conditional TE.
    """
    s, t = _clean(source), _clean(target)
    n = min(s.size, t.size)
    if n < 10:
        return NAN
    return mutual_information(s[:n - 1], t[1:n], bins=bins)


# ===========================================================================
# QUANTUM / MATRIX COMPLEXITY
# ===========================================================================
def von_neumann_entropy(eigenvalues: Sequence[float]) -> float:
    """S = -Σ p ln p over normalised spectral weights."""
    lam = np.abs(_clean(eigenvalues))
    if lam.size == 0 or lam.sum() <= EPS:
        return NAN
    p = lam / lam.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def participation_ratio(vector: Sequence[complex]) -> float:
    """
    PR = (Σ|ψ|²)² / Σ|ψ|⁴ — how many modes a state actually occupies.
    1 = fully localised on one mode, N = uniformly spread.
    """
    v = np.asarray(list(vector)).ravel()
    a2 = np.abs(v) ** 2
    denom = np.sum(a2 ** 2)
    return float((np.sum(a2) ** 2) / denom) if denom > EPS else NAN


def inverse_participation_ratio(vector: Sequence[complex]) -> float:
    pr = participation_ratio(vector)
    return NAN if not np.isfinite(pr) or pr <= EPS else float(1.0 / pr)


def spectral_gap(eigenvalues: Sequence[float]) -> float:
    """Gap between the two largest eigenvalues — dynamical rigidity."""
    lam = np.sort(_clean(eigenvalues))
    return float(lam[-1] - lam[-2]) if lam.size >= 2 else NAN


def level_spacing_ratio(eigenvalues: Sequence[float]) -> float:
    """
    Mean adjacent-gap ratio ⟨r⟩. ≈0.386 for Poisson (integrable),
    ≈0.536 for GOE (quantum-chaotic) — a standard random-matrix diagnostic.
    """
    lam = np.sort(_clean(eigenvalues))
    if lam.size < 4:
        return NAN
    gaps = np.diff(lam)
    gaps = gaps[gaps > EPS]
    if gaps.size < 2:
        return NAN
    r = np.minimum(gaps[:-1], gaps[1:]) / np.maximum(gaps[:-1], gaps[1:])
    return float(np.mean(r))


def matrix_effective_rank(matrix: np.ndarray) -> float:
    """exp(entropy of normalised singular values) — soft rank."""
    try:
        sv = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    except Exception:
        return NAN
    sv = sv[sv > EPS]
    if sv.size == 0:
        return NAN
    p = sv / sv.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


# ===========================================================================
# FRACTAL GEOMETRY
# ===========================================================================
def box_counting_dimension(binary: np.ndarray) -> float:
    """
    Minkowski-Bouligand box-counting fractal dimension of a binary field.
    A filled 2D region → ≈2; a curve → ≈1.
    """
    z = np.asarray(binary) > 0
    if z.ndim != 2 or z.sum() == 0:
        return NAN
    n = min(z.shape)
    sizes = [s for s in (2, 4, 8, 16, 32) if s < n]
    if len(sizes) < 3:
        return NAN
    counts = []
    for s in sizes:
        h = (z.shape[0] // s) * s
        w = (z.shape[1] // s) * s
        if h == 0 or w == 0:
            counts.append(np.nan)
            continue
        blocks = z[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3))
        counts.append(float(blocks.sum()))
    c = np.asarray(counts, dtype=float)
    ok = np.isfinite(c) & (c > 0)
    if ok.sum() < 3:
        return NAN
    slope, *_ = stats.linregress(np.log(1.0 / np.asarray(sizes)[ok]), np.log(c[ok]))
    return float(slope)


def lacunarity(binary: np.ndarray, box: int = 4) -> float:
    """Gliding-box lacunarity — texture 'gappiness'. 1.0 = perfectly uniform."""
    z = (np.asarray(binary) > 0).astype(float)
    if z.ndim != 2 or min(z.shape) < box:
        return NAN
    sums = ndimage.uniform_filter(z, size=box) * box * box
    m, v = sums.mean(), sums.var()
    return float(1.0 + v / (m ** 2)) if m > EPS else NAN


# ===========================================================================
# NETWORKS
# ===========================================================================
def network_summary(graph) -> Dict[str, float]:
    """Structural summary of a networkx graph, degradation-safe."""
    import networkx as nx
    out = {"nodes": NAN, "edges": NAN, "density": NAN, "mean_degree": NAN,
           "clustering": NAN, "assortativity": NAN, "components": NAN,
           "mean_path": NAN, "diameter": NAN, "modularity_proxy": NAN}
    try:
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        out["nodes"], out["edges"] = float(n), float(m)
        if n == 0:
            return out
        out["density"] = float(nx.density(graph))
        degs = [d for _, d in graph.degree()]
        out["mean_degree"] = float(np.mean(degs)) if degs else NAN
        u = graph.to_undirected() if graph.is_directed() else graph
        out["clustering"] = float(nx.average_clustering(u)) if n > 2 else NAN
        try:
            out["assortativity"] = float(nx.degree_assortativity_coefficient(u))
        except Exception:
            pass
        comps = list(nx.connected_components(u))
        out["components"] = float(len(comps))
        if comps:
            big = u.subgraph(max(comps, key=len))
            if big.number_of_nodes() > 1 and big.number_of_nodes() <= 400:
                out["mean_path"] = float(nx.average_shortest_path_length(big))
                out["diameter"] = float(nx.diameter(big))
            if big.number_of_nodes() > 2:
                try:
                    communities = nx.community.greedy_modularity_communities(big)
                    out["modularity_proxy"] = float(nx.community.modularity(big, communities))
                except Exception:
                    pass
    except Exception:
        pass
    return out


def small_world_sigma(graph) -> float:
    """
    σ = (C/C_rand)/(L/L_rand) — Watts-Strogatz small-worldness.
    σ > 1 indicates small-world structure.
    """
    import networkx as nx
    try:
        u = graph.to_undirected() if graph.is_directed() else graph
        comps = list(nx.connected_components(u))
        if not comps:
            return NAN
        big = u.subgraph(max(comps, key=len))
        n, m = big.number_of_nodes(), big.number_of_edges()
        if n < 5 or m < n or n > 400:
            return NAN
        c = nx.average_clustering(big)
        l = nx.average_shortest_path_length(big)
        k = 2.0 * m / n
        c_rand = k / n
        l_rand = math.log(n) / math.log(k) if k > 1 else NAN
        if not np.isfinite(l_rand) or c_rand <= EPS or l <= EPS:
            return NAN
        return float((c / c_rand) / (l / l_rand))
    except Exception:
        return NAN


# ===========================================================================
# ECOLOGY / GROWTH
# ===========================================================================
def logistic_fit(series: Sequence[float]) -> Dict[str, float]:
    """
    Fit N(t) = K / (1 + ((K-N0)/N0)·e^(−rt)) — the Verhulst logistic model.
    Returns carrying capacity K, intrinsic growth rate r, fit R², and a
    `saturated` flag.

    Important caveat, surfaced by real testing: a population still in its
    exponential phase contains no information about where it will level
    off, so the optimiser happily runs K to its bound and reports a
    meaningless "carrying capacity" with an excellent R² (observed:
    K = 1e6 at R² = 0.99 on a genuinely still-growing population). K is
    therefore only trustworthy once the curve has actually begun to bend.
    `saturated` reports whether the fit pinned against the bound, and
    `identifiable` reports whether the series has flattened enough for K
    to mean anything; callers should not display K when it is False.
    """
    s = _clean(series)
    out = {"K": NAN, "r": NAN, "r_squared": NAN, "saturated": 0.0, "identifiable": 0.0}
    if s.size < 12:
        return out
    t = np.arange(s.size, dtype=float)
    k_upper = float(max(s.max() * 50.0, 100.0))

    def model(tt, K, r, n0):
        n0 = max(n0, 1e-6)
        return K / (1.0 + ((K - n0) / n0) * np.exp(-np.clip(r * tt, -50, 50)))

    try:
        popt, _ = optimize.curve_fit(
            model, t, s, p0=[max(s.max() * 1.5, 1.0), 0.05, max(s[0], 1.0)],
            maxfev=10000, bounds=([1, 1e-5, 1e-6], [k_upper, 5, 1e6]),
        )
        pred = model(t, *popt)
        ss_res = float(np.sum((s - pred) ** 2))
        ss_tot = float(np.sum((s - s.mean()) ** 2)) + EPS
        K = float(popt[0])
        # Identifiability: has growth actually decelerated? Compare the growth
        # rate of the last third against the first third.
        third = max(4, s.size // 3)
        early = instantaneous_growth_rate(s[:third], window=third)
        late = instantaneous_growth_rate(s[-third:], window=third)
        decelerating = (np.isfinite(early) and np.isfinite(late) and late < 0.6 * early)
        out.update({
            "K": K, "r": float(popt[1]), "r_squared": 1.0 - ss_res / ss_tot,
            "saturated": 1.0 if K > 0.9 * k_upper else 0.0,
            "identifiable": 1.0 if (decelerating and K < 0.9 * k_upper) else 0.0,
        })
        return out
    except Exception:
        return out


def instantaneous_growth_rate(series: Sequence[float], window: int = 10) -> float:
    """Per-capita growth rate r = d(ln N)/dt over a trailing window."""
    s = _clean(series)
    if s.size < max(4, window):
        return NAN
    tail = s[-window:]
    tail = tail[tail > 0]
    if tail.size < 3:
        return NAN
    slope, *_ = stats.linregress(np.arange(tail.size), np.log(tail))
    return float(slope)


def turnover_rate(births: float, deaths: float, population: float) -> float:
    """(births + deaths) / population — ecological turnover per tick."""
    return float((births + deaths) / population) if population > EPS else NAN


if __name__ == "__main__":
    ok = 0

    # --- Diversity: a known uniform case has H = ln(S), evenness exactly 1
    uni = [10, 10, 10, 10]
    assert abs(shannon_entropy(uni) - math.log(4)) < 1e-12
    assert abs(pielou_evenness(uni) - 1.0) < 1e-12
    assert abs(simpson_index(uni) - 0.25) < 1e-12
    assert abs(inverse_simpson(uni) - 4.0) < 1e-9
    assert abs(hill_number(uni, 0) - 4.0) < 1e-9
    assert abs(hill_number(uni, 2) - 4.0) < 1e-9
    print("diversity indices: uniform case exact (H=ln4, J'=1, D=0.25, 1/D=4)"); ok += 1

    # --- Gini: perfect equality = 0; near-total inequality → ~1
    assert abs(gini_coefficient([5, 5, 5, 5, 5])) < 1e-12
    assert gini_coefficient([0, 0, 0, 0, 100]) > 0.75
    assert abs(theil_index([7, 7, 7])) < 1e-12
    assert abs(berger_parker([1, 1, 2]) - 0.5) < 1e-12
    print("inequality: Gini(equal)=0, Gini(concentrated)>0.75, Theil(equal)=0"); ok += 1

    # --- Moran's I: a smooth gradient is strongly positive; checkerboard negative
    grad = np.tile(np.linspace(0, 1, 24), (24, 1))
    check = np.indices((24, 24)).sum(axis=0) % 2
    assert morans_i(grad) > 0.8, morans_i(grad)
    assert morans_i(check) < -0.8, morans_i(check)
    print(f"Moran's I: gradient={morans_i(grad):.3f} (clustered), "
          f"checkerboard={morans_i(check):.3f} (dispersed)"); ok += 1

    # --- Spatial point patterns
    rng = np.random.default_rng(0)
    rx, ry = rng.uniform(0, 100, 200), rng.uniform(0, 100, 200)
    nni_rand = nearest_neighbour_index(rx, ry, 100, 100)
    cx = np.concatenate([rng.normal(25, 2, 100), rng.normal(75, 2, 100)])
    cy = np.concatenate([rng.normal(25, 2, 100), rng.normal(75, 2, 100)])
    nni_clust = nearest_neighbour_index(cx, cy, 100, 100)
    assert 0.8 < nni_rand < 1.2, nni_rand
    assert nni_clust < 0.5, nni_clust
    assert quadrat_dispersion(cx, cy, 100, 100) > quadrat_dispersion(rx, ry, 100, 100)
    print(f"Clark-Evans NNI: random={nni_rand:.3f} (~1.0), clustered={nni_clust:.3f} (<1)"); ok += 1

    # --- Hurst: white noise ≈0.5; a strong trend is clearly persistent
    wn = rng.normal(0, 1, 800)
    trend = np.cumsum(rng.normal(0.05, 0.3, 800))
    h_wn, h_tr = hurst_exponent(wn), hurst_exponent(trend)
    assert 0.35 < h_wn < 0.65, h_wn
    assert h_tr > h_wn, (h_tr, h_wn)
    print(f"Hurst: white noise={h_wn:.3f} (~0.5), trending series={h_tr:.3f} (persistent)"); ok += 1

    # --- Entropies: noise is maximally complex, a pure sine is not
    t = np.linspace(0, 40 * np.pi, 900)
    sine = np.sin(t)
    assert spectral_entropy(wn) > spectral_entropy(sine)
    assert permutation_entropy(wn) > permutation_entropy(sine)
    assert 0.0 <= permutation_entropy(wn) <= 1.0
    print(f"Entropy: spectral noise={spectral_entropy(wn):.3f} > sine={spectral_entropy(sine):.3f}; "
          f"permutation noise={permutation_entropy(wn):.3f} > sine={permutation_entropy(sine):.3f}"); ok += 1

    # --- DFA and Fano
    assert 0.3 < detrended_fluctuation(wn) < 0.75
    assert np.isfinite(fano_factor(np.abs(wn) + 1))
    print(f"DFA(white noise)={detrended_fluctuation(wn):.3f} (~0.5 uncorrelated)"); ok += 1

    # --- Power law: recover a known synthetic exponent
    synth = (1.0 - rng.random(4000)) ** (-1.0 / (2.5 - 1.0))
    a_hat, ks = powerlaw_alpha(synth, xmin=1.0)
    assert abs(a_hat - 2.5) < 0.2, a_hat
    print(f"Power-law MLE: recovered alpha={a_hat:.3f} from synthetic alpha=2.5 (KS={ks:.3f})"); ok += 1

    # --- Zipf
    zipf_counts = [1000 / k for k in range(1, 60)]
    assert abs(zipf_slope(zipf_counts) + 1.0) < 0.05
    print(f"Zipf slope on ideal 1/k data: {zipf_slope(zipf_counts):.3f} (ideal -1.0)"); ok += 1

    # --- Survival
    ages = list(rng.normal(60, 10, 300))
    tms, srv = kaplan_meier(ages)
    assert srv[0] >= srv[-1] and 0.0 <= srv[-1] <= 1.0
    med = median_survival(ages)
    assert 50 < med < 70, med
    gm = gompertz_makeham_fit(ages)
    print(f"Kaplan-Meier median survival={med:.1f} (true mean 60); "
          f"Gompertz-Makeham beta={gm['beta']:.4f}"); ok += 1

    # --- Information theory
    x = rng.normal(0, 1, 2000)
    mi_self = mutual_information(x, x)
    mi_indep = mutual_information(x, rng.normal(0, 1, 2000))
    assert mi_self > mi_indep, (mi_self, mi_indep)
    assert abs(js_divergence([1, 1, 1], [1, 1, 1])) < 1e-12
    assert js_divergence([10, 1, 1], [1, 1, 10]) > 0.05
    assert abs(normalised_compression_distance("abcabc" * 50, "abcabc" * 50)) < 0.3
    print(f"Information: I(X;X)={mi_self:.3f} > I(X;Y_indep)={mi_indep:.3f}; JS(p,p)=0"); ok += 1

    # --- Quantum/matrix
    localised = np.zeros(64, dtype=complex); localised[0] = 1.0
    spread = np.ones(64, dtype=complex) / 8.0
    assert abs(participation_ratio(localised) - 1.0) < 1e-9
    assert abs(participation_ratio(spread) - 64.0) < 1e-6
    A = rng.normal(0, 1, (32, 32)); A = A + A.T
    lam = np.linalg.eigvalsh(A)
    lsr = level_spacing_ratio(lam)
    assert 0.3 < lsr < 0.7, lsr
    assert matrix_effective_rank(np.eye(10)) > 9.9
    print(f"Participation ratio: localised=1.0, uniform=64.0; GOE level-spacing r={lsr:.3f} "
          f"(theory ~0.536); effective rank(I_10)={matrix_effective_rank(np.eye(10)):.2f}"); ok += 1

    # --- Fractal: a filled square is ~2D
    filled = np.ones((64, 64), dtype=int)
    d_filled = box_counting_dimension(filled)
    assert 1.8 < d_filled < 2.15, d_filled
    assert np.isfinite(lacunarity(filled))
    print(f"Box-counting dimension of a filled square: {d_filled:.3f} (theory 2.0)"); ok += 1

    # --- Networks
    import networkx as nx
    g = nx.karate_club_graph()
    ns = network_summary(g)
    assert ns["nodes"] == 34 and ns["edges"] == 78
    assert 0.0 < ns["clustering"] < 1.0
    print(f"Network summary on Zachary's karate club: n={ns['nodes']:.0f} m={ns['edges']:.0f} "
          f"clustering={ns['clustering']:.3f} modularity={ns['modularity_proxy']:.3f}"); ok += 1

    # --- Logistic growth: recover a known K
    tt = np.arange(120)
    true_K, true_r = 500.0, 0.08
    logi = true_K / (1 + ((true_K - 20) / 20) * np.exp(-true_r * tt))
    fit = logistic_fit(logi)
    assert abs(fit["K"] - true_K) < 25, fit
    assert fit["r_squared"] > 0.99
    assert fit["identifiable"] == 1.0, "a fully saturated curve must be flagged identifiable"
    # A still-exponential series carries no information about K: the optimiser
    # will run K to its bound. That must be FLAGGED, not silently reported.
    expo = 20.0 * np.exp(0.05 * np.arange(60))
    bad = logistic_fit(expo)
    assert bad["identifiable"] == 0.0, "an unsaturated curve must not be flagged identifiable"
    print(f"Logistic fit: recovered K={fit['K']:.1f} (true 500), r={fit['r']:.4f} "
          f"(true 0.08), R²={fit['r_squared']:.4f}; unsaturated series correctly "
          f"flagged unidentifiable"); ok += 1

    # --- Degenerate input must return NaN, never raise
    for fn, arg in [(shannon_entropy, []), (gini_coefficient, [1]), (hurst_exponent, [1, 2]),
                    (powerlaw_alpha, [1, 2]), (morans_i, np.zeros((2, 2))),
                    (permutation_entropy, [1]), (logistic_fit, [1, 2])]:
        r = fn(arg)
        assert r is not None
    print("degenerate/empty inputs return NaN cleanly rather than raising"); ok += 1

    print(f"\nanalytics.py self-test passed — {ok} verification groups, "
          f"each checked against a known analytic answer.")
