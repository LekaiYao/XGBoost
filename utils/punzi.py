"""Approved PbPb23 X Punzi arithmetic; no model selection or sample reassignment."""
import ast
import numpy as np

GRID = np.arange(5, 96, dtype=float) / 100


def sidebands_from_selection(expression):
    tree = ast.parse(expression, mode="eval")
    lower, upper = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "Bmass":
            if len(node.ops) != 1 or not isinstance(node.comparators[0], ast.Constant):
                raise ValueError("Unsupported mass selection; explicit two strict sidebands required")
            value = float(node.comparators[0].value)
            if isinstance(node.ops[0], ast.Gt): lower.append(value)
            elif isinstance(node.ops[0], ast.Lt): upper.append(value)
            else: raise ValueError("Mass boundaries must be strict")
    bands = list(zip(sorted(lower), sorted(upper)))
    if len(lower) != 2 or len(upper) != 2 or not bands[0][0] < bands[0][1] <= bands[1][0] < bands[1][1]:
        raise ValueError("Expected two disjoint training sidebands")
    return bands


def mass_mask(mass, bands):
    return np.logical_or.reduce([(mass > low) & (mass < high) for low, high in bands])


def denominator(background):
    background = np.asarray(background, dtype=float)
    if np.any(~np.isfinite(background)) or np.any(background < 0):
        raise ValueError("Invalid background")
    return 3425.0 / 104.0 + 10.0 * np.sqrt(background)


def scan(mc_score, mc_weight, data_mass, data_score, bands, probability=1.0):
    arrays = [np.asarray(x, dtype=float) for x in (mc_score, mc_weight, data_mass, data_score)]
    mc_score, mc_weight, data_mass, data_score = arrays
    if any(np.any(~np.isfinite(x)) for x in arrays) or np.any(mc_weight <= 0):
        raise ValueError("Nonfinite input or nonpositive weight")
    if len(mc_score) != len(mc_weight) or len(data_mass) != len(data_score):
        raise ValueError("Unaligned input")
    if not 0 < probability <= 1 or mc_weight.sum() <= 0:
        raise ValueError("Invalid inclusion probability or zero MC denominator")
    ratio = .06 / sum(hi-lo for lo, hi in bands)
    sides = [(data_mass > lo) & (data_mass < hi) for lo, hi in bands]
    rows = []
    for cut in GRID:
        counts = [int(np.count_nonzero(side & (data_score > cut))) for side in sides]
        passed = float(mc_weight[mc_score > cut].sum())
        epsilon = passed / float(mc_weight.sum())
        background = sum(counts) * ratio / probability
        d = float(denominator(background))
        rows.append(dict(score=float(cut), left_entries=counts[0], right_entries=counts[1],
                         mc_pass_sumw=passed, mc_pre_score_sumw=float(mc_weight.sum()),
                         epsilon=epsilon, B=background, D=d, FOM=epsilon/d))
    return rows


def optimal(rows):
    if not rows or any(not np.isfinite(r['FOM']) for r in rows):
        raise ValueError("Invalid curve")
    maximum = max(r['FOM'] for r in rows)
    ties = sorted(r['score'] for r in rows if r['FOM'] == maximum)
    return dict(score=ties[0], FOM=maximum, exact_ties=ties,
                boundary=ties[0] in (float(GRID[0]), float(GRID[-1])),
                uninformative=maximum == 0)
