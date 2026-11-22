"""Finite-amplitude response measurements on one aligned latent evaluation."""

import math

import torch

from .mechanism import measure_comparisons
from .objectives import cell_areas


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def norms(tensor):
    l2 = math.sqrt(_dot(tensor.reshape(-1), tensor.reshape(-1)))
    return {"l2": l2, "rms": l2 / math.sqrt(tensor.numel())}


def _dot(left, right):
    # FP32 dot kernels can lose precision on million-element responses.
    # Use double scalar products in bounded strips, with no large FP64 GEMM.
    return math.fsum(float(torch.dot(a.double(), b.double()))
                     for a, b in zip(left.split(65536), right.split(65536), strict=True))


def _cosine(dot, denom):
    return min(1.0, max(-1.0, dot / denom)) if denom else None


def response_geometry(vectors):
    names = list(vectors)
    values = [v.reshape(-1) for v in vectors.values()]
    gram = [[0.0] * len(values) for _ in values]
    for i, left in enumerate(values):
        for j in range(i + 1):
            gram[i][j] = gram[j][i] = _dot(left, values[j])
    lengths = [math.sqrt(row[i]) for i, row in enumerate(gram)]
    eig = torch.linalg.eigvalsh(torch.tensor(gram, dtype=torch.float64)).clamp_min(0).flip(0)
    total = float(eig.sum())
    masses = eig / total if total else eig
    effective = float(torch.exp(-torch.special.xlogy(masses, masses).sum())) if total else 0.0
    return {"arms": names, "gram": gram, "cosine": [[_cosine(gram[i][j], a * b) for j, b in enumerate(lengths)]
                                                     for i, a in enumerate(lengths)],
            "pairwise_l2": [[math.sqrt(max(0.0, gram[i][i] + gram[j][j] - 2.0 * gram[i][j])) for j in range(len(names))]
                            for i in range(len(names))],
            "singular_values_sampled_responses": eig.sqrt().tolist(),
            "entropy_effective_rank_sampled_responses": effective}


def cosine(left, right):
    a, b = left.flatten(), right.flatten()
    return _cosine(_dot(a, b), math.sqrt(_dot(a, a) * _dot(b, b)))


def _scalar(value):
    return (value[0] if value else None) if isinstance(value, list) else value


def proposal_metrics(proposal, state, settings, latent_shape):
    channels = proposal.shape[1]
    patch_grid = [math.ceil(v / 2) for v in latent_shape[-2:]]
    elements = math.prod(patch_grid) * channels
    areas = cell_areas(*patch_grid, *proposal.shape[-2:], settings["feature_grid"], proposal.device)
    expanded = float((proposal.square() * areas).sum(dtype=torch.float64).sqrt())
    realized = _scalar(state.get("realized_shift_rms"))
    error = _scalar(state.get("realization_error_rms"))
    pooled = state.get("pooled_edit", {})
    before, after = _scalar(state.get("objective")), _scalar(state.get("objective_after"))
    geometry = {"block": settings["feature_block"], "patch_grid": patch_grid, "channels": channels,
                "proposal_shape": list(proposal.shape)}
    if state.get("feature_geometry") is not None and geometry != state["feature_geometry"]:
        raise ValueError("feature_geometry_mismatch")
    compact = norms(proposal)
    if state.get("proposed_shift_rms") is not None and not math.isclose(
            compact["rms"], _scalar(state["proposed_shift_rms"]), rel_tol=2e-5, abs_tol=1e-8):
        raise ValueError("logged_compact_rms_mismatch")
    if realized is not None and (not math.isfinite(realized) or realized < 0 or bool(realized) != state["feature_applied"]):
        raise ValueError("realized_edit_application_mismatch")
    return {"geometry": geometry, "compact": compact, "expanded_intended": {"l2": expanded, "rms": expanded / math.sqrt(elements)},
            "applied": state["feature_applied"], "native_rounded_realized_rms": realized,
            "native_rounded_realized_l2": realized * math.sqrt(elements) if realized is not None else None,
            "realization_error_rms": error, "relative_realization_error": ratio(error, expanded / math.sqrt(elements)),
            "representation_relative_proposal_rms": _scalar(state.get("proposed_relative_rms")),
            "pooled_reference_cosine_before": _scalar(state.get("nearest_cosine")),
            "pooled_reference_cosine_after": _scalar(pooled.get("nearest_cosine_after")),
            "pooled_proposal_direction_cosine": _scalar(pooled.get("proposal_direction_cosine")),
            "pooled_applied_shift_rms": _scalar(pooled.get("applied_pooled_shift_rms")),
            "objective_before": before, "objective_after": after,
            "objective_displacement": after - before if before is not None and after is not None else None,
            "reference_slots": state["reference_slots"], "reference_history": state.get("reference_history"),
            "applied_actual_ordinal": state["ordinal"] if state["feature_applied"] else None,
            "sources": {"compact_and_expanded": "recomputed", "realized_rms_and_objective": "runtime_telemetry",
                        "realized_l2": "derived_from_runtime_rms_and_element_count", "realized_tensor": "unavailable"}}


def region_metrics(delta, crops, resolution):
    image_w, image_h = resolution
    h, w = delta.shape[-2:]
    if image_w != 8 * w or image_h != 8 * h:
        raise ValueError("ROI mapping requires the registered Wan21 8x spatial grid.")
    energy = delta.square().sum(dim=0, dtype=torch.float64)
    total = float(energy.sum())
    channels = delta.shape[0]
    rois = {}
    for name, (x0, y0, x1, y1) in crops.items():
        if not (0 <= x0 < x1 <= image_w and 0 <= y0 < y1 <= image_h):
            raise ValueError("ROI lies outside the planner image.")
        a, b, c, d = x0 // 8, y0 // 8, math.ceil(x1 / 8), math.ceil(y1 / 8)
        count = (c - a) * (d - b)
        area_energy = float(energy[b:d, a:c].sum())
        complement = max(0.0, total - area_energy)
        rois[name] = {"pixel_xyxy": [x0, y0, x1, y1], "latent_xyxy": [a, b, c, d],
                      "energy": area_energy, "rms": math.sqrt(area_energy / (count * channels)),
                      "energy_fraction": ratio(area_energy, total), "density_concentration": ratio(area_energy / count, total / (h * w)),
                      "complement_energy": complement,
                      "complement_rms": math.sqrt(complement / ((h * w - count) * channels)) if count < h * w else None}
    # Fixed bins partition the latent grid without overlapping adaptive bins.
    tiles_y, tiles_x = min(h, 8), min(w, 8)
    grid = [[float(energy[y*h//tiles_y:(y+1)*h//tiles_y, x*w//tiles_x:(x+1)*w//tiles_x].sum())
             for x in range(tiles_x)] for y in range(tiles_y)]
    return {"regions": rois, "energy_map": grid}


def spectral(baseline, deltas):
    basis = {**{arm: [(1., value)] for arm, value in deltas.items()}, "_sham": [(1., baseline)]}
    comparisons = {arm: ([(1., "_sham")], [(1., arm)]) for arm in deltas}
    return {arm: values["bands"] for arm, values in
            measure_comparisons(basis, comparisons, low=.0625, high=.125, patch_size=2).items()}


def odd_even(positive, negative):
    return {"odd": norms((positive - negative) * .5), "even": norms((positive + negative) * .5),
            "scope": "matched_initial_input_and_exact_negated_compact_proposal"}
