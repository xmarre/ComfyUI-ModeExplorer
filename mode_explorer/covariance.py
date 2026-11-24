"""Sample-covariance spectra of high-dimensional activations without a D x D matrix.

For N activations flattened to rows of X [N, D] and the sample-centred Xc = X - mean,
the covariance is Sigma = Xc^T Xc / N (population normalization, N, as in the
ReDiverse probe). Its nonzero eigenvalues equal those of the centred sample Gram
G = Xc Xc^T divided by N. At most N - 1 eigenvalues are nonzero, so every quantity
below describes the variance visible to N samples, not the intrinsic dimension of
the model.

    TV    = sum_j lambda_j                         total variance
    PR    = (sum_j lambda_j)^2 / sum_j lambda_j^2  participation ratio
    Conc  = TV / PR                                variance per effective direction
    erank = exp(-sum_j p_j log p_j), p = lambda/TV entropy effective rank

The Gram is accumulated in float64 over feature chunks. Each chunk is centred over
the samples before its product, which avoids cancellation when the mean dominates.
"""

import math

import torch

TOP_EIGENVALUES = 32


def centred_gram(X, chunk=1 << 20, mean=None):
    """Unnormalized centred Gram Xc Xc^T in float64 on X's device.

    X is [N, D] in any floating dtype. ``mean`` optionally supplies a fixed [D]
    centre (for projections onto another cohort's centre); by default the
    sample mean is used. The result does not depend on ``chunk``.
    """
    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError("Covariance input must be a nonempty [N, D] matrix.")
    n, d = X.shape
    G = torch.zeros(n, n, dtype=torch.float64, device=X.device)
    for start in range(0, d, chunk):
        block = X[:, start:start + chunk].double()
        block = block - (block.mean(0, keepdim=True) if mean is None
                         else mean[start:start + chunk].to(block.device, torch.float64))
        G.addmm_(block, block.T)
    return G


def column_mean(X, chunk=1 << 20, dtype=torch.float32):
    """Per-feature mean of [N, D] rows without a full-precision copy of X."""
    out = torch.empty(X.shape[1], dtype=dtype, device=X.device)
    for start in range(0, X.shape[1], chunk):
        out[start:start + chunk] = X[:, start:start + chunk].double().mean(0).to(dtype)
    return out


def eigenpairs(G):
    """Descending eigenvalues and eigenvectors of a symmetric N x N Gram (float64, on the CPU)."""
    G = G.double().cpu()
    G = (G + G.T) * 0.5
    values, vectors = torch.linalg.eigh(G)
    order = torch.arange(values.numel() - 1, -1, -1, device=values.device)
    return values[order], vectors[:, order]


def spectrum_stats(values, n, top=TOP_EIGENVALUES):
    """Statistics of covariance eigenvalues ``values / n`` (values from the unnormalized Gram)."""
    lam = (values.double().clamp_min(0) / n).cpu()
    tv = float(lam.sum())
    squares = float(lam.square().sum())
    if tv <= 0 or squares <= 0:
        return {"n": n, "normalization": "N", "tv": 0.0, "pr": 0.0, "conc": None, "entropy_rank": 0.0,
                "pr_cap": max(n - 1, 0), "top_eigenvalues": [0.0] * min(top, lam.numel()),
                "cumulative_explained": [], "pc1_share": None, "degenerate": True}
    pr = tv * tv / squares
    p = lam[lam > 0] / tv
    entropy_rank = math.exp(float(-(p * p.log()).sum()))
    cumulative = (lam.cumsum(0) / tv)[:top]
    return {
        "n": n, "normalization": "N", "tv": tv, "pr": pr, "conc": tv / pr,
        "entropy_rank": entropy_rank, "pr_cap": n - 1,
        "top_eigenvalues": [float(v) for v in lam[:top]],
        "cumulative_explained": [float(v) for v in cumulative],
        "pc1_share": float(lam[0] / tv), "degenerate": False,
    }


def principal_basis(X, vectors, values, k, chunk=1 << 20, dtype=torch.bfloat16, mean=None, relative_floor=1e-10):
    """ReDiverse-TF basis from the centred Gram eigenpairs (paper Eq. 2).

    u_k = Xc^T v_k / ||Xc^T v_k||, s_k = std(sqrt(lambda_k) v_k) with the biased (/N)
    standard deviation, where lambda_k is an eigenvalue of the unnormalized Gram.
    For a nonzero eigenvalue v_k is orthogonal to the ones vector, so s_k equals
    sqrt(lambda_k / N). Directions whose eigenvalue is at or below
    ``relative_floor`` times the leading eigenvalue are returned as zero rows with
    zero spread and counted out of ``valid``.

    Returns (U [k, D] in ``dtype``, s [k] float32, valid count).
    """
    n, d = X.shape
    k = min(k, n)
    lead = float(values[0]) if values.numel() else 0.0
    usable = [float(values[i]) > max(relative_floor * lead, 0.0) for i in range(k)]
    V = vectors[:, :k].to(device=X.device, dtype=torch.float32)
    U = torch.zeros(k, d, dtype=torch.float32, device=X.device)
    centre = column_mean(X, chunk) if mean is None else mean.to(X.device, torch.float32)
    for start in range(0, d, chunk):
        block = X[:, start:start + chunk].float() - centre[start:start + chunk]
        U[:, start:start + chunk] = V.T @ block
    norms = U.norm(dim=1)
    spreads = torch.zeros(k, dtype=torch.float32)
    valid = 0
    for i in range(k):
        if not usable[i] or not math.isfinite(float(norms[i])) or float(norms[i]) <= 0:
            U[i].zero_()
            continue
        U[i] /= norms[i]
        v = vectors[:, i].double().cpu()
        spreads[i] = float((math.sqrt(float(values[i])) * v).std(unbiased=False))
        valid += 1
    return U.to(dtype), spreads, valid


def project(X, U, mean=None, chunk=1 << 20):
    """Coordinates P = Xc U^T [N, k] of centred rows on unit directions U [k, D] (float64)."""
    n, d = X.shape
    P = torch.zeros(n, U.shape[0], dtype=torch.float64, device=X.device)
    centre = column_mean(X, chunk, torch.float64) if mean is None else mean.to(X.device, torch.float64)
    for start in range(0, d, chunk):
        block = X[:, start:start + chunk].double() - centre[start:start + chunk]
        P.addmm_(block, U[:, start:start + chunk].to(X.device, torch.float64).T)
    return P


def subspace_comparison(reference_spreads, coordinates, own_values, own_vectors, k=None):
    """Compare a reference (teacher) principal basis with a sample cohort (student).

    ``coordinates`` are the cohort's centred projections onto the reference unit
    directions [N, K] (from :func:`project`), ``own_values``/``own_vectors`` the
    cohort's own unnormalized Gram eigenpairs. Everything is expressed in the
    reference subspace; no D-dimensional basis is formed.

    * retained_variance_ratio[k]: cohort variance along u_k over the reference s_k^2.
    * alignment: ||U_ref^T U_own||_F^2 / K over the cohort's top-K directions, in
      [0, 1]; 1 means the cohort's top-K subspace lies inside the reference span.
    * deficit/excess: eigenvalues of C_ref - C_own in the reference subspace. The
      difference is indefinite in general; positive parts are suppressed variance,
      negative parts excess variance. It is not an injectable covariance as is.
    """
    P = coordinates.double()
    n, K = P.shape
    k = min(K if k is None else k, n)
    reference_var = reference_spreads[:K].double().cpu() ** 2
    C_own = (P.T @ P / n).cpu()
    own_var = C_own.diagonal()
    retained = [float(o / r) if r > 0 else None for o, r in zip(own_var, reference_var)]
    # U_ref^T u_own_j = P^T v_j / sqrt(lambda_j) for nonzero own eigenvalues.
    overlaps = []
    for j in range(k):
        lam = float(own_values[j])
        if lam <= 0:
            continue
        overlaps.append((P.T @ own_vectors[:, j].to(P.device, torch.float64)).cpu() / math.sqrt(lam))
    alignment = float(torch.stack(overlaps).square().sum() / k) if overlaps else None
    difference = torch.diag(reference_var) - C_own
    delta, basis = torch.linalg.eigh((difference + difference.T) * 0.5)
    return {
        "directions": K, "n": n, "reference_variance": [float(v) for v in reference_var],
        "cohort_variance": [float(v) for v in own_var], "retained_variance_ratio": retained,
        "subspace_alignment": alignment, "alignment_rank": k,
        "deficit_eigenvalues": [float(v) for v in delta.flip(0) if v > 0],
        "excess_eigenvalues": [float(-v) for v in delta if v < 0],
        "deficit_basis": basis.flip(1)[:, :int((delta > 0).sum())].T.tolist(),
        "deficit_trace": float(delta.clamp_min(0).sum()), "excess_trace": float((-delta).clamp_min(0).sum()),
    }


def positive_part(difference):
    """Positive semidefinite part of a symmetric matrix (spectral clipping)."""
    values, vectors = torch.linalg.eigh((difference + difference.T) * 0.5)
    return (vectors * values.clamp_min(0)) @ vectors.T
