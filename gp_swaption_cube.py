
"""
gp_swaption_cube.py

Reference implementation of a constrained Gaussian Process (GP) regression approach
to build an arbitrage-aware swaption price cube, following the methodology in:

  A. Cousin, A. Deleplace, A. Misko (2022),
  "Gaussian Process Regression for Swaption Cube Construction under No-Arbitrage Constraints"
  Risks 10(12):232. https://doi.org/10.3390/risks10120232

Key features implemented:
- 3D anisotropic stationary Matérn 5/2 kernel
- Finite-dimensional GP via tri-linear hat basis on a regular grid
- Unconstrained log-marginal likelihood and gradient (for MLE of hyperparameters)
- Linear inequality constraints enforcing:
    * Strike monotonicity (payer: decreasing; receiver: increasing)
    * Strike convexity
    * "In-plane triangular" inequality on a fictitious 1-year maturity step
- MAP under linear constraints using scipy.optimize (trust-constr)

This module is provided for educational/research purposes. It is not optimized for production.

Dependencies: numpy, scipy

Author: OGOUNKOLA Ebénézer
License: MIT (code); cite the Risks 2022 paper above if you use this methodology in research/production.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from quadprog import solve_qp
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize, LinearConstraint
from scipy.linalg import cho_factor, cho_solve, solve, eigh

# ------------------------------
# Utilities
# ------------------------------

def _ensure_1d(a: ArrayLike) -> np.ndarray:
    a = np.asarray(a).reshape(-1)
    return a

def _pairwise_sqdist(X: np.ndarray, Y: np.ndarray, length: float) -> np.ndarray:
    # (n,d), (m,d) -> (n,m) squared distance scaled by length^2
    diff = X[:, None, :] - Y[None, :, :]
    return np.sum((diff / length) ** 2, axis=-1)

# ------------------------------
# Kernel: Matérn 5/2 (anisotropic product across T, t, K)
# ------------------------------

@dataclass
class Matern52Kernel3D:
    theta_T: float
    theta_t: float
    theta_K: float
    sigma: float

    def corr_1d(self, h: np.ndarray, theta: float) -> np.ndarray:
        # Matérn 5/2 correlation in 1D for distances |h|
        r = np.abs(h) / theta + 1e-15  # avoid zero division
        sqrt5r = np.sqrt(5.0) * r
        return (1.0 + sqrt5r + 5.0 * r**2 / 3.0) * np.exp(-sqrt5r)

    def K(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # X, Y: (..., 3) in rescaled [0,1]^3 coordinates
        # Compute product kernel across dimensions
        TX, tX, KX = X[:, 0], X[:, 1], X[:, 2]
        TY, tY, KY = Y[:, 0], Y[:, 1], Y[:, 2]

        RT = self.corr_1d(TX[:, None] - TY[None, :], self.theta_T)
        Rt = self.corr_1d(tX[:, None] - tY[None, :], self.theta_t)
        RK = self.corr_1d(KX[:, None] - KY[None, :], self.theta_K)

        return (self.sigma ** 2) * RT * Rt * RK

    def grad_params(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Precompute partial derivatives dK/dparam for params in (theta_T, theta_t, theta_K, sigma).
        Returns a dict of 4 arrays with shape (N, N).
        """
      #  N = X.shape[0]
        K_full = self.K(X, X)
        # Grad w.r.t sigma: dK/dsigma = 2*sigma * Corr
        dK_dsigma = 2.0 * self.sigma * (K_full / (self.sigma ** 2))

        # For theta derivatives, use finite diff on 1D corr factors to keep code simple and readable.
        # (Analytic derivatives are faster but more verbose.)
        eps = 1e-6

        def corr_1d_param(theta_name: str, theta_val: float, axis: int) -> np.ndarray:
           # theta_p = theta_val + eps
            tmp = Matern52Kernel3D(
                theta_T=self.theta_T + (eps if theta_name == "theta_T" else 0.0),
                theta_t=self.theta_t + (eps if theta_name == "theta_t" else 0.0),
                theta_K=self.theta_K + (eps if theta_name == "theta_K" else 0.0),
                sigma=self.sigma,
            )
            K_p = tmp.K(X, X)
            return (K_p - K_full) / eps

        dK_dtheta_T = corr_1d_param("theta_T", self.theta_T, 0)
        dK_dtheta_t = corr_1d_param("theta_t", self.theta_t, 1)
        dK_dtheta_K = corr_1d_param("theta_K", self.theta_K, 2)

        return {
            "theta_T": dK_dtheta_T,
            "theta_t": dK_dtheta_t,
            "theta_K": dK_dtheta_K,
            "sigma": dK_dsigma,
        }

# ------------------------------
# Grid + hat basis
# ------------------------------

@dataclass
class RegularGrid3D:
    T_vals: np.ndarray  # shape (N_T,)
    t_vals: np.ndarray  # shape (N_t,)
    K_vals: np.ndarray  # shape (N_K,)

    def points(self) -> np.ndarray:
        TT, tt, KK = np.meshgrid(self.T_vals, self.t_vals, self.K_vals, indexing='ij')
        P = np.column_stack((TT.ravel(), tt.ravel(), KK.ravel()))
        return P

    def shape(self) -> Tuple[int, int, int]:
        return (self.T_vals.size, self.t_vals.size, self.K_vals.size)

def trilin_hat_basis(x: np.ndarray, grid: RegularGrid3D) -> np.ndarray:
    """
    Evaluate tri-linear hat basis weights at x (3-vector in [0,1]^3) over a regular grid.
    Returns a sparse-like dense vector phi of length N = N_T * N_t * N_K, with up to 8 non-zero entries.
    """
    x = np.asarray(x).reshape(3,)
    Tv, tv, Kv = grid.T_vals, grid.t_vals, grid.K_vals

    def one_dim_weights(val, coords):
        # Find the two neighboring grid points
        idx = np.searchsorted(coords, val) - 1
        idx = np.clip(idx, 0, len(coords) - 2)
        x0, x1 = coords[idx], coords[idx + 1]
        if x1 == x0:
            w0, w1 = 1.0, 0.0
        else:
            w1 = (val - x0) / (x1 - x0)
            w0 = 1.0 - w1
        return idx, idx + 1, w0, w1

    i0, i1, wTi0, wTi1 = one_dim_weights(x[0], Tv)
    j0, j1, wtj0, wtj1 = one_dim_weights(x[1], tv)
    k0, k1, wKk0, wKk1 = one_dim_weights(x[2], Kv)

    NT, Nt, NK = grid.shape()
    N = NT * Nt * NK
    phi = np.zeros(N)
    # Helper to map (i,j,k) -> flat index
    def idx3(i, j, k): return (i * Nt + j) * NK + k

    # 8 corners
    corners = [
        (i0, j0, k0, wTi0 * wtj0 * wKk0),
        (i0, j0, k1, wTi0 * wtj0 * wKk1),
        (i0, j1, k0, wTi0 * wtj1 * wKk0),
        (i0, j1, k1, wTi0 * wtj1 * wKk1),
        (i1, j0, k0, wTi1 * wtj0 * wKk0),
        (i1, j0, k1, wTi1 * wtj0 * wKk1),
        (i1, j1, k0, wTi1 * wtj1 * wKk0),
        (i1, j1, k1, wTi1 * wtj1 * wKk1),
    ]
    for i, j, k, w in corners:
        phi[idx3(i, j, k)] += w
    return phi

def build_Phi(X: np.ndarray, grid: RegularGrid3D) -> np.ndarray:
    """Build Φ(X) where each row is hat basis at X_i."""
    X = np.asarray(X)
    rows = [trilin_hat_basis(x, grid) for x in X]
    return np.vstack(rows)

# ------------------------------
# Hyperparameter MLE (unconstrained likelihood)
# ------------------------------

@dataclass
class GPFiniteApprox:
    grid: RegularGrid3D
    kernel: Matern52Kernel3D
    noise: float  # ς (std)

    def _GammaN(self) -> np.ndarray:
        P = self.grid.points()
        return self.kernel.K(P, P)

    def _Phi(self, X: np.ndarray) -> np.ndarray:
        return build_Phi(X, self.grid)

    def neg_log_marg_lik_and_grad(self, y: np.ndarray, X: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute negative log marginal likelihood and its gradient wrt params:
        [theta_T, theta_t, theta_K, sigma, noise].
        Based on L(Θ) in the paper (Eq. 7), using finite-dimensional approximation (Φ Γ Φ^T + ς^2 I).
        """
        y = _ensure_1d(y)
        Phi = self._Phi(X)
        Gamma = self._GammaN()
        A = Phi @ Gamma @ Phi.T + (self.noise ** 2 + 1e-6) * np.eye(len(y))

        # Solve using Cholesky for stability
        cF = cho_factor(A, lower=True, check_finite=False)
        alpha = cho_solve(cF, y, check_finite=False)

        # Log marginal likelihood
        logdetA = 2.0 * np.sum(np.log(np.diag(cF[0])))
        L = -0.5 * y @ alpha - 0.5 * logdetA - 0.5 * len(y) * np.log(2 * np.pi)
        nll = -L

        # Gradients
        # dA/dGamma = Phi (dGamma) Phi^T; dA/dnoise = 2*noise*I
        grads_K = self.kernel.grad_params(self.grid.points())
        grad = np.zeros(5)

        # helper: trace(A^{-1} dA) and y^T A^{-1} dA A^{-1} y = alpha^T dA alpha
        Ainv_y = alpha  # already A^{-1} y
        Ainv = cho_solve(cF, np.eye(A.shape[0]), check_finite=False)

        def contrib(dA):
            # d/dθ (-L) = 0.5 * [ tr(A^{-1} dA) - y^T A^{-1} dA A^{-1} y ]
            return 0.5 * (np.trace(Ainv @ dA) - Ainv_y @ (dA @ Ainv_y))

        # theta_T
        dA = Phi @ grads_K["theta_T"] @ Phi.T
        grad[0] = contrib(dA)
        # theta_t
        dA = Phi @ grads_K["theta_t"] @ Phi.T
        grad[1] = contrib(dA)
        # theta_K
        dA = Phi @ grads_K["theta_K"] @ Phi.T
        grad[2] = contrib(dA)
        # sigma
        dA = Phi @ grads_K["sigma"] @ Phi.T
        grad[3] = contrib(dA)
        # noise
        dA = 2.0 * self.noise * np.eye(len(y))
        grad[4] = contrib(dA)

        return nll, grad

    def fit_mle(self, y: np.ndarray, X: np.ndarray, bounds: Optional[List[Tuple[float,float]]] = None, restarts: int = 10, seed: int = 0):
        """
        Maximize unconstrained likelihood (as recommended in Bachoc et al., 2019).
        Updates kernel thetas/sigma and noise in-place.
        """
        rng = np.random.RandomState(seed)
        y = _ensure_1d(y)

        if bounds is None:
            # Reasonable default bounds in rescaled domain
            bounds = [
                (1e-3, 2.0),  # theta_T
                (1e-3, 2.0),  # theta_t
                (1e-3, 2.0),  # theta_K
                (1e-6, 10.0), # sigma
                (1e-8, 1.0),  # noise
            ]

        def f_and_g(p):
            k = Matern52Kernel3D(p[0], p[1], p[2], p[3])
            gpf = GPFiniteApprox(self.grid, k, p[4])
            nll, grad = gpf.neg_log_marg_lik_and_grad(y, X)
            return nll, grad

        best = None
        # Initial centers
        p0_center = np.array([self.kernel.theta_T, self.kernel.theta_t, self.kernel.theta_K, self.kernel.sigma, self.noise])

        for r in range(restarts):
            # random start around center (log-normal-ish jitter)
            jitter = np.exp(rng.normal(0.0, 0.2, size=5))
            p0 = np.clip(p0_center * jitter, [b[0] for b in bounds], [b[1] for b in bounds])

            res = minimize(lambda p: f_and_g(p)[0],
                           x0=p0,
                           jac=lambda p: f_and_g(p)[1],
                           method="L-BFGS-B",
                           bounds=bounds)
            if (best is None) or (res.fun < best.fun):
                best = res

        p = best.x 
        self.kernel.theta_T, self.kernel.theta_t, self.kernel.theta_K, self.kernel.sigma = p[:4]
        self.noise = p[4]
        return best

# ------------------------------
# Constraints builder
# ------------------------------

def build_monotonicity_convexity_constraints(grid: RegularGrid3D, kind: str) -> LinearConstraint:
    """
    Build linear inequalities for monotonicity and convexity in strike K.

    kind: "payer" (decreasing in K), "receiver" (increasing in K)

    For all (T_i, t_j): 
      monotonicity: ξ_{i,j,k} - ξ_{i,j,k+1} >= 0  (payer)
                    ξ_{i,j,k+1} - ξ_{i,j,k} >= 0  (receiver)
      convexity:    ξ_{i,j,k+2} - 2 ξ_{i,j,k+1} + ξ_{i,j,k} >= 0
    """
    NT, Nt, NK = grid.shape()
    N = NT * Nt * NK
    TOLERANCE_EPSILON = 1e-5
    def flat(i, j, k): return (i * Nt + j) * NK + k

    rows = []
    lbs = []
    ubs = []

    # Monotonicity
    for i in range(NT):
        for j in range(Nt):
            for k in range(NK - 1):
                row = np.zeros(N)
                if kind == "payer":
                    # ξ_{i,j,k} - ξ_{i,j,k+1} >= 0
                    row[flat(i, j, k)] = 1.0
                    row[flat(i, j, k + 1)] = -1.0
                else:
                    # receiver: ξ_{i,j,k+1} - ξ_{i,j,k} >= 0
                    row[flat(i, j, k + 1)] = 1.0
                    row[flat(i, j, k)] = -1.0
                rows.append(row)
                lbs.append(-TOLERANCE_EPSILON)
                ubs.append(np.inf)

    # Convexity
    for i in range(NT):
        for j in range(Nt):
            for k in range(NK - 2):
                row = np.zeros(N)
                # ξ_{k+2} - 2 ξ_{k+1} + ξ_k >= 0
                row[flat(i, j, k)] = 1.0
                row[flat(i, j, k + 1)] = -2.0
                row[flat(i, j, k + 2)] = 1.0
                rows.append(row)
                lbs.append(-TOLERANCE_EPSILON)
                ubs.append(np.inf)

    A = np.vstack(rows) if rows else np.zeros((0, N))
    lb = np.array(lbs) if lbs else np.zeros((0,))
    ub = np.array(ubs) if ubs else np.zeros((0,))
    return LinearConstraint(A, lb, ub)

def build_inplane_triangular_constraints(grid: RegularGrid3D, fictitious_T: np.ndarray) -> LinearConstraint:
    """
    Build weaker in-plane triangular constraints on a fictitious maturity grid:
      Sw(T_l, T_{l+1}, K) + Sw(T_{l+1}, T_{l+2}, K) - Sw(T_l, T_{l+2}, K) >= 0

    We enforce this at all strike grid nodes K_k, using hat basis rows for those 3 points.
    """
    NT, Nt, NK = grid.shape()
    N = NT * Nt * NK

    # Precompute basis rows for all (T*, t*, K_grid_k)
    def phi_at(Tstar, tstar, k_idx):
        x = np.array([Tstar, tstar, grid.K_vals[k_idx]])
        return trilin_hat_basis(x, grid)

    rows = []
    lbs = []
    ubs = []

    # fictitious grid must lie inside [0,1] (already rescaled)
    # enforce for consecutive triplets (l, l+1, l+2)
    for l in range(len(fictitious_T) - 2):
        T1, T2, T3 = fictitious_T[l], fictitious_T[l + 1], fictitious_T[l + 2]
        # use tenor equal to (T_{l+1}-T_l) and (T_{l+2}-T_{l+1}), and (T_{l+2}-T_l)
        t12 = T2 - T1
        t23 = T3 - T2
        t13 = T3 - T1
        for k in range(NK):
            phi1 = phi_at(T1, t12, k)
            phi2 = phi_at(T2, t23, k)
            phi3 = phi_at(T1, t13, k)
            row = phi1 + phi2 - phi3
            rows.append(row)
            lbs.append(0.0)
            ubs.append(np.inf)

    A = np.vstack(rows) if rows else np.zeros((0, N))
    lb = np.array(lbs) if lbs else np.zeros((0,))
    ub = np.array(ubs) if ubs else np.zeros((0,))
    return LinearConstraint(A, lb, ub)

# ------------------------------
# MAP under constraints
# ------------------------------



def solve_map(Phi: np.ndarray, Gamma: np.ndarray, y: np.ndarray, noise: float, constraints: List[LinearConstraint]) -> np.ndarray:
    """
    Solves the Constrained Quadratic Program (CQP) for the MAP of xi (ξ).
    
    The objective is:
      minimize  xi^T Gamma^{-1} xi + (1/sigma^2) || y - Phi xi ||^2
    
    This is reformulated into the canonical QP form:
      minimize  (1/2) xi^T H xi + f^T xi
      s.t.      C^T xi >= b (quadprog format for inequality constraints)
    
    Args:
        Phi: The observation matrix (Φ).
        Gamma: The covariance matrix of the prior (Γ).
        y: The observation vector.
        noise: The noise variance (σ).
        constraints: A list of scipy.optimize.LinearConstraint objects.
        
    Returns:
        The optimal parameter vector xi_map.
    """
    
    # 1. QP Matrix Calculation
    
    # Ensure y is 1D and get dimensions
    y = np.asarray(y).flatten()
    
    # --- Compute H (Hessian Matrix) ---
    # Use Eigh for stability and to handle PSD Gamma
    w, V = eigh(Gamma)
    w = np.clip(w, 1e-12, None) # Clip eigenvalues to ensure invertibility (for precision)
    Ginv = (V * (1.0 / w)) @ V.T
    
    # H = 2 * (Gamma^-1 + (Phi^T Phi) / sigma^2)
    # This matrix H MUST be symmetric and positive definite (guaranteed here)
    H = 2.0 * (Ginv + (Phi.T @ Phi) / (noise ** 2))
    
    # --- Compute f (Linear Vector) ---
    # f = -2 * (Phi^T y) / sigma^2
    f = -2.0 * (Phi.T @ y) / (noise ** 2)
    
    # --- Force H to be symmetric (crucial for quadprog) ---
    H = (H + H.T) / 2.0


    # 2. Constraint Conversion (Scipy format -> quadprog format)

    # quadprog requires: C.T @ xi >= b, where C is the constraint matrix
    # We must separate equality (A_eq @ xi = b_eq) and inequality (A_ub @ xi <= b_ub) constraints
    
    A_ub_list = []
    b_ub_list = []
    A_eq_list = []
    b_eq_list = []
    
    for c in constraints:
        A = c.A
        lb = c.lb
        ub = c.ub
        
        # Inequality constraints (A @ xi <= ub)
        # quadprog requires C.T @ xi >= b. We convert A @ xi <= ub to (-A) @ xi >= (-ub)
        is_ub = ~np.isinf(ub)
        if np.any(is_ub):
            A_ub_list.append(-A[is_ub, :])
            b_ub_list.append(-ub[is_ub])
            
        # Inequality constraints (A @ xi >= lb)
        # A @ xi >= lb is already in the correct form
        is_lb = ~np.isinf(lb)
        if np.any(is_lb):
            A_ub_list.append(A[is_lb, :])
            b_ub_list.append(lb[is_lb])
            
        # Equality constraints (A @ xi = lb = ub)
        # Must be handled separately by quadprog
        is_eq = (~np.isinf(lb)) & (~np.isinf(ub)) & (lb == ub)
        if np.any(is_eq):
            A_eq_list.append(A[is_eq, :])
            b_eq_list.append(lb[is_eq])
            
    # Combine all inequality constraints (A_ub -> C_T)
    if A_ub_list:
        C_T = np.vstack(A_ub_list)
        b_qp = np.hstack(b_ub_list)
    else:
        # No inequality constraints
        C_T = np.zeros((0, H.shape[0]))
        b_qp = np.array([])
        
    # Combine all equality constraints
    if A_eq_list:
        A_eq = np.vstack(A_eq_list)
        b_eq = np.hstack(b_eq_list)
    else:
        A_eq = np.zeros((0, H.shape[0]))
        b_eq = np.array([])

    
    # 3. Solve the QP
    
    # Call quadprog (solve_qp)
    # Arguments: G (H), a (f), C (C_T), b (b_qp), meq (number of equality constraints)
    
    n_eq = A_eq.shape[0]
    
    # For quadprog, C (constraints matrix) must contain the equality constraints first,
    # followed by inequality constraints. The vector b must follow the same order.
    
    # C_T for quadprog: A_eq.T followed by C_T (inequalities).
    C_T_quadprog = np.hstack([A_eq.T, C_T.T]) 
    b_quadprog = np.hstack([b_eq, b_qp])
    
    xi_opt, fval, *_ = solve_qp(H, f, C_T_quadprog, b_quadprog, meq=n_eq)
    
    return xi_opt

# ------------------------------
# Main model wrapper
# ------------------------------

class ConstrainedSwaptionGPCube:
    """
    Wrapper that holds grid, kernel, and provides:
    - fit (MLE)
    - build constraints
    - compute MAP ξ̂
    - predict at arbitrary x
    """

    def __init__(self, T_grid: np.ndarray, t_grid: np.ndarray, K_grid: np.ndarray,
                 theta_T=0.3, theta_t=0.3, theta_K=0.3, sigma=1.0, noise=0.01,
                 kind: str = "payer"):
        """
        kind: "payer" or "receiver" (controls strike monotonicity direction)
        """
        # Rescale inputs to [0,1]
        self.T_min, self.T_max = float(np.min(T_grid)), float(np.max(T_grid))
        self.t_min, self.t_max = float(np.min(t_grid)), float(np.max(t_grid))
        self.K_min, self.K_max = float(np.min(K_grid)), float(np.max(K_grid))

        def rescale(v, vmin, vmax):
            if vmax == vmin:
                return np.zeros_like(v)
            return (v - vmin) / (vmax - vmin)

        self.T_grid_raw = np.asarray(T_grid).astype(float)
        self.t_grid_raw = np.asarray(t_grid).astype(float)
        self.K_grid_raw = np.asarray(K_grid).astype(float)

        self.T_grid = rescale(self.T_grid_raw, self.T_min, self.T_max)
        self.t_grid = rescale(self.t_grid_raw, self.t_min, self.t_max)
        self.K_grid = rescale(self.K_grid_raw, self.K_min, self.K_max)

        self.grid = RegularGrid3D(self.T_grid, self.t_grid, self.K_grid)
        self.kernel = Matern52Kernel3D(theta_T, theta_t, theta_K, sigma)
        self.noise = noise
        self.kind = kind

        self._Gamma = None
        self._Phi_cache = {}

        self.xi_map = None  # MAP coefficients on grid

    # Rescale helpers for external points
    def _rescale_point(self, x: np.ndarray) -> np.ndarray:
        T, t, K = x
        def rs(v, vmin, vmax):
            return 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
        return np.array([
            rs(T, self.T_min, self.T_max),
            rs(t, self.t_min, self.t_max),
            rs(K, self.K_min, self.K_max),
        ], dtype=float)

    def Phi(self, X_rescaled: np.ndarray) -> np.ndarray:
        key = X_rescaled.tobytes()
        if key in self._Phi_cache:
            return self._Phi_cache[key]
        Phi = build_Phi(X_rescaled, self.grid)
        self._Phi_cache[key] = Phi
        return Phi

    def Gamma(self) -> np.ndarray:
        if self._Gamma is None:
            self._Gamma = self.kernel.K(self.grid.points(), self.grid.points())
        return self._Gamma

    def fit_mle(self, y: ArrayLike, X: ArrayLike, restarts: int = 10, seed: int = 0):
        X = np.asarray(X, dtype=float)
        # Rescale X to [0,1]^3
        X_rescaled = np.stack([
            (X[:,0] - self.T_min) / (self.T_max - self.T_min + 1e-15),
            (X[:,1] - self.t_min) / (self.t_max - self.t_min + 1e-15),
            (X[:,2] - self.K_min) / (self.K_max - self.K_min + 1e-15),
        ], axis=1)
        gpf = GPFiniteApprox(self.grid, self.kernel, self.noise)
        res = gpf.fit_mle(np.asarray(y, dtype=float), X_rescaled, restarts=restarts, seed=seed)
        # sync back
        self.kernel = gpf.kernel
        self.noise = gpf.noise
        # clear cached Gamma
        self._Gamma = None
        self._Phi_cache.clear()
        return res

    def build_constraints(self, year_step_inplane: bool = True) -> List[LinearConstraint]:
        cons = []
        cons.append(build_monotonicity_convexity_constraints(self.grid, self.kind))
        if year_step_inplane:
            # Build fictitious maturity grid with 1-year step in original scale, then rescale
            # If T_raw is in years already, we can interpolate a unit step grid spanning [T_min, T_max].
            T_raw = self.T_grid_raw
            # Create integer-like years covering the observed span
            start = np.ceil(np.min(T_raw))
            end = np.floor(np.max(T_raw))
            if end - start < 2:
                # not enough points for a 1-year triangle; fall back to using the existing T grid
                fict_T_raw = np.unique(T_raw)
            else:
                fict_T_raw = np.arange(start, end + 1, 1.0)

            # Rescale to [0,1]
            fict_T = (fict_T_raw - self.T_min) / (self.T_max - self.T_min + 1e-15)
            cons.append(build_inplane_triangular_constraints(self.grid, fict_T))
        return cons

    def fit_map(self, y: ArrayLike, X: ArrayLike, year_step_inplane: bool = True) -> np.ndarray:
        # Rescale X to [0,1]^3
        X = np.asarray(X, dtype=float)
        X_rescaled = np.stack([
            (X[:,0] - self.T_min) / (self.T_max - self.T_min + 1e-15),
            (X[:,1] - self.t_min) / (self.t_max - self.t_min + 1e-15),
            (X[:,2] - self.K_min) / (self.K_max - self.K_min + 1e-15),
        ], axis=1)

        Phi = self.Phi(X_rescaled)
        Gamma = self.Gamma()
        cons = self.build_constraints(year_step_inplane=year_step_inplane)

        x = solve_map(Phi, Gamma, np.asarray(y, dtype=float), self.noise, cons)
        self.xi_map = x
        return x

    def predict(self, X_query: ArrayLike) -> np.ndarray:
        """
        Evaluate the MAP cube MSN_w at arbitrary (T, t, K) points using the hat basis and xi_map.
        X_query: (n, 3) in original (unscaled) coordinates
        """
        if self.xi_map is None:
            raise RuntimeError("MAP coefficients not set. Call fit_map first.")
        X_query = np.asarray(X_query, dtype=float)
        preds = np.zeros(X_query.shape[0])
        for i, x in enumerate(X_query):
            xr = self._rescale_point(x)
            phi = trilin_hat_basis(xr, self.grid)
            preds[i] = phi @ self.xi_map
        return preds

# ------------------------------
# Minimal usage example (won't actually run a full calibration here)
# ------------------------------

#if __name__ == "__main__":
    # Example: Build a small cube and fit MAP on fake data
 #   T = np.array([5, 10, 15, 20], dtype=float)
  #  t = np.array([1, 5, 10, 20, 30], dtype=float)
   # K = np.linspace(-0.02, 0.02, 7)

    # Create a toy dataset: sample a few points and pretend they're noisy prices
 #   rng = np.random.RandomState(42)
  #  n_obs = 40
  #  X_obs = np.column_stack([
  #      rng.choice(T, size=n_obs),
  #      rng.choice(t, size=n_obs),
   #     rng.choice(K, size=n_obs),
 #   ])
    # Toy "true" function: decreasing in K (payer-like), some convexity
 #   def toy_price(x):
 #       T_, t_, K_ = x
 #       base = 0.02 * np.exp(-0.02 * (T_ + t_))
  #      smile = np.maximum(0.0, 0.02 - 5.0 * (K_ + 0.005) ** 2)
  #      return base + smile

  #  y_obs = np.array([toy_price(x) for x in X_obs]) + 0.0005 * rng.randn(n_obs)

 #   model = ConstrainedSwaptionGPCube(T, t, K, theta_T=0.4, theta_t=0.4, theta_K=0.4, sigma=0.5, noise=0.003, kind="payer")
    # Optional MLE (on toy data this is not very meaningful)
  #  model.fit_mle(y_obs, X_obs, restarts=3, seed=0)

    # Compute MAP under constraints
 #   model.fit_map(y_obs, X_obs, year_step_inplane=True)

    # Predict on a few points
 #   Xq = np.array([
#        [10, 5, -0.01],
 #       [10, 5,  0.00],
 #       [10, 5,  0.01],
  #  ], dtype=float)
  #  preds = model.predict(X_obs[3:6,:])
  #  print("Predictions at query points:", preds)
