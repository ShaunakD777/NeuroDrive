"""Adam: momentum's direction, RMSProp's per-parameter step size, and a start-up correction.

Adam (Adaptive Moment Estimation) keeps two running averages per parameter: the *first moment* m,
the recent average gradient (which way to go, smoothed like momentum), and the *second moment* v,
the recent average squared gradient (its typical size, as in RMSProp).

For every parameter θ with gradient g, learning rate η, step number t (1, 2, 3, ...), decays β₁
and β₂ (the fractions of the old averages that are kept) and a tiny ε against division by zero,
with m and v starting at 0:

    m = β₁ · m + (1 − β₁) · g
    v = β₂ · v + (1 − β₂) · g²
    m̂ = m / (1 − β₁ᵗ),   v̂ = v / (1 − β₂ᵗ)        (bias correction)
    θ_new = θ − η · m̂ / (√v̂ + ε)

Bias correction: because m and v start at 0 they are too small for the first steps (after step 1,
m = 0.1·g with β₁ = 0.9). Dividing by 1 − βᵗ scales them back up (1 − 0.9¹ = 0.1, so m̂ = g); as t
grows, βᵗ goes to 0 and the correction fades away. Formula and defaults follow the Adam paper
(Kingma and Ba, 2015) and PyTorch, with ε added after the square root.

Worked example: L(w) = (w − 3)², η = 0.1, w = 0. Step 1: g = −6, m̂ = −6, v̂ = 36, w = 0.1 (the
first step is always η in size). Step 2: g = −5.8, m = −1.12, v = 0.069604, m̂ = −5.89474,
v̂ = 34.8194, w = 0.1 + 0.1 · 5.89474 / 5.90080 = 0.19990.
"""

import numpy as np

from neurodrive.core.optim.base import Array, check_learning_rate, check_pairs


class Adam:
    """Adam. Keeps a first moment, a second moment and the step count t between steps."""

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.learning_rate = check_learning_rate(learning_rate)
        beta1, beta2, eps = float(beta1), float(beta2), float(eps)
        for name, beta in (("beta1", beta1), ("beta2", beta2)):
            if not 0.0 <= beta < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {beta}")
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {eps}")
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.first_moments: list[Array] | None = None
        self.second_moments: list[Array] | None = None
        self.t = 0

    def reset(self) -> None:
        """Forget both moments and the step count, so the next step behaves like a first step."""
        self.first_moments = None
        self.second_moments = None
        self.t = 0

    def step(self, params: list[Array], grads: list[Array]) -> list[Array]:
        """Update m, v and t, then return [θ − η·m̂ / (√v̂ + ε)] (inputs untouched)."""
        check_pairs(params, grads)
        if self.first_moments is None or self.second_moments is None:
            self.first_moments = [np.zeros(np.shape(p)) for p in params]
            self.second_moments = [np.zeros(np.shape(p)) for p in params]
        elif len(self.first_moments) != len(params) or any(
            m.shape != np.shape(p) for m, p in zip(self.first_moments, params, strict=True)
        ):
            raise ValueError("parameter shapes changed since the last step; call reset() first")
        b1, b2 = self.beta1, self.beta2
        self.t += 1
        self.first_moments = [
            b1 * m + (1.0 - b1) * g for m, g in zip(self.first_moments, grads, strict=True)
        ]
        self.second_moments = [
            b2 * v + (1.0 - b2) * np.square(g)
            for v, g in zip(self.second_moments, grads, strict=True)
        ]
        correction1 = 1.0 - b1**self.t  # 1 − β₁ᵗ
        correction2 = 1.0 - b2**self.t  # 1 − β₂ᵗ
        return [
            p - self.learning_rate * (m / correction1) / (np.sqrt(v / correction2) + self.eps)
            for p, m, v in zip(params, self.first_moments, self.second_moments, strict=True)
        ]
