import numpy as np


def full_svd(X):
    return np.linalg.svd(X, full_matrices=False)


class BCVBlock:
    def __init__(self, heldout_rows, heldout_cols, center=False, svd=full_svd):
        self.heldout_rows = heldout_rows
        self.heldout_cols = heldout_cols
        self.center = center
        self.svd = svd

    def _masks(self, X):
        row_mask = np.zeros(X.shape[0], dtype=bool)
        col_mask = np.zeros(X.shape[1], dtype=bool)

        row_mask[self.heldout_rows] = True
        col_mask[self.heldout_cols] = True

        return row_mask, col_mask

    def _partition(self, X, row_mask, col_mask):
        return (
            X[np.ix_(row_mask, col_mask)],
            X[np.ix_(row_mask, ~col_mask)],
            X[np.ix_(~row_mask, col_mask)],
            X[np.ix_(~row_mask, ~col_mask)],
        )

    def fit(self, X):
        row_mask, col_mask = self._masks(X)

        if self.center:
            self.mean_ = X[~row_mask].mean(axis=0)
            X = X - self.mean_
        else:
            self.mean_ = None

        self.A_, self.B_, self.C_, self.D_ = self._partition(X, row_mask, col_mask)

        self.U_, self.s_, self.Vt_ = self.svd(self.D_)

        return self

    def _resolve_rank(self, rank):
        n_singular_values = len(self.s_)

        if rank is None:
            return n_singular_values

        if not 1 <= rank <= n_singular_values:
            raise ValueError(
                f"rank must be between 1 and {n_singular_values}, got {rank}."
            )

        return rank

    def predict(self, rank=None):
        # A_k=B D_k^{+} C

        rank = self._resolve_rank(rank)

        U = self.U_[:, :rank]
        s = self.s_[:rank]
        V = self.Vt_[:rank].T

        # if constructing full matrix
        # D_pinv = V @ np.diag(1 / s) @ U.T

        s_inv = np.divide(1.0, s, out=np.zeros_like(s), where=s > self._get_tol())
        return (self.B_ @ V * s_inv) @ (U.T @ self.C_)

    def approximation(self, rank=None):
        # $D_k=U_k \Sigma_k V_k^{\top}$

        rank = self._resolve_rank(rank)

        U = self.U_[:, :rank]
        s = self.s_[:rank]
        Vt = self.Vt_[:rank]

        return (U * s) @ Vt

    def residual(self, rank=None):
        return self.A_ - self.predict(rank)

    def error(self, rank=None, normalize=False):
        # \mathrm{BCV}(k)=\left\|A-B D_k^{+} C\right\|_F^2

        error = np.linalg.norm(self.residual(rank), ord="fro") ** 2
        if normalize:
            error /= self.A_.size

        return error

    def predictions(self, max_rank=None):
        # exploits \hat{A}_k=\sum_{j=1}^k \frac{\left(B v_j\right)\left(u_j^{\top} C\right)}{\sigma_j}

        max_rank = self._resolve_rank(max_rank)

        U = self.U_[:, :max_rank]
        s = self.s_[:max_rank]
        V = self.Vt_[:max_rank].T

        BV = self.B_ @ V
        UTC = U.T @ self.C_

        BV = np.divide(
            BV,
            s,
            out=np.zeros_like(BV),
            where=s > self._get_tol(),
        )

        components = BV.T[:, :, None] * UTC[:, None, :]
        return np.cumsum(components, axis=0)

    def residuals(self, max_rank=None):
        return self.A_[None, :, :] - self.predictions(max_rank)

    def errors(self, max_rank=None, normalize=False):
        residuals = self.residuals(max_rank)

        errors = np.linalg.norm(residuals, ord="fro", axis=(1, 2)) ** 2

        if normalize:
            errors /= self.A_.size

        return errors

    def _get_tol(self):
        return np.finfo(self.s_.dtype).eps * max(self.D_.shape) * self.s_[0]
