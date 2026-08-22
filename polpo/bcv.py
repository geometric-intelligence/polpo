import numpy as np


class BCVBlock:
    def __init__(self, row_idx, col_idx):
        self.row_idx = row_idx
        self.col_idx = col_idx

    def _partition(self, X):
        row_mask = np.zeros(X.shape[0], dtype=bool)
        col_mask = np.zeros(X.shape[1], dtype=bool)

        row_mask[self.row_idx] = True
        col_mask[self.col_idx] = True

        return (
            X[np.ix_(row_mask, col_mask)],
            X[np.ix_(row_mask, ~col_mask)],
            X[np.ix_(~row_mask, col_mask)],
            X[np.ix_(~row_mask, ~col_mask)],
        )

    def fit(self, X):
        self.A_, self.B_, self.C_, self.D_ = self._partition(X)

        self.U_, self.s_, self.Vt_ = np.linalg.svd(
            self.D_,
            full_matrices=False,
        )

        return self

    def predict(self, rank):
        # A_k=B D_k^{+} C

        U = self.U_[:, :rank]
        s = self.s_[:rank]
        V = self.Vt_[:rank].T

        # if constructing full matrix
        # D_pinv = V @ np.diag(1 / s) @ U.T

        return (self.B_ @ V / s) @ (U.T @ self.C_)

    def approximation(self, rank):
        # $D_k=U_k \Sigma_k V_k^{\top}$
        U = self.U_[:, :rank]
        s = self.s_[:rank]
        Vt = self.Vt_[:rank]

        return (U * s) @ Vt

    def residual(self, rank):
        return self.A_ - self.predict(rank)

    def error(self, rank, normalize=False):
        # \mathrm{BCV}(k)=\left\|A-B D_k^{+} C\right\|_F^2

        error = np.linalg.norm(self.residual(rank), ord="fro") ** 2
        if normalize:
            error /= self.A_.size

        return error

    def _resolve_max_rank(self, max_rank):
        n_singular_values = len(self.s_)

        if max_rank is None:
            return n_singular_values

        if not 1 <= max_rank <= n_singular_values:
            raise ValueError(
                f"max_rank must be between 1 and {n_singular_values}, "
                f"got {max_rank}."
            )

        return max_rank

    def residuals(self, max_rank=None):
        max_rank = self._resolve_max_rank(max_rank)

        return np.stack([self.residual(rank) for rank in range(1, max_rank + 1)])

    def errors(self, max_rank=None, normalize=False):
        max_rank = self._resolve_max_rank(max_rank)

        return np.array(
            [self.error(rank, normalize=normalize) for rank in range(1, max_rank + 1)]
        )
