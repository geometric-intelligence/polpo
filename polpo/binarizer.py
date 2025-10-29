import itertools

import numpy as np


class IntervalBinarizer:
    def __init__(self, lims):
        self.lims = lims
        self.n_bins = len(lims) - 1

    def _get_masks(self, X, recon=False):
        masks = [
            np.squeeze((X >= lower_lim) & (X < upper_lim))
            for lower_lim, upper_lim in zip(self.lims, self.lims[1:])
        ]
        if not recon:
            return masks, None

        inv_indices = np.empty(X.shape[0], dtype=int)
        inv_indices[np.hstack([np.nonzero(mask)[0] for mask in masks])] = np.arange(
            inv_indices.size
        )
        return masks, inv_indices

    def _apply_mask(self, array, mask):
        if isinstance(array, list):
            return list(itertools.compress(array, mask))

        return array[mask]

    def _apply_masks(self, masks, *args):
        return tuple(
            [tuple(self._apply_mask(array, mask) for mask in masks) for array in args]
        )

    def __call__(self, X, *args, recon=False):
        masks, indices = self._get_masks(X, recon=recon)

        arrays = self._apply_masks(masks, X, *args)
        if not recon:
            return arrays

        return arrays + (indices,)

    def _merge_bins(self, bins):
        is_list = False
        for elems in bins:
            if len(elems) == 0:
                continue

            if isinstance(elems, list):
                is_list = True

            break

        if is_list:
            return list(itertools.chain(*bins))

        return np.concatenate(bins)

    def _sort(self, array, indices):
        if isinstance(array, list):
            return [array[index] for index in indices]

        return array[indices]

    def reconstruct(self, indices, *args):
        out = tuple(
            self._sort(self._merge_bins(array_bins), indices) for array_bins in args
        )
        if len(out) == 1:
            return out[0]

        return out


class TrimesterBinarizer(IntervalBinarizer):
    def __init__(self):
        self.bins = ("pre", "first", "second", "third", "post")
        lims = (-np.inf, 0, 14, 28, 42, np.inf)

        super().__init__(lims=lims)


class StageBinarizer(IntervalBinarizer):
    def __init__(self, merge_pre_preg=False):
        self.bins = ("pre", "preg", "post")
        lims = (-np.inf, 0, 42, np.inf)

        if merge_pre_preg:
            self.bins = self.bins[1:]
            lims = lims[:1] + lims[2:]

        super().__init__(lims=lims)
