import numpy as np


class VolumeSlicer:
    def __init__(self, index_ordering=(0, 1, 2), common_size=True):
        self.index_ordering = index_ordering
        self.common_size = common_size

    def _pad_to_common_size(self, slices):
        common_width = max([slice_.shape[0] for slice_ in slices])
        common_height = max([slice_.shape[1] for slice_ in slices])

        padded = []
        for slice_ in slices:
            pad_h = common_height - slice_.shape[0]
            pad_w = common_width - slice_.shape[1]

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            padded.append(
                np.pad(
                    slice_,
                    ((pad_top, pad_bottom), (pad_left, pad_right)),
                    mode="constant",
                )
            )

        return padded

    def take_slices(self, img_fdata, slice_indices=None):
        if slice_indices is None:
            slice_indices = [
                img_fdata.shape[index] // 2 for index in self.index_ordering
            ]

        slices = []
        for index, slice_index in zip(self.index_ordering, slice_indices):
            slicing_indices = [slice(None)] * 3
            slicing_indices[index] = slice_index
            slices.append(img_fdata[tuple(slicing_indices)])

        if self.common_size:
            slices = self._pad_to_common_size(slices)

        return slices
