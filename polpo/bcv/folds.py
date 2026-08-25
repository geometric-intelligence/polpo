from itertools import product


def compute_held_out_blocks(held_out_rows, held_out_cols):
    return {
        (row_key, col_key): (rows, cols)
        for (row_key, rows), (col_key, cols) in product(
            held_out_rows.items(), held_out_cols.items()
        )
    }
